"""Model definition.

The full model (``NetworkTrafficModel``) consists of three components:
  - TemporalEncoder: Transformer encoder over the session byte matrix.
  - DualGATEncoder (contextual encoder): GAT over the contextual graph.
  - FeatureFusion: cross-attention fusion of the two branches.

``NetworkTrafficTemporal`` and ``NetworkTrafficContextual`` are the
single-branch ablation models.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch_geometric.nn import GATv2Conv, global_mean_pool


class FeatureExtractor(nn.Module):
    """MLP that flattens a [64, 64] byte matrix into a feature vector."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.GELU(),
            nn.Linear(128, out_dim)
        )

    def forward(self, x):
        return self.mlp(x.view(-1, 64 * 64))  # handles the dynamic batch dimension


class DualGATEncoder(nn.Module):
    """Contextual encoder: two GAT branches (packet-length and time features)
    over the contextual graph, followed by a feature enhancement MLP."""

    def __init__(self, in_dim, hidden_dim, heads):
        super().__init__()
        # Feature extractors
        self.packet_extractor = FeatureExtractor(64 * 64, in_dim)
        self.time_extractor = FeatureExtractor(64 * 64, in_dim)

        # GAT branches
        self.packet_gat = nn.ModuleList([
            GATv2Conv(in_dim, hidden_dim, heads),
            GATv2Conv(hidden_dim * heads, hidden_dim, heads=1)
        ])

        self.time_gat = nn.ModuleList([
            GATv2Conv(in_dim, hidden_dim, heads),
            GATv2Conv(hidden_dim * heads, hidden_dim, heads=1)
        ])

        # Feature enhancement
        self.enhancer = nn.Sequential(
            nn.Linear(2 * hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, 2 * hidden_dim)
        )

    def forward(self, packet_feat, time_feat, edge_index):
        x_p = self.packet_extractor(packet_feat)
        for gat in self.packet_gat:
            x_p = F.elu(gat(x_p, edge_index))

        x_t = self.time_extractor(time_feat)
        for gat in self.time_gat:
            x_t = F.elu(gat(x_t, edge_index))

        return self.enhancer(torch.cat([x_p, x_t], dim=1))


class TemporalEncoder(nn.Module):
    """Transformer encoder over the session byte matrix."""

    def __init__(self, feat_dim, num_heads, num_layers, dropout=0.1):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(64, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, feat_dim)
        )
        # Positional encoding compatible with PyG batch processing
        self.pos_encoder = PositionalEncoding(feat_dim, dropout)

        self.transformer = TransformerEncoder(
            TransformerEncoderLayer(
                d_model=feat_dim,
                nhead=num_heads,
                dim_feedforward=4 * feat_dim,
                batch_first=True,
                dropout=dropout
            ), num_layers=num_layers
        )

    def forward(self, matrix, mask):
        key_padding_mask = (mask.sum(dim=-1) == 64)  # [batch, 64] True where a packet is fully padding

        batch_size, seq_len, byte_len = matrix.shape
        x = self.projection(matrix.view(-1, byte_len))  # [batch_size * seq_len, feat_dim]
        x = x.view(batch_size, seq_len, -1)             # [batch_size, seq_len, feat_dim]

        x = x * math.sqrt(self.pos_encoder.d_model)
        x = self.pos_encoder(x.permute(1, 0, 2))        # [seq_len, batch, feat_dim]
        x = x.permute(1, 0, 2)                          # [batch, seq_len, feat_dim]
        return self.transformer(x, src_key_padding_mask=key_padding_mask)


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model, dropout=0.1, max_len=512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(0)
        x = x + self.pe[:seq_len, :].unsqueeze(1)
        return self.dropout(x)


class FeatureFusion(nn.Module):
    """Cross-attention fusion: the graph-level contextual feature attends to
    the temporal feature sequence, and the result is concatenated with the
    contextual feature and projected."""

    def __init__(self, gat_dim, trans_dim, num_heads):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=trans_dim,
            kdim=gat_dim,
            vdim=gat_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.fusion = nn.Sequential(
            nn.Linear(trans_dim + gat_dim, 4 * trans_dim),
            nn.GELU(),
            nn.Linear(4 * trans_dim, trans_dim)
        )

    def forward(self, trans_feat, gat_feat):
        # trans_feat: [B, T, D_t] temporal feature sequence
        # gat_feat:   [B, D_g]    graph-level contextual feature
        gat_expanded = gat_feat.unsqueeze(1)  # [B, 1, D_g]
        attn_out, _ = self.cross_attn(
            query=gat_expanded,
            key=trans_feat,
            value=trans_feat
        )
        return self.fusion(torch.cat([attn_out.squeeze(1), gat_feat], dim=1))


class NetworkTrafficModel(nn.Module):
    """Full model: contextual encoder + temporal encoder + cross-attention fusion."""

    def __init__(self, gat_in_dim=64, gat_hidden_dim=128, gat_heads=4,
                 temporal_feat_dim=256, temporal_num_heads=4, temporal_num_layers=2,
                 fusion_gat_dim=256, fusion_trans_dim=256, fusion_num_heads=4,
                 classifier_in_dim=256, classifier_hidden_dim=128, num_classes=13,
                 dropout=0.3):
        super().__init__()
        self.gat_encoder = DualGATEncoder(in_dim=gat_in_dim, hidden_dim=gat_hidden_dim, heads=gat_heads)
        self.temporal_encoder = TemporalEncoder(feat_dim=temporal_feat_dim, num_heads=temporal_num_heads,
                                                num_layers=temporal_num_layers)
        self.fusion = FeatureFusion(gat_dim=fusion_gat_dim, trans_dim=fusion_trans_dim,
                                    num_heads=fusion_num_heads)
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in_dim, classifier_hidden_dim),
            nn.LayerNorm(classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, num_classes)
        )

    def forward(self, data, visualize=False):
        # Contextual branch
        # x_packet / x_time: [total context sessions in the batch, 64, 64]
        # gat_nodes: [total context sessions in the batch, 256]
        gat_nodes = self.gat_encoder(data.x_packet, data.x_time, data.edge_index)
        gat_graph = global_mean_pool(gat_nodes, data.batch)  # [B, 256]

        # Temporal branch
        # main_matrix: [B, 64, 64] -> trans_feat: [B, 64, 256]
        trans_feat = self.temporal_encoder(data.main_matrix, data.main_mask)

        # Fusion and classification
        fused = self.fusion(trans_feat, gat_graph)  # [B, 256]
        results = self.classifier(fused)

        if visualize:  # expose intermediate features, e.g. for t-SNE
            return results, gat_graph, trans_feat.mean(dim=1), fused
        return results


class NetworkTrafficTemporal(nn.Module):
    """Ablation model: temporal branch only (no context)."""

    def __init__(self, gat_in_dim=64, gat_hidden_dim=128, gat_heads=4,
                 temporal_feat_dim=256, temporal_num_heads=4, temporal_num_layers=2,
                 fusion_gat_dim=256, fusion_trans_dim=256, fusion_num_heads=4,
                 classifier_in_dim=256, classifier_hidden_dim=128, num_classes=13,
                 dropout=0.3):
        super().__init__()
        self.temporal_encoder = TemporalEncoder(feat_dim=temporal_feat_dim, num_heads=temporal_num_heads,
                                                num_layers=temporal_num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in_dim, classifier_hidden_dim),
            nn.LayerNorm(classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, num_classes)
        )

    def forward(self, data):
        trans_feat = self.temporal_encoder(data.main_matrix, data.main_mask)  # [B, 64, 256]
        trans_feat = trans_feat.mean(dim=1)  # [B, 256]
        return self.classifier(trans_feat)


class NetworkTrafficContextual(nn.Module):
    """Ablation model: contextual branch only (no temporal features)."""

    def __init__(self, gat_in_dim=64, gat_hidden_dim=128, gat_heads=4,
                 temporal_feat_dim=256, temporal_num_heads=4, temporal_num_layers=2,
                 fusion_gat_dim=256, fusion_trans_dim=256, fusion_num_heads=4,
                 classifier_in_dim=256, classifier_hidden_dim=128, num_classes=13,
                 dropout=0.3):
        super().__init__()
        self.gat_encoder = DualGATEncoder(in_dim=gat_in_dim, hidden_dim=gat_hidden_dim, heads=gat_heads)
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in_dim, classifier_hidden_dim),
            nn.LayerNorm(classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, num_classes)
        )

    def forward(self, data):
        gat_nodes = self.gat_encoder(data.x_packet, data.x_time, data.edge_index)
        gat_graph = global_mean_pool(gat_nodes, data.batch)  # [B, 256]
        return self.classifier(gat_graph)
