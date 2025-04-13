import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch_geometric.data import Batch
import math
import time
class FeatureExtractor(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.GELU(),
            nn.Linear(128, out_dim)
        )
    
    def forward(self, x):
        return self.mlp(x.view(-1, 64*64)) 

class DualGATEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, heads):
        super().__init__()
        self.packet_extractor = FeatureExtractor(64*64, in_dim)
        self.time_extractor = FeatureExtractor(64*64, in_dim)
        
        self.packet_gat = nn.ModuleList([
            GATv2Conv(in_dim, hidden_dim, heads),
            GATv2Conv(hidden_dim*heads, hidden_dim, heads=1)
        ])
        
        self.time_gat = nn.ModuleList([
            GATv2Conv(in_dim, hidden_dim, heads),
            GATv2Conv(hidden_dim*heads, hidden_dim, heads=1)
        ])
        
        self.enhancer = nn.Sequential(
            nn.Linear(2*hidden_dim, 4*hidden_dim),
            nn.GELU(),
            nn.Linear(4*hidden_dim, 2*hidden_dim)
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
    def __init__(self, feat_dim, num_heads, num_layers, dropout=0.1):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(64, 256),
            nn.BatchNorm1d(256), 
            nn.ReLU(),
            nn.Linear(256, feat_dim)
        )
        self.pos_encoder = PositionalEncoding(feat_dim, dropout)
        
        self.transformer = TransformerEncoder(
            TransformerEncoderLayer(
                d_model=feat_dim,
                nhead=num_heads,
                dim_feedforward=4*feat_dim,
                batch_first=True,
                dropout=dropout
            ), num_layers=num_layers
        )
    def _init_weights(self):
        def init(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        self.projection.apply(init)
    def forward(self, matrix, mask):
        if torch.isnan(matrix).any() or torch.isinf(matrix).any():
            print("Input matrix contains NaN/Inf")
            matrix = torch.nan_to_num(matrix) 
        assert torch.isfinite(matrix).all(), "Input matrix contains NaN/Inf"

        if torch.isnan(mask).any() or torch.isinf(mask).any():
            print("Input mask contains NaN/Inf")
            mask = torch.nan_to_num(mask)  
        assert torch.isfinite(mask).all(), "Input mask contains NaN/Inf"

        key_padding_mask = (mask.sum(dim=-1) == 64) 
        assert torch.isfinite(key_padding_mask).all(), "Input mask contains NaN/Inf"
        
        batch_size, seq_len, byte_len = matrix.shape
        matrix = matrix.view(-1, byte_len) 
        x = self.projection(matrix) 
        x = x.view(batch_size, seq_len, -1)
        

        assert torch.isfinite(x).all(), "Projection output contains NaN/Inf"

        x = x * math.sqrt(self.pos_encoder.d_model)
        x = x.permute(1, 0, 2) 
        x = self.pos_encoder(x)
        assert torch.isfinite(x).all(), "Pos output contains NaN/Inf"
        x = x.permute(1, 0, 2) 
        out = self.transformer(x, src_key_padding_mask=key_padding_mask)
        assert torch.isfinite(out).all(), "Transformer output contains NaN/Inf"
        return out

class PositionalEncoding(nn.Module):
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


class FeatureFusion_gat(nn.Module):
    def __init__(self, gat_dim, trans_dim, num_heads):
        super().__init__()
        self.norm_trans = nn.LayerNorm(trans_dim)
        self.norm_gat = nn.LayerNorm(gat_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=trans_dim,
            kdim=gat_dim,
            vdim=gat_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.fusion = nn.Sequential(
            nn.Linear(trans_dim + gat_dim, 4*trans_dim),
            nn.GELU(),
            nn.Linear(4*trans_dim, trans_dim)
        )

    def forward(self, trans_feat, gat_feat):
        assert torch.isfinite(trans_feat).all(), "trans_feat NaN/Inf"
        assert torch.isfinite(gat_feat).all(), "gat_feat NaN/Inf"
        gat_expanded = gat_feat.unsqueeze(1)  # [B, 1, D_g]
        attn_out, _ = self.cross_attn(
            query=gat_expanded,
            key=trans_feat,
            value=trans_feat
        )
        assert not torch.isinf(attn_out).any(), "attn_out has inf!"
        return self.fusion(torch.cat([attn_out.squeeze(1), gat_feat], dim=1))

class FeatureFusion_trans_mean(nn.Module):
    def __init__(self, gat_dim, trans_dim, num_heads):
        super().__init__()
        self.norm_trans = nn.LayerNorm(trans_dim)
        self.norm_gat = nn.LayerNorm(gat_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=trans_dim,
            kdim=gat_dim,
            vdim=gat_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.fusion = nn.Sequential(
            nn.Linear(trans_dim + gat_dim, 4*trans_dim),
            nn.GELU(),
            nn.Linear(4*trans_dim, trans_dim)
        )

    def forward(self, trans_feat, gat_feat):
        assert not torch.isnan(trans_feat).any(), "trans_feat has nan!"
        assert not torch.isinf(gat_feat).any(), "gat_feat has inf!"

        gat_expanded = gat_feat.unsqueeze(1)  # [B, 1, D_g]
        attn_out, _ = self.cross_attn(
            query=trans_feat,
            key=gat_expanded,
            value=gat_expanded
        )
        attn_avg = attn_out.mean(dim=1)
        assert not torch.isnan(attn_avg).any(), "attn_avg has NaN!"
        trans_feat_avg = trans_feat.mean(dim=1)
        assert not torch.isnan(trans_feat_avg).any(), "trans_feat_avg has NaN!"
        combined = torch.cat([attn_avg, trans_feat_avg], dim=1)
        return self.fusion(combined)

class FeatureFusion_trans_flatten(nn.Module):
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
            nn.Linear(64*trans_dim + 64*gat_dim, 128*trans_dim),
            nn.GELU(),
            nn.Linear(128*trans_dim, trans_dim)
        )

    def forward(self, trans_feat, gat_feat):
        assert not torch.isnan(trans_feat).any(), "trans_feat has nan!"
        assert not torch.isinf(gat_feat).any(), "gat_feat has inf!"

        gat_expanded = gat_feat.unsqueeze(1)
        attn_out, _ = self.cross_attn(
            query=trans_feat,
            key=gat_expanded,
            value=gat_expanded
        )
        attn_flatten = attn_out.reshape(attn_out.size(0), -1)
        assert not torch.isnan(attn_flatten).any(), "attn_flatten has NaN!"

        trans_feat_flatten = trans_feat.reshape(trans_feat.size(0), -1)
        assert not torch.isnan(trans_feat_flatten).any(), "trans_feat_flatten has NaN!"
        combined = torch.cat([attn_flatten, trans_feat_flatten], dim=1)
        return self.fusion(combined)


class FeatureFusion_transformer(nn.Module):
    def __init__(self,  gat_dim, trans_dim, num_heads, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=trans_dim,
            kdim=gat_dim,
            vdim=gat_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout
        )
        
        self.norm1 = nn.LayerNorm(trans_dim)
        self.norm2 = nn.LayerNorm(trans_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(trans_dim, 4 * trans_dim),
            nn.GELU(),
            nn.Linear(4 * trans_dim, trans_dim),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, trans_feat, gat_feat):

        gat_expanded = gat_feat.unsqueeze(1)
        attn_output, _ = self.cross_attn(
            query=trans_feat, 
            key=gat_expanded, 
            value=gat_expanded 
        )

        trans_feat = trans_feat + self.dropout(attn_output)
        ffn_output = self.ffn(trans_feat)
        trans_feat = trans_feat + self.dropout(ffn_output)
        return trans_feat.reshape(trans_feat.size(0), -1) 
























class NetworkTrafficModel(nn.Module):
    def __init__(self, gat_in_dim=64, gat_hidden_dim=128, gat_heads=4,
                 temporal_feat_dim=256, temporal_num_heads=4, temporal_num_layers=2,
                 fusion_gat_dim=256, fusion_trans_dim=256, fusion_num_heads=4,
                 classifier_in_dim=256, classifier_hidden_dim=128, num_classes=13,
                 dropout=0.3):
        super().__init__()
        self.gat_encoder = DualGATEncoder(in_dim=gat_in_dim, hidden_dim=gat_hidden_dim, heads=gat_heads)
        self.temporal_encoder = TemporalEncoder(feat_dim=temporal_feat_dim, num_heads=temporal_num_heads,
                                                num_layers=temporal_num_layers)
        self.fusion = FeatureFusion_gat(gat_dim=fusion_gat_dim, trans_dim=fusion_trans_dim, num_heads=fusion_num_heads)
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in_dim, classifier_hidden_dim),
            nn.LayerNorm(classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, num_classes)
        )

    def forward(self, data, visualize=False):
        # 图特征编码
        starttime = time.time()
        gat_nodes = self.gat_encoder(data.x_packet, data.x_time, data.edge_index)
        endtime = time.time()
        #print(f'gat embedding time is: {(endtime-starttime)/len(data)}')
        #x_packet[batch_size个样本中总上下文会话数量, 64, 64];x_time[batch_size个样本中总上下文会话数量, 64, 64];
        #gat_nodes[batch_size个样本中总上下文会话数量, 256]
        gat_graph = global_mean_pool(gat_nodes, data.batch)  # [B, 256]
        #gat_graph[batch_size, 256]
        # 时序特征编码
        starttime = time.time()
        trans_feat = self.temporal_encoder(data.main_matrix, data.main_mask)#main_matrix[batch_size, 64, 64]
        endtime = time.time()
        #print(f'temporal embedding time is: {(endtime-starttime)/len(data)}')
        #trans_feat[batch_size, 64, 256]
        #trans_feat = trans_feat.mean(dim=1)  # [B, 256]
        
        # 特征融合
        starttime = time.time()
        fused = self.fusion(trans_feat, gat_graph)#fused[batch_size, 256]
        endtime = time.time()
        #print(f'fusion time is: {(endtime-starttime)/len(data)}')
        starttime = time.time()
        results = self.classifier(fused)
        endtime = time.time()
        #print(f'classification time is: {(endtime-starttime)/len(data)}')
        #为了说明特征的重要度
        if visualize:
            return results, gat_graph, trans_feat.mean(dim=1), fused
        return results#[batch_size, 13]
    

#用于消融实验，去掉上下文，只用当前的特征
class NetworkTrafficTemporal(nn.Module):
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
        # 时序特征编码
        trans_feat = self.temporal_encoder(data.main_matrix, data.main_mask)#main_matrix[batch_size, 64, 64]
        #trans_feat[batch_size, 64, 256]
        trans_feat = trans_feat.mean(dim=1)  # [B, 256]
        return self.classifier(trans_feat)#[batch_size, 13]
    
#用于消融实验，去掉当前特征，只用上下文特征来分类
class NetworkTrafficContextual(nn.Module):
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
        # 图特征编码
        gat_nodes = self.gat_encoder(data.x_packet, data.x_time, data.edge_index)
        #x_packet[batch_size个样本中总上下文会话数量, 64, 64];x_time[batch_size个样本中总上下文会话数量, 64, 64];
        #gat_nodes[batch_size个样本中总上下文会话数量, 256]
        gat_graph = global_mean_pool(gat_nodes, data.batch)  # [B, 256]
        #gat_graph[batch_size, 256]
        return self.classifier(gat_graph)#[batch_size, 13]
    



class NetworkTrafficUnknown(nn.Module):
    def __init__(self, gat_in_dim=64, gat_hidden_dim=128, gat_heads=4,
                 temporal_feat_dim=256, temporal_num_heads=4, temporal_num_layers=2,
                 fusion_gat_dim=256, fusion_trans_dim=256, fusion_num_heads=4,
                 classifier_in_dim=256, classifier_hidden_dim=128, num_classes=13,
                 dropout=0.3):
        super().__init__()
        self.gat_encoder = DualGATEncoder(in_dim=gat_in_dim, hidden_dim=gat_hidden_dim, heads=gat_heads)
        self.temporal_encoder = TemporalEncoder(feat_dim=temporal_feat_dim, num_heads=temporal_num_heads,
                                                num_layers=temporal_num_layers)
        self.fusion = FeatureFusion_gat(gat_dim=fusion_gat_dim, trans_dim=fusion_trans_dim, num_heads=fusion_num_heads)

    def forward(self, data, visualize=False):
        # 图特征编码
        gat_nodes = self.gat_encoder(data.x_packet, data.x_time, data.edge_index)
        gat_graph = global_mean_pool(gat_nodes, data.batch)

        # 时序特征编码
        trans_feat = self.temporal_encoder(data.main_matrix, data.main_mask)

        # 特征融合
        fused = self.fusion(trans_feat, gat_graph)
        return fused
    