"""Contextual graph construction.

Each session is represented as a graph whose nodes are the context sessions
(the current session itself plus neighboring sessions that share an IP
address). Node features are the wavelet spectrograms of the packet-length and
time-interval sequences; edges connect sessions that share an IP address,
following temporal order.
"""

from collections import OrderedDict, defaultdict

import torch
from torch_geometric.data import Data

from config import get_args, get_mapping


class CustomData(Data):
    """PyG Data with custom batching rules for the 3-D matrices."""

    def __cat_dim__(self, key, value, *args, **kwargs):
        # Stack the 3-D matrices along the batch dimension
        if key in ['main_matrix', 'main_mask']:
            return 0
        elif key == 'current_idx':
            return None  # keep per-sample, no concatenation
        return super().__cat_dim__(key, value, *args, **kwargs)

    def __inc__(self, key, value, *args, **kwargs):
        # Offset current_idx by the node count of each graph so that
        # indices stay valid after batching
        if key == 'current_idx':
            return self.num_nodes
        return super().__inc__(key, value, *args, **kwargs)


def build_graph_with_context(result, dataset):
    """Build the contextual graph of a session.

    Args:
        result (dict): the raw record of the current session, containing its
            five-tuple, class name, byte matrix, padding mask, and context
            sessions (each with spectrograms and sequence features).
        dataset (str): dataset name used to look up the label mapping.

    Returns:
        CustomData: a single graph ready for batching by PyG.
    """
    args = get_args()
    mapping, num_classes = get_mapping(dataset)

    # Deduplicate context sessions by five-tuple while keeping their order
    if not result.get('contextual', []):
        raise ValueError("Empty contextual data, cannot build graph")
    context_dict = OrderedDict()
    for ctx in result['contextual']:
        context_dict[ctx['five_tuple']] = ctx

    # Sort context sessions by start time and locate the current session
    sorted_sessions = sorted(context_dict.values(), key=lambda x: x['start_time'])
    context_five_tuples = [s['five_tuple'] for s in sorted_sessions]
    current_key = result['five_tuple']
    current_idx = next((i for i, sess in enumerate(sorted_sessions)
                        if sess['five_tuple'] == current_key), -1)

    # Node features
    packet_features = []
    time_features = []
    # For the no-wavelet ablation, raw packet-length / time-interval
    # sequences broadcast to [64, 64] are used instead
    packet_lens_list = []
    time_intervals_list = []

    for session in sorted_sessions:
        packet_features.append(torch.tensor(session['lens_spectrogram'], dtype=torch.float32))
        time_features.append(torch.tensor(session['intervals_spectrogram'], dtype=torch.float32))
        if args.wavelet != 'yes':
            packet_lens = torch.tensor(session['packet_lens'], dtype=torch.float32)
            packet_lens_list.append(packet_lens.unsqueeze(0).repeat(64, 1))
            time_intervals = torch.tensor(session['time_intervals'], dtype=torch.float32)
            time_intervals_list.append(time_intervals.unsqueeze(0).repeat(64, 1))

    # Build directed edges: each session connects to previous sessions that
    # share the same source or destination IP
    ip_to_indices = defaultdict(list)  # node indices seen so far, per IP
    edge_set = set()
    for idx, session in enumerate(sorted_sessions):
        ips = {session['src_ip'], session['dst_ip']}
        for ip in ips:
            for old_idx in ip_to_indices[ip]:
                edge_set.add((old_idx, idx))
            ip_to_indices[ip].append(idx)
    edge_list = list(edge_set)

    num_nodes = len(packet_features)
    if num_nodes == 0:
        raise ValueError("Generated graph has no nodes")

    if args.wavelet == 'yes':
        x_packet = torch.stack(packet_features)  # [num_nodes, 64, 64]
        x_time = torch.stack(time_features)      # [num_nodes, 64, 64]
    else:
        x_packet = torch.stack(packet_lens_list)
        x_time = torch.stack(time_intervals_list)

    return CustomData(
        x_packet=x_packet,
        x_time=x_time,
        # edge_index is a [2, N] tensor: row 0 = source node, row 1 = target node
        edge_index=torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        if edge_list else torch.empty((2, 0), dtype=torch.long),
        current_idx=torch.tensor([current_idx], dtype=torch.long),
        y=torch.tensor(mapping[result['class_name']], dtype=torch.long),
        # unsqueeze(0) adds a batch dimension so that the matrices stack correctly
        main_matrix=torch.tensor(result['matrix'], dtype=torch.float32).unsqueeze(0),
        main_mask=torch.tensor(result['padding_mask'], dtype=torch.bool).unsqueeze(0),
        batch=torch.zeros(num_nodes, dtype=torch.long),
        five_tuple=current_key,
        graph_nodes=torch.tensor([num_nodes], dtype=torch.long),
        context_five_tuples=context_five_tuples
    )

# Notes on PyG batching:
# The PyG DataLoader merges all graphs of a batch into one large disconnected
# graph, offsetting node and edge indices automatically. For a batch with two
# graphs of n1 and n2 nodes:
#   - x_packet has shape [n1 + n2, 64, 64]
#   - edge_index of graph 2 is shifted by n1, so no cross-graph edges appear
#   - the Batch object carries a `batch` attribute marking which sub-graph
#     each node belongs to, e.g. [0, ..., 0, 1, ..., 1]
# GAT/GCN layers naturally support a dynamic number of nodes through
# edge_index, and pooling layers such as global_mean_pool aggregate dynamic
# node features into graph-level representations.
