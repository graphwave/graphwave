import numpy as np
import torch
from torch_geometric.data import Data
from collections import OrderedDict, defaultdict
from config import get_mapping,get_args

args = get_args()
mapping, num_classes = get_mapping(args.dataset)
class CustomData(Data):
    def __cat_dim__(self, key, value, *args, **kwargs):

        if key in ['main_matrix', 'main_mask']:
            return 0  
        elif key == 'current_idx':
            return None 
        return super().__cat_dim__(key, value, *args, **kwargs)
    
    def __inc__(self, key, value, *args, **kwargs):

        if key == 'current_idx':
            return self.num_nodes  
        return super().__inc__(key, value, *args, **kwargs)


def build_graph_with_context(result):
 
    if not result.get('contextual', []):
        raise ValueError("Empty context data, unable to build graph structure")
    context_dict = OrderedDict()
    for ctx in result['contextual']:
        key = ctx['five_tuple']
        context_dict[key] = ctx
    

    all_sessions = list(context_dict.values())
    sorted_sessions = sorted(all_sessions, key=lambda x: x['start_time'])
    current_key = result['five_tuple']

    current_idx = next((i for i, sess in enumerate(sorted_sessions) if sess['five_tuple'] == current_key), -1)

    packet_features = []
    time_features = []
    

    packet_lens_list = [] 
    time_intervals_list = []

    

    for session in sorted_sessions:
        packet_features.append(torch.tensor(session['lens_spectrogram'], dtype=torch.float32))
        time_features.append(torch.tensor(session['intervals_spectrogram'], dtype=torch.float32))
        if(args.wavelet != 'yes'):
            packet_lens = torch.tensor(session['packet_lens'], dtype=torch.float32)
            packet_lens_matrix = packet_lens.unsqueeze(0).repeat(64, 1) 
            packet_lens_list.append(packet_lens_matrix)
            time_intervals = torch.tensor(session['time_intervals'], dtype=torch.float32)
            time_intervals_matrix = time_intervals.unsqueeze(0).repeat(64, 1)
            time_intervals_list.append(time_intervals_matrix)
    

    ip_to_indices = defaultdict(list)  
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
        raise ValueError("The generated graph has no nodes")
    if(args.wavelet == 'yes'):
        return CustomData(
            x_packet=torch.stack(packet_features),  
            x_time=torch.stack(time_features),     
            edge_index=torch.tensor(edge_list, dtype=torch.long).t().contiguous() if edge_list else torch.empty((2,0), dtype=torch.long),
     
            current_idx=torch.tensor([current_idx], dtype=torch.long),
            y=torch.tensor(mapping[result['class_name']], dtype=torch.long),
            main_matrix=torch.tensor(result['matrix'], dtype=torch.float32).unsqueeze(0),
            main_mask=torch.tensor(result['padding_mask'], dtype=torch.bool).unsqueeze(0),
            batch=torch.zeros(len(packet_features), dtype=torch.long)  
        )
    else:
        return CustomData(
            x_packet=torch.stack(packet_lens_list),  # [num_nodes, 64, 64]
            x_time=torch.stack(time_intervals_list),      # [num_nodes, 64, 64]
            edge_index=torch.tensor(edge_list, dtype=torch.long).t().contiguous() if edge_list else torch.empty((2,0), dtype=torch.long),
       
            current_idx=torch.tensor([current_idx], dtype=torch.long),
            y=torch.tensor(mapping[result['class_name']], dtype=torch.long),
            main_matrix=torch.tensor(result['matrix'], dtype=torch.float32).unsqueeze(0),#
            main_mask=torch.tensor(result['padding_mask'], dtype=torch.bool).unsqueeze(0),
            batch=torch.zeros(len(packet_features), dtype=torch.long) 
        )
