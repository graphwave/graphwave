import torch
import numpy as np
from torch.utils.data import Dataset, random_split
from build_graph import build_graph_with_context
import os
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

class TrafficDataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        self.raw_data = None
        self.valid_indices = []
        self.error_indices = []

        raw_data = np.load(file_path, allow_pickle=True)
        for idx, item in enumerate(raw_data):
            try:
                _ = build_graph_with_context(item)
                self.valid_indices.append(idx)
            except Exception as e:
                self.error_indices.append(idx)
                print(f"Error in {os.path.basename(file_path)} item {idx}: {e}")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        if self.raw_data is None:
            self.raw_data = np.load(self.file_path, allow_pickle=True)
        actual_idx = self.valid_indices[idx]
        item = self.raw_data[actual_idx]
        try:
            graph = build_graph_with_context(item)
            return graph
        except Exception as e:
            print(f"Unexpected error in item {actual_idx}: {e}")
            raise