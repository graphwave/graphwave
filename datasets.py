import os

import numpy as np
from torch.utils.data import Dataset

from build_graph import build_graph_with_context


class TrafficDataset(Dataset):
    """Loads raw session records from a ``*_contextual.npy`` file and builds
    a contextual graph for every valid session in advance."""

    def __init__(self, file_path, dataset):
        raw_data = np.load(file_path, allow_pickle=True)
        self.graphs = []

        for idx, item in enumerate(raw_data):
            try:
                graph = build_graph_with_context(item, dataset)
                self.graphs.append(graph)
            except Exception as e:
                print(f"Error in {os.path.basename(file_path)} item {idx}: {e}")

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]
