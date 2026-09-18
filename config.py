"""Dataset label mappings and command-line arguments."""

mapping_binary = {
    'Benign': 0,
    'Malicious': 1
}

mapping_botnet = {
    'Benign': 0,
    'Blackhole': 1,
    'IRC': 2,
    'Menti': 3,
    'Murlo': 4,
    'Neris': 5,
    'RBot': 6,
    'Smoke-bot': 7,
    'Sogou': 8,
    'TBot': 9,
    'Virut': 10,
    'Weasel': 11,
    'Zero-access': 12,
    'Zeus': 13
}


# Central registry of all dataset mappings
DATASET_MAPPINGS = {
    "botnet2014": mapping_botnet,
    "binary": mapping_binary
}


from typing import Dict, Tuple


def get_mapping(dataset_name: str) -> Tuple[Dict, int]:
    """Get the label mapping and the number of classes of a dataset.

    Args:
        dataset_name (str): Dataset name (case-insensitive), see ``DATASET_MAPPINGS``.

    Returns:
        tuple[dict, int]: (label mapping, number of classes)

    Raises:
        ValueError: If the dataset name is unknown.
    """
    dataset_name = dataset_name.lower()

    if dataset_name not in DATASET_MAPPINGS:
        available = list(DATASET_MAPPINGS.keys())
        raise ValueError(f"Invalid dataset name: {dataset_name}. Available: {available}")

    mapping = DATASET_MAPPINGS[dataset_name]
    num_classes = len(mapping)

    return mapping, num_classes


import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Training arguments')
    parser.add_argument('--data_path', type=str,
                        default='middleResults/temporal_contextual_data/botnet2014',
                        required=False,
                        help='Directory that contains the generated *_contextual.npy files')
    parser.add_argument('--dataset', type=str, default='botnet2014', required=False,
                        help='Dataset name, see DATASET_MAPPINGS in this file')
    parser.add_argument('--contextual', type=str, default='no', required=False,
                        help="'yes': use the contextual (GAT) branch only (ablation)")
    parser.add_argument('--temporal', type=str, default='no', required=False,
                        help="'yes': use the temporal (Transformer) branch only (ablation); "
                             "when both flags are 'no', the full cross-attention fusion model is used")
    parser.add_argument('--wavelet', type=str, default='yes', required=False,
                        help="'yes': use wavelet spectrograms as node features, otherwise raw "
                             "packet-length / time-interval sequences (ablation)")
    parser.add_argument('--tsne', type=str, default='no', required=False,
                        help="'yes': additionally render t-SNE plots of the three feature spaces")
    args = parser.parse_args()
    return args
