mapping_botnet = {
        'Benign':0,
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
        'Weasel':11,
        'Zero-access':12,
        'Zeus':13
        
    }

mapping_IDS2017 = {
        'Benign':0,
        'Bot': 1,
        'BruteForce': 2,
        'DDoS': 3,
        'FTP-Patator': 4,
        'Infiltration': 5,
        'Slowhttptest': 6,
        'Slowloris': 7,
        'SqlInjection': 8,
        'SSH-Patator': 9,
        'XSS':10
    }

mapping_USTC = {
        'Benign':0,
        'Cridex':1,
        'Geodo': 2,
        'Htbot': 3,
        'Miuref': 4,
        'Neris': 5,
        'Nsis-ay': 6,
        'Shifu': 7,
        'Tinba': 8,
        'Virut': 9,
        'Zeus': 10
    }

mapping_dos2017 = {
        'ddossim':0,
        'goldeneye': 1,
        'hulk': 2,
        'rudy-selected': 3,
        'slowbody2': 4,
        'slowheaders': 5,
        'slowloris': 6,
        'slowread': 7
    }

mapping_apt2024 = {
        'benign':0,
        'attack': 1
    }

mapping_aciiot = {
        'benign':0,
        'bruteforce': 1,
        'dos':2,
        'recon':3
    }

mapping_andmal2017 = {
        'Benign':0,
        'Dowgin': 1,
        'Ewind':2,
        'Feiwo':3,
        'Gooligan':4,
        'Kemoge':5,
        'koodous':6,
        'Mobidash':7,
        'Selfmite':8,
        'Shuanet':9,
        'Youmi':10
    }

mapping_unsw = {
        'normal':0,
        'worms': 1,
        'shellcode':2,
        'reconnaissance':3,
        'generic':4,
        'fuzzers':5,
        'exploits':6,
        'dos':7,
        'backdoor':8,
        'analysis':9
    }


DATASET_MAPPINGS = {
    "botnet2014": mapping_botnet,
    "ids2017": mapping_IDS2017,
    "ustc": mapping_USTC,
    "dos2017": mapping_dos2017,
    "apt2024": mapping_apt2024,
    "aciiot":mapping_aciiot,
    "andmal2017":mapping_andmal2017,
    "unsw":mapping_unsw
}


from typing import Dict, Tuple

def get_mapping(dataset_name: str) -> Tuple[Dict, int]:
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
    parser.add_argument('--data_path', type=str, default='/XXXX/botnet2014/', required=False)
    parser.add_argument('--dataset', type=str, default='botnet2014', required=False)
    parser.add_argument('--contextual', type=str, default = 'no', required=False)#
    parser.add_argument('--temporal', type=str, default = 'no', required=False)#
    parser.add_argument('--wavelet', type=str, default = 'yes', required=False)#
    parser.add_argument('--tsne', type=str, default = 'no', required=False)
    args = parser.parse_args()
    return args