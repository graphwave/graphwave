
import argparse
import binascii
import json
import os
import time
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt
from scapy.all import sniff, IP, TCP, UDP, ICMP
from utils import get_contents_in_dir
from nfstream import NFStreamer
from scapy.all import sniff
from scapy.layers.inet import IP
import scapy.all as scapy
from scapy.utils import rdpcap
import pickle

def merge_pcaps(pcaps_dir, pcap_save_path, max_batch_size=200):

    pcaps = get_contents_in_dir(pcaps_dir, ['.'], ['.pcap', '.pcapng'])
    
    cmd_base = 'mergecap -F pcap -w '
    
    temp_pcaps = []
    
    while len(pcaps) > 0:
        batch = pcaps[:max_batch_size] 
        pcaps = pcaps[max_batch_size:] 

        temp_pcap = f'{pcaps_dir}/temp_{len(temp_pcaps)}.pcap'
        temp_pcaps.append(temp_pcap)
        
        cmd = cmd_base + temp_pcap + ' ' + ' '.join(batch)
        ret = os.system(cmd)
        if ret == 0:
            print(f'Merged batch of {len(batch)} pcaps into {temp_pcap} successfully.')
        else:
            print(f'Error merging batch of pcaps: {batch}')
            exit(1)
    
    while len(temp_pcaps) > max_batch_size:
        batch = temp_pcaps[:max_batch_size]
        temp_pcaps = temp_pcaps[max_batch_size:]
        
        temp_final_pcap = f'{pcaps_dir}/final_temp_{len(temp_pcaps)}.pcap'
        cmd = cmd_base + temp_final_pcap + ' ' + ' '.join(batch)
        ret = os.system(cmd)
        if ret == 0:
            print(f'Merged batch of temporary files into {temp_final_pcap} successfully.')
            temp_pcaps.append(temp_final_pcap)
        else:
            print(f'Error merging batch of temp pcaps: {batch}')
            exit(1)
    
    final_cmd = cmd_base + pcap_save_path + ' ' + ' '.join(temp_pcaps)
    ret = os.system(final_cmd)
    if ret == 0:
        print(f'Merged all temp pcaps into final file {pcap_save_path} successfully.')
    else:
        print(f'Error merging temp pcaps into final file: {pcap_save_path}')
        exit(1)

    for temp_pcap in temp_pcaps:
        try:
            os.remove(temp_pcap)
            print(f'Removed temporary pcap file {temp_pcap}.')
        except Exception as e:
            print(f'Error removing temporary file {temp_pcap}: {e}')




def filter_protocols_in_pcap(pcap_path):
    display_filter = "not (arp or dhcp) and (tcp or udp)"
    pcap_dir, pcap_name = os.path.split(pcap_path)
    pcap_name = os.path.splitext(pcap_name)[0]
    out_path = os.path.join(pcap_dir, pcap_name)
    cmd = f'tshark -F pcap -r {pcap_path} -w {out_path}_tmp.pcap -Y "{display_filter}"'
    ret = os.system(cmd)
    if ret == 0:
        print(f'filter protocols with display filter {display_filter} in pcap successfully')
    else:
        print(f'filter protocols in pcap error')
        exit(1)
    os.system(f'rm -f {pcap_path}')
    os.system(f'mv {out_path}_tmp.pcap {out_path}.pcap')


def split_pcap_to_sessions(pcap_path, save_dir):
    if os.path.exists(save_dir):
        os.system(f'rm -rf {save_dir}')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    ret = os.system(f'editcap -F pcap {pcap_path} - | mono SplitCap.exe -r - -s session -o {save_dir} -p 1000')
    if ret == 0:
        print(f'split {pcap_path} to sessions successfully')
    else:
        print(f'split {pcap_path} error')
        exit(1)



def parse_session_pcap_to_matrix(session_pcap_path, session_len, packet_len, packet_offset):
    try:
        packets = rdpcap(session_pcap_path)
    except Exception as e:
        print(f"Error reading pcap file: {e}")
        return None, None

    if len(packets) == 0:
        return None, None


    start_time = packets[0].time


    if not packets[0].haslayer(scapy.IP):
        return None, None
    ip_layer = packets[0][scapy.IP]
    src_ip = ip_layer.src
    dst_ip = ip_layer.dst
    proto = ip_layer.proto

    src_port, dst_port = None, None
    if proto == 6 and packets[0].haslayer(scapy.TCP):
        tcp_layer = packets[0][scapy.TCP]
        src_port = tcp_layer.sport
        dst_port = tcp_layer.dport
    elif proto == 17 and packets[0].haslayer(scapy.UDP):
        udp_layer = packets[0][scapy.UDP]
        src_port = udp_layer.sport
        dst_port = udp_layer.dport
    else:
        return None, None

    five_tuple = (src_ip, src_port, dst_ip, dst_port, proto)
    five_tuple = normalize_five_tuple(five_tuple)
    five_tuple_str = f"{src_ip}-{src_port}-{dst_ip}-{dst_port}-{proto}"

    packets_dec = []
    for packet in packets[:session_len]:
        raw_data = bytes(packet)
        start = packet_offset
        end = start + packet_len
        data_part = raw_data[start:end]
        frame_dec = list(data_part)
        packets_dec.append(frame_dec)

    if len(packets_dec) < 3:
        return None, None


    df = pd.DataFrame(packets_dec).fillna(-1)
    packets_dec_matrix = df.values.astype(np.int16)

    session_matrix = np.full((session_len, packet_len), -1, dtype=np.int16)
    rows = min(packets_dec_matrix.shape[0], session_len)
    cols = min(packets_dec_matrix.shape[1], packet_len)
    session_matrix[:rows, :cols] = packets_dec_matrix[:rows, :cols]


    def mask_features(indices, matrix):
        valid = [i - packet_offset for i in indices if 0 <= i - packet_offset < packet_len]
        matrix[:, valid] = -1


    mask_features([18,19,24,25,26,27,28,29,30,31,32,33], session_matrix)
    

    proto_col = 9 
    if proto == 6:
        mask_features([38,39,40,41,42,43,44,45,50,51], session_matrix[session_matrix[:, proto_col] == 6])
    elif proto == 17:
        mask_features([40,41], session_matrix[session_matrix[:, proto_col] == 17])


    result = {
        "start_time": start_time,
        "five_tuple": five_tuple,
        "matrix": session_matrix.tolist()
    }


    return result, (session_matrix == -1).astype(np.uint8)


def normalize_five_tuple(five_tuple):

    src_ip, src_port, dst_ip, dst_port, proto = five_tuple
    

    if src_ip > dst_ip:
        src_ip, dst_ip = dst_ip, src_ip
        src_port, dst_port = dst_port, src_port
    elif src_ip == dst_ip and src_port > dst_port:
        src_port, dst_port = dst_port, src_port
        
    return (src_ip, src_port, dst_ip, dst_port, proto)


from datetime import datetime
def parse_session_pcap_to_matrix2(session_pcap_path, session_len, packet_len, packet_offset):


    with open(session_pcap_path, 'rb') as f:
        content = f.read()
    hexc = binascii.hexlify(content)

    # 
    if hexc[:8] == b'd4c3b2a1':
        little_endian = True
    else:
        little_endian = False


    hexc = hexc[48:]

    if len(hexc) >= 16:
        if not little_endian:
            ts_sec = int(hexc[:8], 16)  
            ts_usec = int(hexc[8:16], 16) 
        else:
            ts_sec = int(hexc[8:16], 16)  
            ts_usec = int(hexc[:8], 16)

        start_time = f"{ts_sec}.{ts_usec:06d}"

    else:
        start_time = None

    filename = os.path.basename(session_pcap_path) 
    parts = filename.split('_') 
    if len(parts) < 5:
        return None
    if(parts[0].split('.')[1] == 'TCP'):
        proto = 6
    elif(parts[0].split('.')[1] == 'UDP'):
        proto = 17
    else:
        return
    src_ip = parts[1].replace('-', '.') 
    src_port = int(parts[2])  
    dst_ip = parts[3].replace('-', '.')  
    dst_port = int(parts[4].split('.')[0])  

    five_tuple = (src_ip, src_port, dst_ip, dst_port, proto)
    five_tuple = normalize_five_tuple(five_tuple)

    packets_dec = []
    while len(hexc) > 0 and len(packets_dec) < session_len:
        frame_len = hexc[16:24]
        if little_endian:
            frame_len = binascii.hexlify(binascii.unhexlify(frame_len)[::-1]) 
        frame_len = int(frame_len, 16)

        hexc = hexc[32:] 
        frame_hex = hexc[packet_offset * 2:min(packet_len * 2, frame_len * 2)]
        frame_dec = [int(frame_hex[i:i + 2], 16) for i in range(0, len(frame_hex), 2)]
        packets_dec.append(frame_dec)

        


        hexc = hexc[frame_len * 2:]

    if len(packets_dec) < 3:
        return None


    packets_dec_matrix = pd.DataFrame(packets_dec).fillna(-1).values.astype(np.int16)#
    session_matrix = np.ones((session_len, packet_len), dtype=np.int16) * -1

    row_idx = min(packets_dec_matrix.shape[0], session_len)
    col_idx = min(packets_dec_matrix.shape[1], packet_len)
    session_matrix[:row_idx, :col_idx] = packets_dec_matrix[:row_idx, :col_idx]

    
    common_irr_fea_idx = [18, 19, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]#
    tcp_irr_fea_idx = [38, 39, 40, 41, 42, 43, 44, 45, 50, 51]#
    udp_irr_fea_idx = [40, 41]#40,41-udp checksum
    common_irr_fea_idx = [idx - packet_offset for idx in common_irr_fea_idx]
    session_matrix[:, common_irr_fea_idx] = -1
    # 

    for idx in tcp_irr_fea_idx:

        session_matrix[session_matrix[:, 23 - packet_offset] == 6, idx - packet_offset] = -1

    for idx in udp_irr_fea_idx:
        session_matrix[session_matrix[:, 23 - packet_offset] == 17, idx - packet_offset] = -1
    session_matrix = session_matrix[:session_len, :packet_len]
    padding_mask = (session_matrix == -1).astype(np.uint8)

    result = {
        "start_time": start_time,
        "five_tuple": five_tuple,
        "matrix": session_matrix.tolist(),
        "padding_mask":padding_mask.tolist()
    }

    
    return result 


def parse_pcap_metadata(pcap_path):
    """
    :return: pcap_metadata (pd.DataFrame)
    """
    # tshark  -T fields $fields -r xxx.pcap -E header=y -E separator=, -E occurrence=f > xxx.csv
    # $fields = -e xxx -e xxx ...
    pcap_dir, pcap_name = os.path.split(pcap_path)
    csv_name = os.path.splitext(pcap_name)[0] + '.csv'
    csv_path = os.path.join(pcap_dir, csv_name)
    fields = '-e frame.time_epoch -e frame.len -e ip.src -e ip.dst -e ipv6.src -e ipv6.dst ' \
             '-e tcp.srcport -e tcp.dstport ' \
             '-e tcp.flags.urg -e tcp.flags.ack -e tcp.flags.push -e tcp.flags.reset -e tcp.flags.syn -e tcp.flags.fin ' \
             '-e udp.srcport -e udp.dstport'
    cmd = f'tshark -T fields {fields} -r {pcap_path} -E header=y -E separator=, -E occurrence=f > {csv_path}'
    ret = os.system(cmd)
    if ret == 0:
        print(f'parse {pcap_path} metadata successfully')
    else:
        print(f'parse {pcap_path} error')
        exit(1)

    pcap_metadata = pd.read_csv(csv_path)
    return pcap_metadata


def get_session_start_time(pcap_metadata, session_pcap_path):

    _, session_pcap_name = os.path.split(session_pcap_path)
    five_tuple_info = session_pcap_name.split('.')[1]
    protocol, src_ip, src_port, dst_ip, dst_port = five_tuple_info.split('_')
    if 'a' in src_ip or 'b' in src_ip or 'c' in src_ip or 'd' in src_ip or 'e' in src_ip or 'f' in src_ip:
        src_ip = src_ip.replace('-', ':')
        dst_ip = dst_ip.replace('-', ':')
    else:
        src_ip = src_ip.replace('-', '.')
        dst_ip = dst_ip.replace('-', '.')
    five_tuple_key = '_'.join(sorted([src_ip, src_port, dst_ip, dst_port, protocol]))

    print('five_tuple_key: ', five_tuple_key)

    if 'five_tuple_key' not in pcap_metadata.columns:
        pcap_metadata.loc[pd.isnull(pcap_metadata['ipv6.src']), 'src_ip'] = \
            pcap_metadata.loc[pd.isnull(pcap_metadata['ipv6.src']), 'ip.src']
        pcap_metadata.loc[pd.isnull(pcap_metadata['ipv6.dst']), 'dst_ip'] = \
            pcap_metadata.loc[pd.isnull(pcap_metadata['ipv6.dst']), 'ip.dst']
        pcap_metadata.loc[pd.isnull(pcap_metadata['ip.src']), 'src_ip'] = \
            pcap_metadata.loc[pd.isnull(pcap_metadata['ip.src']), 'ipv6.src']
        pcap_metadata.loc[pd.isnull(pcap_metadata['ip.dst']), 'dst_ip'] = \
            pcap_metadata.loc[pd.isnull(pcap_metadata['ip.dst']), 'ipv6.dst']
        pcap_metadata.loc[pd.isnull(pcap_metadata['udp.srcport']), 'src_port'] = \
            pcap_metadata.loc[pd.isnull(pcap_metadata['udp.srcport']), 'tcp.srcport']
        pcap_metadata.loc[pd.isnull(pcap_metadata['udp.dstport']), 'dst_port'] = \
            pcap_metadata.loc[pd.isnull(pcap_metadata['udp.dstport']), 'tcp.dstport']
        pcap_metadata.loc[pd.isnull(pcap_metadata['tcp.srcport']), 'src_port'] = \
            pcap_metadata.loc[pd.isnull(pcap_metadata['tcp.srcport']), 'udp.srcport']
        pcap_metadata.loc[pd.isnull(pcap_metadata['tcp.dstport']), 'dst_port'] = \
            pcap_metadata.loc[pd.isnull(pcap_metadata['tcp.dstport']), 'udp.dstport']
        pcap_metadata.loc[pd.isnull(pcap_metadata['udp.srcport']), 'protocol'] = 'TCP'
        pcap_metadata.loc[pd.isnull(pcap_metadata['tcp.srcport']), 'protocol'] = 'UDP'

        # filter ip and ipv6 is NaN or tcp ports and udp ports is NaN
        pcap_metadata = pcap_metadata.loc[(pd.notnull(pcap_metadata['ip.src']) & pd.notnull(pcap_metadata['ip.dst'])) |
                                          (pd.notnull(pcap_metadata['ipv6.src']) & pd.notnull(
                                              pcap_metadata['ipv6.dst']))]
        pcap_metadata = pcap_metadata.loc[
            (pd.notnull(pcap_metadata['tcp.srcport']) & pd.notnull(pcap_metadata['tcp.dstport'])) |
            (pd.notnull(pcap_metadata['udp.srcport']) & pd.notnull(pcap_metadata['udp.dstport']))]
        pcap_metadata['five_tuple_key'] = pcap_metadata.apply(
            lambda row: '_'.join(sorted([str(row['ip.src']), str(int(row.src_port)),
                                         str(row['ip.dst']), str(int(row.dst_port)),
                                         row.protocol])), axis=1)

    return pcap_metadata.loc[pcap_metadata['five_tuple_key'] == five_tuple_key, 'frame.time_epoch'].min(), \
           five_tuple_key, pcap_metadata


def get_session_contextual_packet_len_seq(pcap_metadata, session_start_time,
                                          agg_scale, agg_name, agg_points_num, five_tuple_key, beta=0.5):
    """
    return agg_seq (ndarray), pcap_metadata (pd.DataFrane), session_features_seq (ndarray)， seqs(list) when agg_ms
    else return agg_seq (ndarray), pcap_metadata (pd.DataFrane), None
    """
    time_key = f'time_{agg_name}'

    if time_key not in pcap_metadata.columns:
        pcap_metadata[time_key] = (pcap_metadata['frame.time_epoch'] / agg_scale).map(int)
    session_start_time = int(session_start_time / agg_scale)

    start_time = session_start_time - agg_points_num / 2 + 1
    end_time = session_start_time + agg_points_num / 2
    seq = pcap_metadata.loc[(pcap_metadata[time_key] >= start_time) & (pcap_metadata[time_key] <= end_time),
                            [time_key, 'frame.len']]
    session_seq = pcap_metadata.loc[(pcap_metadata[time_key] >= start_time) &
                                    (pcap_metadata[time_key] <= end_time) &
                                    (pcap_metadata['five_tuple_key'] == five_tuple_key),
                                    [time_key, 'frame.len']]

    seq = seq.groupby(time_key).sum()
    session_seq = session_seq.groupby(time_key).sum()
    agg_seq = np.zeros(agg_points_num)
    agg_session_seq = np.zeros(agg_points_num)
    for i in seq.index:#
        agg_seq[int(i - start_time)] += seq.loc[i, 'frame.len']
    for i in session_seq.index:
        agg_session_seq[int(i - start_time)] += session_seq.loc[i, 'frame.len']
    max_agg_seq_packet_len = agg_seq.max()
    max_agg_session_seq_packet_len = agg_session_seq.max()
    agg_seq = agg_seq / max_agg_seq_packet_len * max_agg_session_seq_packet_len * beta#
    agg_seq += agg_session_seq#

    if agg_name == 'ms':#
        session_features_seq = pcap_metadata.loc[pcap_metadata['five_tuple_key'] == five_tuple_key,
                                                 ['frame.len', 'frame.time_epoch', 'five_tuple_key',
                                                  'tcp.flags.urg', 'tcp.flags.ack', 'tcp.flags.push',
                                                  'tcp.flags.reset', 'tcp.flags.syn', 'tcp.flags.fin']]

        return agg_seq, pcap_metadata, session_features_seq.values.tolist(), seq.loc[:, 'frame.len'].tolist()
    else:
        return agg_seq, pcap_metadata, None


def wavelet_transform(seq, wave_name, agg_points_num):
    """
    :return: normalized spectrogram (ndarray: [freqs, t])
    """
    scales = np.arange(1, agg_points_num + 1)
    fc = pywt.central_frequency(wave_name)
    scales = 2 * fc * agg_points_num / scales
    cwtmatr, freqs = pywt.cwt(seq, scales, wave_name)  
    spectrogram = np.log2((abs(cwtmatr)) ** 2 + 1)
    spectrogram = (spectrogram - np.min(spectrogram)) / (np.max(spectrogram) + 1)

    return spectrogram



def gen_temporal_data(pcap_path, sessions_dir, data_path, session_len=64, packet_len=64, packet_offset=14):
    """
    :return: session_pcaps_used
    """
    # split
    split_pcap_to_sessions(pcap_path, sessions_dir)
    # parse
    parse_start_time = time.time()
    session_pcaps = get_contents_in_dir(sessions_dir, ['.'], ['.pcap'])
    temporal_data = np.zeros((len(session_pcaps), session_len, packet_len))
    temporal_result = []
    temporal_mask = np.zeros((len(session_pcaps), session_len, packet_len))
    session_pcaps_used = []
    idx = 0
    time_temporal_matrix = []
    for session_pcap in session_pcaps:

        starttime = time.time()
        session_matrix = parse_session_pcap_to_matrix2(session_pcap,
                                                                    session_len, packet_len, packet_offset)
        endtime = time.time()
        time_temporal_matrix.append(endtime-starttime)


        if session_matrix is None:
            print(f'{session_pcap} is too short (session len < 3)')
            continue

        temporal_result.append(session_matrix)

        idx += 1
        session_pcaps_used.append(session_pcap)


    if time_temporal_matrix:
        mean_time = sum(time_temporal_matrix) / len(time_temporal_matrix)
        print(f"file{pcap_path}，total number of sessions: {len(time_temporal_matrix)} ")
        print(f"file{pcap_path}，Total time spent extracting byte matrix: {sum(time_temporal_matrix)}秒 ")
        print(f"file{pcap_path}，The average time spent extracting the byte matrix per session is: {mean_time} ")
    else:
        print("No session duration data is recorded")
    # ----------------------------------------------------------
    # 
    #  
    temporal_data = temporal_result
    #temporal_data = temporal_result[:len(session_pcaps_used), :, :]
    #temporal_mask = temporal_mask[:len(session_pcaps_used), :, :]
    parse_end_time = time.time()
    # save
    data_dir, data_name = os.path.split(data_path)
    data_name = os.path.splitext(data_name)[0] + '_temporal.npy'
    data_path = os.path.join(data_dir, data_name)
    mask_name = os.path.splitext(data_name)[0] + '_mask.npy'
    mask_path = os.path.join(data_dir, mask_name)
    used_name = os.path.splitext(data_name)[0] + '_session_used.json'
    used_path = os.path.join(data_dir, used_name)
    #if not os.path.exists(data_dir):
    #    os.makedirs(data_dir)
    np.save(data_path, temporal_data)
    #np.save(mask_path, temporal_mask)
    with open(used_path, 'w+') as f:
        json.dump(session_pcaps_used, f)
    print(f'save {data_path}, {mask_path} and {used_path} successfully, '
          f'total {len(session_pcaps_used)} samples, '
          f'temporal feature extract time cost: {parse_end_time -parse_start_time} s, '
          f'average {(parse_end_time - parse_start_time) / len(session_pcaps_used)} s / session')
    # visualize
    # visualize_data(temporal_data, data_name)
    return session_pcaps_used

#
def gen_contextual_data2(pcap_path, session_pcaps, wave_name, data_path, agg_seqs_path=None):
    """
        if agg_seqs_path is not None, use previous agg sequences. pcap_path, session_pcaps can ignore.
    """
    if agg_seqs_path is not None:
        agg_seqs = np.load(agg_seqs_path)
        data_num = agg_seqs.shape[0]
        contextual_data = np.zeros((data_num, 3, 128, 128))
        # 
        session_features_seqs = None
        seqs = None
        # agg_seqs shape: [N, 3, 128]
        for i in range(data_num):
            ms_spectrogram = wavelet_transform(agg_seqs[i, 0, :], wave_name, 128)
            s_spectrogram = wavelet_transform(agg_seqs[i, 1, :], wave_name, 128)
            min_spectrogram = wavelet_transform(agg_seqs[i, 2, :], wave_name, 128)

            contextual_data[i, 0, :, :] = ms_spectrogram[:, :]
            contextual_data[i, 1, :, :] = s_spectrogram[:, :]
            contextual_data[i, 2, :, :] = min_spectrogram[:, :]
        print(f'get contextual feature from previous contextual agg sequence {agg_seqs_path} successfully')
    else:
        # parse whole pcap file, generate frame_len, time... from each frame
        pcap_metadata = parse_pcap_metadata(pcap_path)
        # aggregate and transform
        process_start_time = time.time()
        contextual_data = np.zeros((len(session_pcaps), 3, 128, 128))#
        agg_seqs = np.zeros((len(session_pcaps), 3, 128))#
        session_features_seqs = []#
        seqs = []#
        for idx, session_pcap in enumerate(session_pcaps):
            session_start_time, five_tuple_key, pcap_metadata = get_session_start_time(pcap_metadata, session_pcap)
            print('session_start_time:',session_start_time)
            # ms aggregate
            ms_agg_seq, pcap_metadata, session_features_seq, seq = get_session_contextual_packet_len_seq(pcap_metadata,
                                                                                                         session_start_time,
                                                                                                         0.001, 'ms', 128,
                                                                                                         five_tuple_key)
            # s aggregate
            s_agg_seq, pcap_metadata, _ = get_session_contextual_packet_len_seq(pcap_metadata, session_start_time,
                                                                                1, 's', 128, five_tuple_key)
            min_agg_seq, pcap_metadata, _ = get_session_contextual_packet_len_seq(pcap_metadata, session_start_time,
                                                                                  1, 'min', 128, five_tuple_key)#
            ms_spectrogram = wavelet_transform(ms_agg_seq, wave_name, 128)#
            s_spectrogram = wavelet_transform(s_agg_seq, wave_name, 128)
            min_spectrogram = wavelet_transform(min_agg_seq, wave_name, 128)

            contextual_data[idx, 0, :, :] = ms_spectrogram[:, :]
            contextual_data[idx, 1, :, :] = s_spectrogram[:, :]
            contextual_data[idx, 2, :, :] = min_spectrogram[:, :]

            agg_seqs[idx, 0, :] = ms_agg_seq[:]
            agg_seqs[idx, 1, :] = s_agg_seq[:]
            agg_seqs[idx, 2, :] = min_agg_seq[:]


            session_features_seqs.append(session_features_seq)
            seqs.append(seq)

            print(f'get contextual data of {session_pcap} successfully')
        process_end_time = time.time()
        print(f'contextual feature extract time cost: {process_end_time - process_start_time} s, '
              f'average {(process_end_time - process_start_time) / len(session_pcaps)} s / session')
    # save
    data_dir, data_name = os.path.split(data_path)
    session_features_seqs_name = os.path.splitext(data_name)[0] + '_session_features_seqs.joblib'#
    agg_seqs_name = os.path.splitext(data_name)[0] + '_agg_seqs.npy'#
    seqs_name = os.path.splitext(data_name)[0] + '_seqs.npy'#
    data_name = os.path.splitext(data_name)[0] + f'_{wave_name}_contextual.npy'#
    session_features_seqs_path = os.path.join(data_dir, session_features_seqs_name)
    agg_seqs_path = os.path.join(data_dir, agg_seqs_name)
    seqs_path = os.path.join(data_dir, seqs_name)
    data_path = os.path.join(data_dir, data_name)

    np.save(data_path, contextual_data)#
    print(f'save {data_path} successfully')
    if session_features_seqs is not None:  #
        joblib.dump(session_features_seqs, session_features_seqs_path)
        print(f'save {session_features_seqs_path} successfully')
        np.save(agg_seqs_path, agg_seqs)
        print(f'save {agg_seqs_path} successfully')
        joblib.dump(seqs, seqs_path)
        print(f'save {seqs_path} successfully')

import glob

def query_sessions(df, 
                   src_ip_list=None, 
                   dst_ip_list=None, 
                   ip_pairs=None, 
                   time_range=None,
                   max_context=10,
                   max_window=300,
                   current_time=None):
    df = df.copy()
    mask = pd.Series(False, index=df.index)
    df['src_ip'] = df['five_tuple'].apply(lambda x: x[0])
    df['dst_ip'] = df['five_tuple'].apply(lambda x: x[2])
    df['time_diff'] = abs(df['start_time'] - current_time)  
    ip_set = set()
    if ip_pairs is not None:
        for s, d in ip_pairs:
            ip_set.add(s)
            ip_set.add(d)
    

    if src_ip_list is not None or ip_pairs is not None:
        src_ips = set()
        if src_ip_list is not None:
            src_ips.update(src_ip_list)
        if ip_pairs is not None:
            src_ips.update(ip_set)
        mask |= df['src_ip'].isin(src_ips)
    

    if dst_ip_list is not None or ip_pairs is not None:
        dst_ips = set()
        if dst_ip_list is not None:
            dst_ips.update(dst_ip_list)
        if ip_pairs is not None:
            dst_ips.update(ip_set)
        mask |= df['dst_ip'].isin(dst_ips)
    mask1 = mask

    if time_range is not None:
        min_time, max_time = (time_range[0])
        time_mask = (df['start_time'] >= min_time) & (df['start_time'] <= max_time)
        mask &= time_mask
    filtered_df = df[mask].copy()
    
    if len(filtered_df) < max_context:

        time_mask = (df['start_time'] >= current_time - max_window) & \
                    (df['start_time'] <= current_time + max_window)
        mask1 &= time_mask
        filtered_df1 = df[mask1].copy()
        filtered_df = filtered_df1.sort_values(by='time_diff', ascending=True)
        result_df = filtered_df.head(max_context)
    else:
        filtered_df = filtered_df.sort_values(by='time_diff', ascending=True)
        result_df = filtered_df.head(max_context)
    return result_df.drop(columns=['time_diff'])


def gen_contextual_data(wave_name, data_path):
    """
        function:read *_temporal.npy file, find the contextual info from all *_sequences.npy files
    """

    all_sessions = load_all_sessions(data_path)

    df = pd.DataFrame(all_sessions)


    df = df.sort_values(by='start_time')


    file_pattern = os.path.join(data_path, "*_temporal.npy")
    temporal_files = glob.glob(file_pattern)

    if not temporal_files:
        print(f"dir {data_path} Not Found *_temporal.npy file")
        return
    
    for file_path in temporal_files:
        print(f"For file: {file_path}matching context===")
        merged_sessions = []
        try:

            parts = file_path.split('/')

            last_part = parts[-1]

            class_name = last_part.split('_')[0]
            data = np.load(file_path, allow_pickle=True)

            count = 1
            starttime = time.time()
            for session in data:

                
                if(count % 100 == 0):
                    print(f'For file {file_path}matching context, processing {count} sessions')
                if(count > 2000):
                    break
                if not isinstance(session, dict):
                    print("Data format error: session is not a dictionary type")
                    continue
                count += 1
                try:
                    start_time = float(session["start_time"])
                    five_tuple = session["five_tuple"]
                    matrix = np.array(session["matrix"])  
       
                    contextual = query_sessions(df, None, None, [(five_tuple[0], five_tuple[2])], [(start_time-60, start_time +60)],\
                                                max_context=10, max_window=300,current_time = start_time)

                  
                    if contextual.empty:
                        contextual_dict = None
                    else:
                   
                        contextual_dict = contextual.to_dict(orient='records')

            
                    merged_session = session.copy()
                    
                    merged_session['class_name'] = class_name
           
                    

                    for item in contextual_dict:
                  
                        if 'time_intervals' in item:
                  
                            result = wavelet_transform(item['time_intervals'], wave_name, 64)
                    
                            item["intervals_spectrogram"] = result

                        else:
                            print("'time_intervals' is missing in dictionary")
                        if 'packet_lens' in item:
                      
                            result = wavelet_transform(item['packet_lens'], wave_name, 64)
              
              
                            item["lens_spectrogram"] = result
                        else:
                            print("'time_intervals' is missing in dictionary")
                    #merged_session['intervals_spectrogram'] = intervals_spectrogram
                    #merged_session['lens_spectrogram'] = lens_spectrogram
                    merged_session['contextual'] = contextual_dict

                    merged_sessions.append(merged_session)
 
                except KeyError as e:
                    print(f"Field {e} does not exist in the dictionary")
                except Exception as e:
                    print(f"An error occurred while processing the session: {str(e)}")
            endtime = time.time()
            print(f'For file {file_path}matching context, processing complete, total time: {endtime-starttime}')
            print(f'For file {file_path}matching context, processing complete, total sessions: {len(data)}')
            print(f'For file {file_path}matching context, processing complete, average time per session: {(endtime-starttime)/len(data)}')
        except Exception as e:
            print(f"Load file {file_path} error: {str(e)}")
        data_name = os.path.join(data_path, class_name+f'_{wave_name}_contextual.npy')#
        np.save(data_name, merged_sessions)
    

def gen_session_feature_sequence2(session_pcaps_used, time_seq_len=64, packet_seq_len=64, data_path=None):

    start_time = time.time()
    for pcap_file in session_pcaps_used:

        streamer = NFStreamer(
            source=pcap_file,
            splt_analysis=64,  
            udps=None,
            n_dissections=1  
        )
        flow = next(iter(streamer), None)
        print(flow.splt_ps)
        print(flow.splt_piat_ms)

    end_time = time.time()
    print(f'nfstream total time: ', end_time - start_time)
    sys.exit(0)


def gen_session_feature_sequence(session_pcaps_used, time_seq_len=64, packet_seq_len=64, data_path=None):
    #os.makedirs(data_path, exist_ok=True)
    start_time_waste = time.time()
    all_packet_lens = []
    all_time_intervals = []
    sessions = []
    for pcap_file in session_pcaps_used:
        try:

            packet_lens = np.zeros(packet_seq_len, dtype=np.uint16)
            time_intervals = np.zeros(time_seq_len, dtype=np.float32)

            prev_time = None
            packet_count = 0
            first_packet = True
            five_tuple= []
            start_time = None

            def _process_packet(pkt):
                nonlocal packet_count, first_packet, prev_time, five_tuple, start_time
                if not pkt.haslayer(IP) or packet_count >= packet_seq_len:
                    return

                if first_packet:
                    ip = pkt[IP]
                    src_ip = ip.src
                    dst_ip = ip.dst
                    proto = ip.proto
                    src_port, dst_port = 0, 0
                    
                    if pkt.haslayer(TCP):
                        src_port = pkt[TCP].sport
                        dst_port = pkt[TCP].dport
                    elif pkt.haslayer(UDP):
                        src_port = pkt[UDP].sport
                        dst_port = pkt[UDP].dport
                    elif pkt.haslayer(ICMP):
                        src_port = pkt[ICMP].type
                        dst_port = pkt[ICMP].code
                    

                    five_tuple = (src_ip, src_port, dst_ip, dst_port, proto)
                    five_tuple = normalize_five_tuple(five_tuple)
 
                    start_time = pkt.time
                    first_packet = False

                packet_lens[packet_count] = len(pkt)
                

                current_time = pkt.time
                if prev_time is None: 
                    time_intervals[packet_count] = 0.0
                else:
                    time_intervals[packet_count] = current_time - prev_time
                prev_time = current_time
                
                packet_count += 1

            sniff(
                offline=pcap_file,
                store=False,
                filter="ip",  
                prn=_process_packet,
                count=packet_seq_len, 
                quiet=True
            )
            

            if packet_count == 0:
                print(f"Skip empty files: {pcap_file}")
                continue
            

            packet_lens_final = packet_lens[:packet_count] if packet_count < packet_seq_len else packet_lens
            time_intervals_final = time_intervals[:packet_count] if packet_count < time_seq_len else time_intervals
            

            if packet_count < packet_seq_len:
                packet_lens_final = np.pad(packet_lens_final, (0, packet_seq_len - packet_count), 'constant')
            if packet_count < time_seq_len:
                time_intervals_final = np.pad(time_intervals_final, (0, time_seq_len - packet_count), 'constant')

            session_data = {
                'five_tuple':five_tuple,
                'start_time':start_time,
                'packet_lens':packet_lens_final,
                'time_intervals':time_intervals_final
            }

            sessions.append(session_data.copy())  
        except Exception as e:
            print(f"Error processing file {pcap_file}: {str(e)}")
    end_time = time.time()
    print(f'sniff total time: ', end_time - start_time_waste)

    if sessions:
        data_dir, data_name = os.path.split(data_path)
        data_name = os.path.splitext(data_name)[0] + '_sequences.npy'
        np.save(os.path.join(data_dir, data_name), sessions)
        print(f"Save {len(sessions)} sessions to {os.path.join(data_dir, data_name)}")
    else:
        print("No valid data to save.")


def load_all_sessions(data_dir):


    file_pattern = os.path.join(data_dir, "*_sequences.npy")
    sequence_files = glob.glob(file_pattern)
    
    if not sequence_files:
        raise FileNotFoundError(f"No *_sequences.npy file found in directory {data_dir}")
    
    all_sessions = []  
    
    for file_path in sequence_files:
        print(f"Loading file: {file_path}")
        try:

            data = np.load(file_path, allow_pickle=True)
            

            for session in data:

                if not all(key in session for key in ['five_tuple', 'start_time', 'packet_lens', 'time_intervals']):
                    print(f"Warning: Incomplete session in file {file_path}, skipped")
                    continue
                

                all_sessions.append({
                    'five_tuple': session['five_tuple'],
                    'start_time': session['start_time'],
                    'packet_lens': session['packet_lens'],
                    'time_intervals': session['time_intervals']
                })
                
        except Exception as e:
            print(f"Error loading file {file_path}: {str(e)}")
            continue
    
    print(f"Loaded {len(all_sessions)} sessions")
    return all_sessions



def visualize_data(data, data_name):
    def plot_figure(matrix, cmap, save_name):
        fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(9, 9),
                                subplot_kw={'xticks': [], 'yticks': []})
        chosen_idx = np.random.choice(len(matrix), 9)
        for idx, ax in enumerate(axs.flat):
            ax.imshow(matrix[chosen_idx[idx]], cmap=cmap, interpolation='bilinear')
            ax.set_title(f'#{chosen_idx[idx]}')

        plt.tight_layout()
        plt.savefig(f'{save_name}.png')

    if len(data.shape) == 4:  # case for contextual data: N x agg_scales x freqs x t
        for idx, agg_scale in enumerate(['ms', 's', 'min']):
            plot_figure(data[:, idx, :, :], 'viridis', f'{os.path.splitext(data_name)[0]}_{agg_scale}')
    else:
        plot_figure(data, 'Blues', os.path.splitext(data_name)[0])


def gen_single_traffic_type_data(pcaps_path, class_name, sessions_dir, data_path, wave_name, base_dir,
                                 agg_seqs_path=None, contextual=True):
    # merge pcaps of single class
    pcap_file = os.path.join(base_dir,f'{class_name}.pcap')
    if os.path.isdir(pcaps_path):
        merge_pcaps(pcaps_path, pcap_file)
    else:
        ret = os.system(f'cp {pcaps_path} {pcap_file}')
        if ret == 0:
            print(f'copy {pcaps_path} to {pcap_file} successfully')
        else:
            print(f'copy {pcaps_path} to {pcap_file} error')

    filter_protocols_in_pcap(pcap_file)#

    session_pcaps_used = gen_temporal_data(pcap_file, sessions_dir, data_path)
    print(f'{class_name} has {len(session_pcaps_used)} sessions')

    if not contextual:
        return

    starttime = time.time()
    gen_session_feature_sequence(session_pcaps_used, 64, 64, data_path)
    endtime = time.time()
    print(f'{class_name} gen_session_feature_sequence average time cost: ', (endtime - starttime)/len(session_pcaps_used))



def gen_multi_traffic_type_data(pcaps_path, data_path, wave_name, base_dir):
    pcaps = get_contents_in_dir(pcaps_path, '.', ['.pcap'])
    if os.path.isdir(pcaps[0]):
        for d in pcaps:

            class_name = os.path.split(d)[1]
            gen_single_traffic_type_data(d, class_name, os.path.join(base_dir,f'{class_name}_sessions'),
                                         os.path.join(data_path, f'{class_name}.npy'), wave_name, base_dir)
    else:
        for p in pcaps:

            class_name = os.path.splitext(os.path.split(p)[1])[0]

            gen_single_traffic_type_data(p, class_name, os.path.join(base_dir,f'{class_name}_sessions'),
                                        os.path.join(data_path, f'{class_name}.npy'), wave_name, base_dir)

        gen_contextual_data(wave_name, data_path)#




if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--multiple', action='store_true', required=False)
    args.add_argument('--contextual', action='store_true', required=False)
    args.add_argument('--pcaps_path', type=str, required=True)
    args.add_argument('--base_dir', type=str, required=False)
    args.add_argument('--class_name', type=str, required=False)
    args.add_argument('--sessions_dir', type=str, required=False)
    args.add_argument('--data_path', type=str, required=True)
    args.add_argument('--wave_name', type=str, required=True)
    args.add_argument('--session_pcaps_used', type=str, required=False, default=None)
    args.add_argument('--agg_seqs_path', type=str, required=False, default=None)
    args = args.parse_args()
    print(args)


    if args.multiple:
        gen_multi_traffic_type_data(args.pcaps_path, args.data_path, args.wave_name, args.base_dir)
    
    elif args.contextual:
        if args.session_pcaps_used:
            with open(args.session_pcaps_used, 'r') as f:
                session_pcaps_used = json.load(f)
        gen_contextual_data(args.pcaps_path, session_pcaps_used, args.wave_name, args.data_path, args.agg_seqs_path)
    
    else:
        gen_single_traffic_type_data(args.pcaps_path, args.class_name, args.sessions_dir,
                                     args.data_path, args.wave_name, args.base_dir)
