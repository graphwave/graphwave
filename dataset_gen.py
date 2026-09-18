"""Dataset generation for GraphWave.

Pipeline per traffic class:
  1. Merge the class pcaps into one file and filter out ARP/DHCP and
     non-TCP/UDP traffic.
  2. Split into bidirectional 5-tuple sessions (via SplitCap).
  3. Extract the byte matrix of each session (the first 64 packets, first
     64 bytes each) -> temporal features.
  4. Extract packet-length and inter-arrival-time sequences of each session,
     and match context sessions in time -> contextual features, transformed
     into wavelet spectrograms.
Run with --multiple to process a directory of classes end-to-end:

python dataset_gen.py --multiple --pcaps_path="./middleResults/pcap_data/botnet2014/" \
    --wave_name="cgau8" --data_path="./middleResults/temporal_contextual_data/botnet2014/" \
    --base_dir="./middleResults/pcap_data/botnet2014/"

Note: developed and tested on Python 3.7. On Python 3.11, np.save may
complain about ragged (irregular) array structures.
"""
import argparse
import binascii
import glob
import json
import os
import time

import numpy as np
import pandas as pd
import pywt
from scapy.all import ICMP, IP, TCP, UDP, sniff

from utils import get_contents_in_dir


def merge_pcaps(pcaps_dir, pcap_save_path, max_batch_size=200):
    """Merge all pcaps in a directory into one file, in batches of
    ``max_batch_size`` to keep each mergecap command line short."""
    pcaps = get_contents_in_dir(pcaps_dir, ['.'], ['.pcap', '.pcapng'])

    cmd_base = 'mergecap -F pcap -w '
    temp_pcaps = []

    # Merge input pcaps into temporary batch files
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

    # Merge temporary files further until few enough remain
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

    # Final merge into the target file
    final_cmd = cmd_base + pcap_save_path + ' ' + ' '.join(temp_pcaps)
    ret = os.system(final_cmd)
    if ret == 0:
        print(f'Merged all temp pcaps into final file {pcap_save_path} successfully.')
    else:
        print(f'Error merging temp pcaps into final file: {pcap_save_path}')
        exit(1)

    # Clean up temporary files
    for temp_pcap in temp_pcaps:
        try:
            os.remove(temp_pcap)
            print(f'Removed temporary pcap file {temp_pcap}.')
        except Exception as e:
            print(f'Error removing temporary file {temp_pcap}: {e}')


def filter_protocols_in_pcap(pcap_path):
    """Keep only TCP/UDP traffic, and drop ARP and DHCP packets."""
    display_filter = "not (arp or dhcp) and (tcp or udp)"
    pcap_dir, pcap_name = os.path.split(pcap_path)
    pcap_name = os.path.splitext(pcap_name)[0]
    out_path = os.path.join(pcap_dir, pcap_name)
    cmd = f'tshark -F pcap -r {pcap_path} -w {out_path}_tmp.pcap -Y "{display_filter}"'
    ret = os.system(cmd)
    if ret == 0:
        print(f'filter protocols with display filter {display_filter} in pcap successfully')
    else:
        print('filter protocols in pcap error')
        exit(1)
    os.system(f'rm -f {pcap_path}')
    os.system(f'mv {out_path}_tmp.pcap {out_path}.pcap')


def split_pcap_to_sessions(pcap_path, save_dir):
    """Split a pcap into 5-tuple session pcaps with SplitCap
    (limit each output file to 1000 packets)."""
    if os.path.exists(save_dir):
        os.system(f'rm -rf {save_dir}')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # 1. editcap converts the input pcap to classic pcap format and pipes it;
    # 2. SplitCap (run with mono) splits it into session files.
    ret = os.system(f'editcap -F pcap {pcap_path} - | mono SplitCap.exe -r - -s session -o {save_dir} -p 1000')
    if ret == 0:
        print(f'split {pcap_path} to sessions successfully')
    else:
        print(f'split {pcap_path} error')
        exit(1)


def normalize_five_tuple(five_tuple):
    """Normalize a five-tuple to a fixed order (sorted by IP and port)."""
    src_ip, src_port, dst_ip, dst_port, proto = five_tuple

    if src_ip > dst_ip:
        src_ip, dst_ip = dst_ip, src_ip
        src_port, dst_port = dst_port, src_port
    elif src_ip == dst_ip and src_port > dst_port:
        src_port, dst_port = dst_port, src_port

    return (src_ip, src_port, dst_ip, dst_port, proto)


def parse_session_pcap_to_matrix(session_pcap_path, session_len, packet_len, packet_offset):
    """Parse a session pcap into a byte matrix.

    Note: the raw pcap binary parsing below assumes a standard pcap file.
    It works for pcaps produced by ``mergecap -F pcap``, but may be fragile
    for other producers; prefer the pcap generated by this pipeline.

    Returns:
        result dict with start_time / five_tuple / matrix / padding_mask,
        or None when the session is too short (< 3 packets).
    """
    with open(session_pcap_path, 'rb') as f:
        content = f.read()
    hexc = binascii.hexlify(content)

    # Detect byte order from the pcap global magic number
    if hexc[:8] == b'd4c3b2a1':
        little_endian = True
    else:
        little_endian = False

    # Remove the 24-byte global pcap header
    hexc = hexc[48:]

    # Extract the timestamp of the first packet
    if len(hexc) >= 16:
        if not little_endian:
            ts_sec = int(hexc[:8], 16)
            ts_usec = int(hexc[8:16], 16)
        else:
            ts_sec = int(hexc[8:16], 16)
            ts_usec = int(hexc[:8], 16)
        start_time = f"{ts_sec}.{ts_usec:06d}"  # keep 6-digit microseconds
    else:
        start_time = None

    # Parse the five-tuple from the SplitCap file name,
    # e.g. "SplitCap.TCP_10-0-2-15_1044_208-73-211-152_80.pcap"
    filename = os.path.basename(session_pcap_path)
    parts = filename.split('_')
    if len(parts) < 5:
        return None
    if parts[0].split('.')[1] == 'TCP':
        proto = 6
    elif parts[0].split('.')[1] == 'UDP':
        proto = 17
    else:
        return None
    src_ip = parts[1].replace('-', '.')
    src_port = int(parts[2])
    dst_ip = parts[3].replace('-', '.')
    dst_port = int(parts[4].split('.')[0])

    five_tuple = normalize_five_tuple((src_ip, src_port, dst_ip, dst_port, proto))

    # Parse raw bytes of each packet
    packets_dec = []
    while len(hexc) > 0 and len(packets_dec) < session_len:
        frame_len = hexc[16:24]
        if little_endian:
            frame_len = binascii.hexlify(binascii.unhexlify(frame_len)[::-1])  # reverse due to little endian
        frame_len = int(frame_len, 16)

        hexc = hexc[32:]  # remove the 16-byte per-packet header
        frame_hex = hexc[packet_offset * 2:min(packet_len * 2, frame_len * 2)]  # skip the ethernet header
        frame_dec = [int(frame_hex[i:i + 2], 16) for i in range(0, len(frame_hex), 2)]
        packets_dec.append(frame_dec)

        hexc = hexc[frame_len * 2:]

    if len(packets_dec) < 3:
        return None

    # Pad shorter rows with -1 and build the session matrix
    packets_dec_matrix = pd.DataFrame(packets_dec).fillna(-1).values.astype(np.int16)
    session_matrix = np.ones((session_len, packet_len), dtype=np.int16) * -1
    row_idx = min(packets_dec_matrix.shape[0], session_len)
    col_idx = min(packets_dec_matrix.shape[1], packet_len)
    session_matrix[:row_idx, :col_idx] = packets_dec_matrix[:row_idx, :col_idx]

    # Mask irrelevant header fields (-1), so the model does not rely on them
    # IP header: 18,19-Identification; 24,25-Header Checksum; 26-33-src/dst IP
    common_irr_fea_idx = [18, 19, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]
    # TCP: 38-41-sequence number; 42-45-ack number; 50,51-TCP checksum
    tcp_irr_fea_idx = [38, 39, 40, 41, 42, 43, 44, 45, 50, 51]
    # UDP: 40,41-UDP checksum
    udp_irr_fea_idx = [40, 41]
    common_irr_fea_idx = [idx - packet_offset for idx in common_irr_fea_idx]
    session_matrix[:, common_irr_fea_idx] = -1
    # protocol field is the 10th byte of the IP header (index 23 - offset)
    for idx in tcp_irr_fea_idx:
        session_matrix[session_matrix[:, 23 - packet_offset] == 6, idx - packet_offset] = -1
    for idx in udp_irr_fea_idx:
        session_matrix[session_matrix[:, 23 - packet_offset] == 17, idx - packet_offset] = -1
    session_matrix = session_matrix[:session_len, :packet_len]
    padding_mask = (session_matrix == -1).astype(np.uint8)  # 1 marks masked/padding positions

    result = {
        "start_time": start_time,
        "five_tuple": five_tuple,
        "matrix": session_matrix.tolist(),
        "padding_mask": padding_mask.tolist()
    }

    return result


def wavelet_transform(seq, wave_name, agg_points_num):
    """Continuous wavelet transform of a sequence into a normalized spectrogram.

    For an input of dimension ``agg_points_num``, the output spectrogram has
    shape [agg_points_num, agg_points_num]: each column corresponds to one
    time point of the input signal, each row to a scale.
    """
    scales = np.arange(1, agg_points_num + 1)
    fc = pywt.central_frequency(wave_name)
    scales = 2 * fc * agg_points_num / scales
    cwtmatr, freqs = pywt.cwt(seq, scales, wave_name)  # cwtmatr: (freqs, t)
    spectrogram = np.log2((abs(cwtmatr)) ** 2 + 1)
    spectrogram = (spectrogram - np.min(spectrogram)) / (np.max(spectrogram) + 1)

    return spectrogram


def gen_temporal_data(pcap_path, sessions_dir, data_path, session_len=64, packet_len=64, packet_offset=14):
    """Extract the byte matrix of every session of one class.

    The first 14 bytes (ethernet header) are skipped. Sessions shorter than
    3 packets are dropped.
    """
    # split
    split_pcap_to_sessions(pcap_path, sessions_dir)
    # parse
    parse_start_time = time.time()
    session_pcaps = get_contents_in_dir(sessions_dir, ['.'], ['.pcap'])
    temporal_result = []
    session_pcaps_used = []
    for session_pcap in session_pcaps:
        session_result = parse_session_pcap_to_matrix(session_pcap, session_len, packet_len, packet_offset)
        if session_result is None:
            print(f'{session_pcap} is too short (session len < 3)')
            continue
        temporal_result.append(session_result)
        session_pcaps_used.append(session_pcap)

    parse_end_time = time.time()
    # save
    data_dir, data_name = os.path.split(data_path)
    data_name = os.path.splitext(data_name)[0] + '_temporal.npy'
    data_path = os.path.join(data_dir, data_name)
    used_name = os.path.splitext(data_name)[0] + '_session_used.json'
    used_path = os.path.join(data_dir, used_name)
    np.save(data_path, temporal_result)
    with open(used_path, 'w+') as f:
        json.dump(session_pcaps_used, f)
    print(f'save {data_path} and {used_path} successfully, '
          f'total {len(session_pcaps_used)} samples, '
          f'temporal feature extract time cost: {parse_end_time - parse_start_time} s, '
          f'average {(parse_end_time - parse_start_time) / len(session_pcaps_used)} s / session')
    return session_pcaps_used


def gen_session_feature_sequence(session_pcaps_used, time_seq_len=64, packet_seq_len=64, data_path=None):
    """Extract the packet-length and inter-arrival-time sequences (first 64
    packets) of each session with scapy streaming, and save them as
    ``*_sequences.npy``."""
    start_time = time.time()
    sessions = []
    for pcap_file in session_pcaps_used:
        try:
            # Pre-allocate to avoid dynamic growth
            packet_lens = np.zeros(packet_seq_len, dtype=np.uint16)
            time_intervals = np.zeros(time_seq_len, dtype=np.float32)
            prev_time = None
            packet_count = 0
            first_packet = True
            five_tuple = []
            start_ts = None

            def _process_packet(pkt):
                nonlocal packet_count, first_packet, prev_time, five_tuple, start_ts
                if not pkt.haslayer(IP) or packet_count >= packet_seq_len:
                    return
                # Extract the five-tuple from the first packet
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

                    five_tuple = normalize_five_tuple((src_ip, src_port, dst_ip, dst_port, proto))
                    start_ts = pkt.time
                    first_packet = False
                # Packet length
                packet_lens[packet_count] = len(pkt)

                # Inter-arrival time
                current_time = pkt.time
                if prev_time is None:  # first packet
                    time_intervals[packet_count] = 0.0
                else:
                    time_intervals[packet_count] = current_time - prev_time
                prev_time = current_time

                packet_count += 1

            # Stream-read with sniff (packets are not stored)
            sniff(
                offline=pcap_file,
                store=False,
                filter="ip",
                prn=_process_packet,
                count=packet_seq_len,
                quiet=True
            )

            if packet_count == 0:
                print(f"Skip empty file: {pcap_file}")
                continue

            # Pad the remaining positions if the session is shorter than the sequence length
            if packet_count < packet_seq_len:
                packet_lens = np.pad(packet_lens[:packet_count], (0, packet_seq_len - packet_count), 'constant')
            if packet_count < time_seq_len:
                time_intervals = np.pad(time_intervals[:packet_count], (0, time_seq_len - packet_count), 'constant')

            session_data = {
                'five_tuple': five_tuple,
                'start_time': start_ts,
                'packet_lens': packet_lens,
                'time_intervals': time_intervals
            }
            sessions.append(session_data.copy())  # copy to avoid reference overwrite
        except Exception as e:
            print(f"Error processing file {pcap_file}: {str(e)}")
    end_time = time.time()
    print('sniff total time: ', end_time - start_time)    # save
    if sessions:
        data_dir, data_name = os.path.split(data_path)
        data_name = os.path.splitext(data_name)[0] + '_sequences.npy'
        np.save(os.path.join(data_dir, data_name), sessions)
        print(f"Saved {len(sessions)} sessions to {os.path.join(data_dir, data_name)}")
    else:
        print("No valid data to save.")


def load_all_sessions(data_dir):
    """Load and merge all ``*_sequences.npy`` files under a directory."""
    file_pattern = os.path.join(data_dir, "*_sequences.npy")
    sequence_files = glob.glob(file_pattern)

    if not sequence_files:
        raise FileNotFoundError(f"No *_sequences.npy files found in {data_dir}")

    all_sessions = []
    for file_path in sequence_files:
        print(f"Loading file: {file_path}")
        try:
            data = np.load(file_path, allow_pickle=True)
            for session in data:
                # Skip sessions with missing fields
                if not all(key in session for key in ['five_tuple', 'start_time',
                                                      'packet_lens', 'time_intervals']):
                    print(f"Warning: incomplete session in {file_path}, skipped")
                    continue
                all_sessions.append({
                    'five_tuple': session['five_tuple'],
                    'start_time': session['start_time'],
                    'packet_lens': session['packet_lens'],
                    'time_intervals': session['time_intervals']
                })
        except Exception as e:
            print(f"Failed to load file {file_path}: {str(e)}")
            continue

    print(f"Loaded {len(all_sessions)} sessions in total")
    return all_sessions


def query_sessions(df, src_ip_list=None, dst_ip_list=None, ip_pairs=None,
                   time_range=None, max_context=10, max_window=300, current_time=None):
    """Select context sessions for the current session (boolean-indexing version).

    Candidates are sessions that overlap with the given IP pairs / IP lists
    within the time range; they are sorted by temporal distance to
    ``current_time`` and the top ``max_context`` are returned. If fewer than
    ``max_context`` sessions are found, the window is widened to
    ``max_window`` seconds around the current session (adaptive window).
    """
    df = df.copy()
    mask = pd.Series(False, index=df.index)
    df['src_ip'] = df['five_tuple'].apply(lambda x: x[0])
    df['dst_ip'] = df['five_tuple'].apply(lambda x: x[2])
    df['time_diff'] = abs(df['start_time'] - current_time)

    # Collect all IPs involved in the requested IP pairs
    ip_set = set()
    if ip_pairs is not None:
        for s, d in ip_pairs:
            ip_set.add(s)
            ip_set.add(d)

    # Attacker context (source IPs)
    if src_ip_list is not None or ip_pairs is not None:
        src_ips = set()
        if src_ip_list is not None:
            src_ips.update(src_ip_list)
        if ip_pairs is not None:
            src_ips.update(ip_set)
        mask |= df['src_ip'].isin(src_ips)

    # Target context (destination IPs)
    if dst_ip_list is not None or ip_pairs is not None:
        dst_ips = set()
        if dst_ip_list is not None:
            dst_ips.update(dst_ip_list)
        if ip_pairs is not None:
            dst_ips.update(ip_set)
        mask |= df['dst_ip'].isin(dst_ips)
    mask1 = mask
    # Temporal context (time window)
    if time_range is not None:
        min_time, max_time = (time_range[0])
        time_mask = (df['start_time'] >= min_time) & (df['start_time'] <= max_time)
        mask &= time_mask
    filtered_df = df[mask].copy()

    if len(filtered_df) < max_context:
        # Widen to the maximum allowed window (max_window seconds) and retry
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
    """Match context sessions for every session of every class, transform
    their packet-length / time-interval sequences into wavelet spectrograms,
    and save one ``<class>_<wave_name>_contextual.npy`` file per class.
    """
    # Load all session sequences of all classes
    all_sessions = load_all_sessions(data_path)
    df = pd.DataFrame(all_sessions)
    # Sort by start time to speed up range queries
    df = df.sort_values(by='start_time')

    file_pattern = os.path.join(data_path, "*_temporal.npy")
    temporal_files = glob.glob(file_pattern)

    if not temporal_files:
        print(f"No *_temporal.npy files found in {data_path}")
        return

    for file_path in temporal_files:
        print(f"Matching context for file: {file_path} ===")
        merged_sessions = []
        try:
            # Class name is the prefix of the temporal file name
            class_name = os.path.basename(file_path).split('_')[0]
            data = np.load(file_path, allow_pickle=True)
            count = 1
            starttime = time.time()
            for session in data:
                # Process at most the first 2000 sessions per class
                if count > 2000:
                    break
                if count % 100 == 0:
                    print(f'Matching context for {file_path}, processing session {count}')
                if not isinstance(session, dict):
                    print("Invalid data format: session is not a dict")
                    continue
                count += 1
                try:
                    start_time = float(session["start_time"])
                    five_tuple = session["five_tuple"]

                    # Find the context sessions of the current session
                    contextual = query_sessions(df, None, None,
                                                [(five_tuple[0], five_tuple[2])],
                                                [(start_time - 60, start_time + 60)],
                                                max_context=10, max_window=300,
                                                current_time=start_time)

                    contextual_dict = None if contextual.empty else contextual.to_dict(orient='records')

                    merged_session = session.copy()
                    merged_session['class_name'] = class_name

                    for item in contextual_dict:
                        # Wavelet-transform the packet-length and
                        # time-interval sequences of each context session
                        if 'time_intervals' in item:
                            item["intervals_spectrogram"] = wavelet_transform(
                                item['time_intervals'], wave_name, 64)
                        else:
                            print("Missing 'time_intervals' key in context session.")
                        if 'packet_lens' in item:
                            item["lens_spectrogram"] = wavelet_transform(
                                item['packet_lens'], wave_name, 64)
                        else:
                            print("Missing 'packet_lens' key in context session.")
                    merged_session['contextual'] = contextual_dict

                    merged_sessions.append(merged_session)
                except KeyError as e:
                    print(f"Field {e} not found in session dict")
                except Exception as e:
                    print(f"Error processing session: {str(e)}")
            endtime = time.time()
            print(f'Matching context for {file_path} done, time cost: {endtime - starttime}')
            print(f'Matching context for {file_path} done, total sessions: {len(data)}')
            print(f'Matching context for {file_path} done, average time per session: {(endtime - starttime) / len(data)}')
        except Exception as e:
            print(f"Failed to load file {file_path}: {str(e)}")
        data_name = os.path.join(data_path, class_name + f'_{wave_name}_contextual.npy')
        np.save(data_name, merged_sessions)


def gen_single_traffic_type_data(pcaps_path, class_name, sessions_dir, data_path, wave_name, base_dir,
                                 agg_seqs_path=None, contextual=True):
    """Generate the dataset of one traffic class (merge -> filter -> split ->
    temporal features -> session sequences)."""
    # merge pcaps of a single class
    pcap_file = os.path.join(base_dir, f'{class_name}.pcap')
    if os.path.isdir(pcaps_path):
        merge_pcaps(pcaps_path, pcap_file)
    else:
        ret = os.system(f'cp {pcaps_path} {pcap_file}')
        if ret == 0:
            print(f'copy {pcaps_path} to {pcap_file} successfully')
        else:
            print(f'copy {pcaps_path} to {pcap_file} error')

    filter_protocols_in_pcap(pcap_file)  # keep TCP/UDP only, drop ARP and DHCP
    # extract the byte matrix of each session
    session_pcaps_used = gen_temporal_data(pcap_file, sessions_dir, data_path)
    print(f'{class_name} has {len(session_pcaps_used)} sessions')

    if not contextual:
        return
    # extract the packet-length and time-interval sequences
    starttime = time.time()
    gen_session_feature_sequence(session_pcaps_used, 64, 64, data_path)
    endtime = time.time()
    print(f'{class_name} gen_session_feature_sequence average time cost: ',
          (endtime - starttime) / len(session_pcaps_used))


def gen_multi_traffic_type_data(pcaps_path, data_path, wave_name, base_dir):
    """Generate the dataset of all classes under ``pcaps_path``, then match
    context sessions for every session of every class."""
    pcaps = get_contents_in_dir(pcaps_path, '.', ['.pcap'])
    if os.path.isdir(pcaps[0]):
        # one folder = one class, possibly containing multiple pcaps
        for d in pcaps:
            class_name = os.path.split(d)[1]
            gen_single_traffic_type_data(d, class_name, os.path.join(base_dir, f'{class_name}_sessions'),
                                         os.path.join(data_path, f'{class_name}.npy'), wave_name, base_dir)
    else:
        # one pcap file = one class
        for p in pcaps:
            class_name = os.path.splitext(os.path.split(p)[1])[0]
            gen_single_traffic_type_data(p, class_name, os.path.join(base_dir, f'{class_name}_sessions'),
                                         os.path.join(data_path, f'{class_name}.npy'), wave_name, base_dir)
        # match context sessions for each session of each class
        gen_contextual_data(wave_name, data_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--multiple', action='store_true', required=False,
                        help='Process a directory of classes end-to-end (mode 1)')
    parser.add_argument('--pcaps_path', type=str, required=True,
                        help='Directory of per-class pcaps or folders (mode 1), or a single pcap/folder (mode 2)')
    parser.add_argument('--base_dir', type=str, required=False)
    parser.add_argument('--class_name', type=str, required=False)
    parser.add_argument('--sessions_dir', type=str, required=False)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--wave_name', type=str, required=True)
    args = parser.parse_args()
    print(args)

    # Mode 1: one command to generate the full dataset of all classes
    # python dataset_gen.py --multiple --pcaps_path=/xxx/xxx/ --data_path=/path/to/save \
    #     --wave_name=cgau8 --base_dir=/xxx/xxx/
    if args.multiple:
        gen_multi_traffic_type_data(args.pcaps_path, args.data_path, args.wave_name, args.base_dir)
    # Mode 2: generate the dataset of a single traffic class
    # python dataset_gen.py --pcaps_path=/xxx/xxx/traffic_type --class_name=xxx \
    #     --sessions_dir=/path/to/sessions --data_path=/path/to/save --wave_name=cgau8
    else:
        gen_single_traffic_type_data(args.pcaps_path, args.class_name, args.sessions_dir,
                                     args.data_path, args.wave_name, args.base_dir)
