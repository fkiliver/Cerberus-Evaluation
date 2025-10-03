import argparse
import json
import os
import random
from typing import List, Tuple, Set
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Cerberus IDS Robustness Evaluation Framework")
    parser.add_argument('--dataset', type=str, required=True, help='dataset name (e.g., trace, S1)')
    parser.add_argument('--num', type=int, default=500, help='number of poisoning/mimicry events')
    parser.add_argument('--split', type=float, default=0.7, help='train/test split ratio')
    parser.add_argument('--mode', type=str, choices=['train', 'infer', 'both'], default='both', help='evaluation mode')
    parser.add_argument('--dict_filter', action='store_true', help='Only use nodes from event_dictionary.txt for insertion')
    return parser.parse_args()

def load_malicious_lines(label_path: str) -> Set[int]:
    with open(label_path, 'r', encoding='utf-8') as f:
        return set(int(line.strip()) for line in f if line.strip())

def load_event_dict(dict_path: str) -> Set[str]:
    with open(dict_path, 'r', encoding='utf-8') as f:
        return set(line.strip() for line in f if line.strip())

def is_benign_by_dict(event: dict, event_dict: set) -> bool:
    try:
        proc_name = (
            event.get('datum', {})
                 .get('com.bbn.tc.schema.avro.cdm18.Event', {})
                 .get('properties', {})
                 .get('map', {})
                 .get('proc_name', None)
        )
        if proc_name and proc_name in event_dict:
            return True
    except Exception:
        pass
    return False

def load_dataset_files(dataset_dir: str, malicious_lines: Set[int]) -> List[Tuple[dict, int]]:
    files = sorted([os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir)
                    if f.startswith("ta1-trace-e3-official-1.json")])
    events = []
    line_counter = 1  # Assume label.txt starts from 1
    for fname in files:
        with open(fname, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    label = 1 if line_counter in malicious_lines else 0
                    events.append((obj, label))
                except Exception:
                    pass
                line_counter += 1
    return events

def parse_atlas_s1_logs_with_label(s1_dir: str, label_npy_path: str) -> List[Tuple[dict, int]]:
    """
    Read ATLAS S1 format logs (firefox.txt, dns) and npy labels, return [(event_dict, label)]
    """
    events = []
    firefox_file = os.path.join(s1_dir, "firefox.txt")
    dns_file = os.path.join(s1_dir, "dns")

    # First collect all events in order (consistent with npy index)
    for fname, event_type in [(firefox_file, "firefox_log"), (dns_file, "dns_log")]:
        if os.path.exists(fname):
            with open(fname, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.rstrip('\n')
                    if not line.strip():
                        continue
                    if event_type == "firefox_log":
                        # Try to structure
                        try:
                            ts_split = line.split(" ", 2)
                            timestamp = ts_split[0] + " " + ts_split[1]
                            after_dash = ts_split[2].split("]:", 1)
                            thread = after_dash[0].replace("[", "").replace("]", "")
                            rest = after_dash[1].strip() if len(after_dash) > 1 else ""
                            event = {
                                "type": event_type,
                                "timestamp": timestamp,
                                "thread": thread,
                                "content": rest,
                                "raw": line
                            }
                        except Exception:
                            event = {"type": event_type, "raw": line}
                    else:  # dns_log
                        try:
                            parts = line.split()
                            idx = int(parts[0])
                            timestamp = parts[1] + " " + parts[2]
                            src = parts[3]
                            dst = parts[5]
                            protocol = parts[6]
                            payload = " ".join(parts[7:])
                            event = {
                                "type": event_type,
                                "index": idx,
                                "timestamp": timestamp,
                                "src": src,
                                "dst": dst,
                                "protocol": protocol,
                                "payload": payload,
                                "raw": line
                            }
                        except Exception:
                            event = {"type": event_type, "raw": line}
                    events.append(event)

    # Read npy file (each number is a malicious line number, index starts from 0)
    if os.path.exists(label_npy_path):
        mal_indices = set(np.load(label_npy_path).astype(int).tolist())
    else:
        mal_indices = set()

    # Assign label to each log
    events_with_label = []
    for i, event in enumerate(events):
        label = 1 if i in mal_indices else 0
        events_with_label.append((event, label))

    return events_with_label

def split_data(events: List[Tuple[dict, int]], split_ratio: float):
    benign = [item for item in events if item[1] == 0]
    mal = [item for item in events if item[1] == 1]
    train_size = int(len(benign) * split_ratio)
    train_set = benign[:train_size]
    test_set = benign[train_size:] + mal
    return train_set, test_set

def poison_training(train_set: List[Tuple[dict, int]], mal_set: List[Tuple[dict, int]], num: int, event_dict: set, use_dict: bool):
    if not mal_set:
        print("[!] No malicious nodes available for poisoning")
        return train_set
    # S1 dataset has no threadId field
    if "datum" in mal_set[0][0]:
        mal_event, _ = random.choice(mal_set)
        mal_threadId = mal_event.get('datum', {}).get('com.bbn.tc.schema.avro.cdm18.Event', {}).get('threadId', {}).get('int', None)
        if use_dict:
            # Only use benign nodes from dictionary
            if mal_threadId is not None:
                related_benign = [
                    e for e in train_set
                    if (
                        e[0].get('datum', {}).get('com.bbn.tc.schema.avro.cdm18.Event', {}).get('threadId', {}).get('int', None) == mal_threadId
                        and is_benign_by_dict(e[0], event_dict)
                    )
                ]
            else:
                related_benign = [e for e in train_set if is_benign_by_dict(e[0], event_dict)]
            if not related_benign:
                print("[!] No nodes belonging to event_dict found in training set, downgrading to all benign")
                related_benign = train_set
        else:
            # No restriction, as long as it's benign
            if mal_threadId is not None:
                related_benign = [
                    e for e in train_set
                    if e[0].get('datum', {}).get('com.bbn.tc.schema.avro.cdm18.Event', {}).get('threadId', {}).get('int', None) == mal_threadId
                ]
            else:
                related_benign = train_set
            if not related_benign:
                related_benign = train_set
    else:
        # S1 dataset has no threadId
        if use_dict:
            related_benign = [e for e in train_set if is_benign_by_dict(e[0], event_dict)]
            if not related_benign:
                print("[!] No nodes belonging to event_dict found in training set, downgrading to all benign")
                related_benign = train_set
        else:
            related_benign = train_set

    selected = random.sample(related_benign, min(num, len(related_benign)))
    poisoned_train = train_set.copy()
    for e in selected:
        insert_pos = random.randint(0, len(poisoned_train))
        poisoned_train.insert(insert_pos, e)
    print(f"[*] Poisoning (training phase{' with dictionary filtering' if use_dict else ''} inserted event count: {len(selected)}")
    return poisoned_train

def mimicry_inference(test_set: List[Tuple[dict, int]], mal_set: List[Tuple[dict, int]], num: int, event_dict: set, use_dict: bool):
    if use_dict:
        benign_events = [item for item in test_set if item[1] == 0 and is_benign_by_dict(item[0], event_dict)]
    else:
        benign_events = [item for item in test_set if item[1] == 0]
    mal_events = [item for item in test_set if item[1] == 1]
    if not benign_events or not mal_events:
        print("[!] No poisoning possible in inference phase")
        return test_set
    chosen_benign = random.sample(benign_events, min(num, len(benign_events)))
    chosen_mal = random.choice(mal_events)[0]
    modified_test = test_set.copy()
    try:
        idx = next((i for i, item in enumerate(modified_test) if item[0] == chosen_mal), -1)
    except Exception:
        idx = -1
    if idx == -1:
        idx = len(modified_test) - 1
    for e in chosen_benign:
        modified_test.insert(idx + 1, e)
    print(f"[*] Poisoning (inference phase{' with dictionary filtering' if use_dict else ''} inserted event count: {len(chosen_benign)}")
    return modified_test

def save_json_lines(events: List[Tuple[dict, int]], path: str):
    with open(path, 'w', encoding='utf-8') as f:
        for evt, _ in events:
            json.dump(evt, f)
            f.write('\n')

def save_s1_format(events: List[Tuple[dict, int]], s1_dir: str, output_prefix: str):
    """
    Output in S1 original format: write firefox.txt and dns separately
    :param events: [(event_dict, label)]
    :param s1_dir: Output path, e.g., ./dataset/S1/logs
    :param output_prefix: 'cerberus_train' or 'cerberus_test'
    """
    firefox_lines = []
    dns_lines = []
    for event, _ in events:
        raw = event.get("raw", "")
        if event.get("type") == "firefox_log":
            firefox_lines.append(raw)
        elif event.get("type") == "dns_log":
            dns_lines.append(raw)
    with open(os.path.join(s1_dir, f"{output_prefix}_firefox.txt"), "w", encoding="utf-8") as f1:
        for line in firefox_lines:
            f1.write(line + "\n")
    with open(os.path.join(s1_dir, f"{output_prefix}_dns"), "w", encoding="utf-8") as f2:
        for line in dns_lines:
            f2.write(line + "\n")

def run_framework(args):
    # ATLAS S1 branch
    if args.dataset.lower() == "s1" or args.dataset.lower() == "atlas_s1":
        print("[*] Detected ATLAS S1 dataset format, using S1 parser")
        dataset_dir = f"./dataset/S1/logs"
        label_npy_path = os.path.join(dataset_dir, "S1_number_.npy")
        events = parse_atlas_s1_logs_with_label(dataset_dir, label_npy_path)
        print(f"[*] Total S1 log lines: {len(events)}, malicious: {sum(1 for _, label in events if label == 1)}")
        event_dict = set()
        train_set, test_set = split_data(events, args.split)
        mal_set = [item for item in test_set if item[1] == 1]
        benign_train_set = [item for item in train_set if item[1] == 0]
        if args.mode == 'train':
            print("[*] Training phase poisoning only")
            poisoned_train = poison_training(benign_train_set, mal_set, args.num, event_dict, args.dict_filter)
            save_s1_format(poisoned_train, dataset_dir, "cerberus_train")
            save_s1_format(test_set, dataset_dir, "cerberus_test")
            print(f"[+] Output: {dataset_dir}/cerberus_train_firefox.txt & _dns")
            print(f"[+] Output: {dataset_dir}/cerberus_test_firefox.txt & _dns")
        elif args.mode == 'infer':
            print("[*] Inference phase poisoning only")
            modified_test = mimicry_inference(test_set, mal_set, args.num, event_dict, args.dict_filter)
            save_s1_format(train_set, dataset_dir, "cerberus_train")
            save_s1_format(modified_test, dataset_dir, "cerberus_test")
            print(f"[+] Output: {dataset_dir}/cerberus_train_firefox.txt & _dns")
            print(f"[+] Output: {dataset_dir}/cerberus_test_firefox.txt & _dns")
        elif args.mode == 'both':
            print("[*] Comprehensive poisoning evaluation")
            n1 = args.num // 2
            n2 = args.num - n1
            poisoned_train = poison_training(benign_train_set, mal_set, n1, event_dict, args.dict_filter)
            modified_test = mimicry_inference(test_set, mal_set, n2, event_dict, args.dict_filter)
            save_s1_format(poisoned_train, dataset_dir, "cerberus_train")
            save_s1_format(modified_test, dataset_dir, "cerberus_test")
            print(f"[+] Output: {dataset_dir}/cerberus_train_firefox.txt & _dns")
            print(f"[+] Output: {dataset_dir}/cerberus_test_firefox.txt & _dns")
        return  # Exit after S1 branch

    # Custom dataset branch
    elif args.dataset.lower() == "custom":
        print("[*] Custom dataset detected. Using custom_parser interface.")
        custom_dir = f"./dataset/custom"
        try:
            import importlib.util
            parser_path = os.path.join(os.path.dirname(__file__), "custom_parser.py")
            spec = importlib.util.spec_from_file_location("custom_parser", parser_path)
            custom_parser = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(custom_parser)
            events = custom_parser.parse_custom_dataset(custom_dir)
        except Exception as e:
            print("[!] Error in custom dataset parser:", e)
            return
        print(f"[*] Custom dataset total events: {len(events)}, malicious: {sum(1 for _, label in events if label == 1)}")
        event_dict = set()
        train_set, test_set = split_data(events, args.split)
        mal_set = [item for item in test_set if item[1] == 1]
        benign_train_set = [item for item in train_set if item[1] == 0]
        train_output_path = os.path.join(custom_dir, "cerberus_train.json")
        test_output_path = os.path.join(custom_dir, "cerberus_test.json")
        if args.mode == 'train':
            print("[*] Training phase poisoning only")
            poisoned_train = poison_training(benign_train_set, mal_set, args.num, event_dict, args.dict_filter)
            save_json_lines(poisoned_train, train_output_path)
            save_json_lines(test_set, test_output_path)
            print(f"[+] Output: {train_output_path} training set")
            print(f"[+] Output: {test_output_path} test set")
        elif args.mode == 'infer':
            print("[*] Inference phase poisoning only")
            modified_test = mimicry_inference(test_set, mal_set, args.num, event_dict, args.dict_filter)
            save_json_lines(train_set, train_output_path)
            save_json_lines(modified_test, test_output_path)
            print(f"[+] Output: {train_output_path} training set")
            print(f"[+] Output: {test_output_path} test set")
        elif args.mode == 'both':
            print("[*] Comprehensive poisoning evaluation")
            n1 = args.num // 2
            n2 = args.num - n1
            poisoned_train = poison_training(benign_train_set, mal_set, n1, event_dict, args.dict_filter)
            modified_test = mimicry_inference(test_set, mal_set, n2, event_dict, args.dict_filter)
            save_json_lines(poisoned_train, train_output_path)
            save_json_lines(modified_test, test_output_path)
            print(f"[+] Output: {train_output_path} training set")
            print(f"[+] Output: {test_output_path} test set")
        return  # Exit after custom branch

    # Default: DARPA/trace
    else:
        dataset_dir = f"./dataset/{args.dataset}"
        label_path = os.path.join(dataset_dir, "label.txt")
        dict_path = "event_dictionary.txt"
        print("[*] Reading label, event dictionary, and dataset...")
        malicious_lines = load_malicious_lines(label_path)
        event_dict = load_event_dict(dict_path)
        events = load_dataset_files(dataset_dir, malicious_lines)
        print(f"[*] Total dataset lines: {len(events)}, malicious: {len(malicious_lines)}")
        train_set, test_set = split_data(events, args.split)
        mal_set = [item for item in test_set if item[1] == 1]
        benign_train_set = [item for item in train_set if item[1] == 0]
        train_output_path = os.path.join(dataset_dir, "cerberus_train.json")
        test_output_path = os.path.join(dataset_dir, "cerberus_test.json")
        if args.mode == 'train':
            print("[*] Training phase poisoning only")
            poisoned_train = poison_training(benign_train_set, mal_set, args.num, event_dict, args.dict_filter)
            save_json_lines(poisoned_train, train_output_path)
            save_json_lines(test_set, test_output_path)
            print(f"[+] Output: {train_output_path} training set")
            print(f"[+] Output: {test_output_path} test set")
        elif args.mode == 'infer':
            print("[*] Inference phase poisoning only")
            modified_test = mimicry_inference(test_set, mal_set, args.num, event_dict, args.dict_filter)
            save_json_lines(train_set, train_output_path)
            save_json_lines(modified_test, test_output_path)
            print(f"[+] Output: {train_output_path} training set")
            print(f"[+] Output: {test_output_path} test set")
        elif args.mode == 'both':
            print("[*] Comprehensive poisoning evaluation")
            n1 = args.num // 2
            n2 = args.num - n1
            poisoned_train = poison_training(benign_train_set, mal_set, n1, event_dict, args.dict_filter)
            modified_test = mimicry_inference(test_set, mal_set, n2, event_dict, args.dict_filter)
            save_json_lines(poisoned_train, train_output_path)
            save_json_lines(modified_test, test_output_path)
            print(f"[+] Output: {train_output_path} training set")
            print(f"[+] Output: {test_output_path} test set")


if __name__ == '__main__':
    args = parse_args()
    run_framework(args)
