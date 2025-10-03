# custom_parser.py

def parse_custom_dataset(custom_dir: str) -> list:
    """
    Template for custom dataset parser.
    Args:
        custom_dir: Path to your custom dataset directory.
    Returns:
        events_with_label: List of (event_dict, label)
            event_dict: JSON-serializable dictionary for each event.
            label: 0 (benign) or 1 (malicious)
    """
    events_with_label = []
    # ===== Implement your own parsing logic below =====
    # Example: each line is a JSON dict in 'custom_log.jsonl', with optional 'label' field
    import os, json, random
    file_path = os.path.join(custom_dir, "custom_log.jsonl")
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                event = json.loads(line)
                # If your events have a 'label' field:
                # label = event.get("label", 0)
                label = random.randint(0, 1)    # Otherwise, assign random labels for example
                events_with_label.append((event, label))
            except Exception:
                continue
    # ===== End of template =====
    return events_with_label
