"""
Generic IDS Robustness Evaluator
================================
Required input files
--------------------
baseline.json   – results on the clean dataset
attack.json     – results on the poisoned / adversarial dataset

Both JSON files must contain *integer* fields:
{
  "TP": 123,
  "FP": 4,
  "TN": 567,
  "FN": 8
}

Metrics
-------
DAD : Detection-Accuracy Degradation
FPA : False-Positive Amplification
ESR : Evasion Success Rate

Usage
-----
python customized_eval.py --baseline baseline.json --attack attack.json
"""

import argparse
import json
import os
from typing import Dict


def load_confusion_matrix(path: str) -> Dict[str, int]:
    """Read TP/FP/TN/FN from a json file and uppercase the keys."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {k.upper(): int(v) for k, v in data.items()}


def accuracy(cm: Dict[str, int]) -> float:
    denom = sum(cm.values())
    return (cm["TP"] + cm["TN"]) / denom if denom else 0.0


def fpr(cm: Dict[str, int]) -> float:
    denom = cm["FP"] + cm["TN"]
    return cm["FP"] / denom if denom else 0.0


def esr(cm: Dict[str, int]) -> float:
    denom = cm["TP"] + cm["FN"]
    return cm["FN"] / denom if denom else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, help="clean-run confusion matrix (json)")
    parser.add_argument("--attack",   required=True, help="attack-run confusion matrix (json)")
    parser.add_argument("--eps", type=float, default=0.01,
                        help="epsilon to avoid division by zero (for FPA)")
    args = parser.parse_args()

    if not (os.path.exists(args.baseline) and os.path.exists(args.attack)):
        print("Error: baseline or attack file not found.")
        return

    cm_base = load_confusion_matrix(args.baseline)
    cm_att  = load_confusion_matrix(args.attack)

    # Metrics
    acc_b = accuracy(cm_base)
    acc_a = accuracy(cm_att)
    dad   = (acc_b - acc_a) / acc_b if acc_b else 0.0

    fpr_b = fpr(cm_base)
    fpr_a = fpr(cm_att)
    fpa   = (fpr_a - fpr_b) / (fpr_b + args.eps)

    esr_v = esr(cm_att)  # Use attack confusion matrix

    print("======= Customized Evaluation =======")
    print(f"DAD : {dad:.6f}")
    print(f"FPA : {fpa:.6f}")
    print(f"ESR : {esr_v:.6f}")


if __name__ == "__main__":
    main()
