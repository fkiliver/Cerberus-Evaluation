import os

def read_magic_result(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    results = {}
    for line in lines:
        for key in ["AUC", "F1", "PRECISION", "RECALL", "TN", "FN", "TP", "FP"]:
            if line.strip().startswith(key + ":"):
                val = line.strip().split(":")[1].strip()
                try:
                    results[key] = float(val)
                except ValueError:
                    results[key] = val
    return results

def calc_acc(TP, TN, FP, FN):
    denom = TP + TN + FP + FN
    return (TP + TN) / denom if denom else 0

def calc_fpr(FP, TN):
    denom = FP + TN
    return FP / denom if denom else 0

def calc_esr(TP, FN):
    denom = TP + FN
    return FN / denom if denom else 0

def compute_metrics_magic(poisoned, baseline, eps=0.01):
    TP_p = poisoned.get("TP", 0)
    FP_p = poisoned.get("FP", 0)
    TN_p = poisoned.get("TN", 0)
    FN_p = poisoned.get("FN", 0)
    TP_b = baseline.get("TP", 0)
    FP_b = baseline.get("FP", 0)
    TN_b = baseline.get("TN", 0)
    FN_b = baseline.get("FN", 0)
    # Accuracy
    acc_p = calc_acc(TP_p, TN_p, FP_p, FN_p)
    acc_b = calc_acc(TP_b, TN_b, FP_b, FN_b)
    # FPR
    fpr_p = calc_fpr(FP_p, TN_p)
    fpr_b = calc_fpr(FP_b, TN_b)
    # ESR (Always use TP/FN under poisoning)
    esr = calc_esr(TP_p, FN_p)
    # DAD
    dad = (acc_b - acc_p) / acc_b if acc_b else 0
    # FPA
    fpa = (fpr_p - fpr_b) / (fpr_b + eps)
    return dad, fpa, esr

def main():
    poison_path = "magic_result.txt"
    base_path = "magic_baseline.txt"
    if not os.path.exists(poison_path):
        print(f"File {poison_path} does not exist! Please save magic's output to this file first.")
        return
    if not os.path.exists(base_path):
        print(f"File {base_path} does not exist! Please save clean set magic's output to this file first.")
        return
    poison = read_magic_result(poison_path)
    base = read_magic_result(base_path)
    dad, fpa, esr = compute_metrics_magic(poison, base)
    print("=== magic metrics ===")
    print(f"DAD: {dad:.6f}")
    print(f"FPA: {fpa:.6f}")
    print(f"ESR: {esr:.6f}")

if __name__ == '__main__':
    main()
