import os

def parse_airtag_result(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f.readlines()]
    TP = FN = FP = TN = None
    for i, line in enumerate(lines):
        if line == "test1":
            # The following 4 lines are TP FN FP TN respectively
            try:
                TP = int(lines[i+1])
                FN = int(lines[i+2])
                FP = int(lines[i+3])
                TN = int(lines[i+4])
                break
            except Exception:
                continue
    return TP, FP, TN, FN  # Note: order converted to TP,FP,TN,FN (for code reuse)

def calc_acc(TP, TN, FP, FN):
    denom = TP + TN + FP + FN
    return (TP + TN) / denom if denom else 0

def calc_fpr(FP, TN):
    denom = FP + TN
    return FP / denom if denom else 0

def calc_esr(TP, FN):
    denom = TP + FN
    return FN / denom if denom else 0

def compute_metrics_airtag(TP_p, FP_p, TN_p, FN_p, TP_b, FP_b, TN_b, FN_b, eps=0.01):
    acc_p = calc_acc(TP_p, TN_p, FP_p, FN_p)
    acc_b = calc_acc(TP_b, TN_b, FP_b, FN_b)
    fpr_p = calc_fpr(FP_p, TN_p)
    fpr_b = calc_fpr(FP_b, TN_b)
    esr = calc_esr(TP_p, FN_p)
    dad = (acc_b - acc_p) / acc_b if acc_b else 0
    fpa = (fpr_p - fpr_b) / (fpr_b + eps)
    return dad, fpa, esr

def main():
    poison_path = "airtag_result.txt"
    base_path = "airtag_baseline.txt"
    if not os.path.exists(poison_path):
        print(f"File {poison_path} does not exist! Please save airtag's output to this file first.")
        return
    if not os.path.exists(base_path):
        print(f"File {base_path} does not exist! Please save clean set airtag's output to this file first.")
        return
    TP_p, FP_p, TN_p, FN_p = parse_airtag_result(poison_path)
    TP_b, FP_b, TN_b, FN_b = parse_airtag_result(base_path)
    dad, fpa, esr = compute_metrics_airtag(TP_p, FP_p, TN_p, FN_p, TP_b, FP_b, TN_b, FN_b)
    print("=== airtag metrics ===")
    print(f"DAD: {dad:.6f}")
    print(f"FPA: {fpa:.6f}")
    print(f"ESR: {esr:.6f}")

if __name__ == '__main__':
    main()
