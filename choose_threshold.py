# choose_threshold.py
"""
Choose Threshold for a Single Feature (ZED)

示例:
  python choose_threshold.py --csv runs/scores_mixK5_T13_P90.csv \
    --feature AbsDelta01_P90 --pos_label 0 --higher_is_better
"""

import argparse
import numpy as np
import pandas as pd



def orient_scores(scores: np.ndarray, higher_is_better: bool) -> np.ndarray:
    return scores if higher_is_better else -scores

def compute_roc(y: np.ndarray, s: np.ndarray, pos_label: int = 0):

    order = np.argsort(-s)
    y = y[order]
    s = s[order]

    P = int(np.sum(y == pos_label))   # 正类数
    N = int(np.sum(y != pos_label))   # 负类数
    if P == 0 or N == 0:
        raise ValueError("正类或负类样本数量为 0，无法计算 ROC。")

    # 累积 TP / FP
    tps = np.cumsum((y == pos_label).astype(np.int64))
    fps = np.cumsum((y != pos_label).astype(np.int64))

    # 仅在分数发生变化处取点，避免重复
    distinct = np.r_[True, s[1:] != s[:-1]]
    tps, fps, thresholds = tps[distinct], fps[distinct], s[distinct]

    TPR = tps / P
    FPR = fps / N

    # 头尾补点，便于 AUC 与插值
    FPR = np.r_[0.0, FPR, 1.0]
    TPR = np.r_[0.0, TPR, 1.0]
    thresholds = np.r_[thresholds[0] + 1e-12, thresholds, thresholds[-1] - 1e-12]
    return FPR, TPR, thresholds

def auc_trapezoid(FPR: np.ndarray, TPR: np.ndarray) -> float:
    """梯形法计算 AUC。"""
    return float(np.trapezoid(TPR, FPR))

# -------------------------------
# 阈值选择与混淆矩阵
# -------------------------------

def best_threshold_by_balacc(y: np.ndarray, s: np.ndarray, thr_list: np.ndarray, pos_label: int = 0):

    best = dict(thr=None, TP=0, FN=0, TN=0, FP=0, TPR=0.0, TNR=0.0, BA=-1.0)
    pos = (y == pos_label)
    neg = ~pos

    for thr in thr_list:
        pred_pos = s >= thr
        TP = int(np.sum(pred_pos & pos))
        FN = int(np.sum((~pred_pos) & pos))
        FP = int(np.sum(pred_pos & neg))
        TN = int(np.sum((~pred_pos) & neg))
        TPR = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        TNR = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        BA  = 0.5 * (TPR + TNR)
        if BA > best["BA"]:
            best.update(thr=float(thr), TP=TP, FN=FN, TN=TN, FP=FP, TPR=float(TPR), TNR=float(TNR), BA=float(BA))
    return best



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True, help="打分 CSV 文件路径")
    ap.add_argument("--feature", type=str, required=True, help="要评估的特征列名（任意列名，如 AbsDelta01_P90）")
    ap.add_argument("--pos_label", type=int, default=0, choices=[0,1], help="正类标签，默认 0=FAKE")
    ap.add_argument("--higher_is_better", action="store_true",
                    help="若指定：分数越大越像正类(假)。否则将自动取负号统一方向。")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if "label" not in df.columns:
        raise ValueError("CSV 缺少 label 列（1=real, 0=fake）")
    if args.feature not in df.columns:
        raise ValueError(f"CSV 不含列: {args.feature}")

    # 读取并清洗
    y = df["label"].to_numpy().astype(int)
    s_raw = df[args.feature].to_numpy().astype(float)
    ok = np.isfinite(s_raw) & np.isin(y, [0,1])
    y, s_raw = y[ok], s_raw[ok]

    # 方向统一：之后统一用“>= thr 判为正类(假)”
    s = orient_scores(s_raw, args.higher_is_better)

    # 计算 ROC / AUC
    FPR, TPR, thresholds = compute_roc(y, s, pos_label=args.pos_label)
    auc = auc_trapezoid(FPR, TPR)

    # 选 Balanced Accuracy 最优阈值（在 thresholds 网格上）
    best = best_threshold_by_balacc(y, s, thresholds, pos_label=args.pos_label)

    # 打印
    print(f"[{args.feature}] AUC={auc:.4f}  Thr={best['thr']:.6f}  Rule: predict POS if score >= Thr")
    print(f"BalancedAcc={best['BA']:.4f} | Confusion @Thr: TP={best['TP']} FN={best['FN']} | TN={best['TN']} FP={best['FP']}")

if __name__ == "__main__":
    main()
