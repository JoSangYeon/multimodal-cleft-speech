"""
원고 Table 2·3에 대응하는 LOPO 결과표 생성.

기존 형식  : mean ± SD (fold 간)
새 형식    : point estimate (95% CI)
  - point : 34개 fold의 out-of-fold 예측을 pooling(n=1,254)하여 1회 산출
  - CI    : 환자 34명을 복원추출하는 cluster bootstrap 2,000회의 백분위 구간

사용법: python make_manuscript_tables.py [--result_root ./RESULT_LOPO_V2]
"""

import os
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             accuracy_score, recall_score, precision_score,
                             f1_score, brier_score_loss)

# 원고 Table 2·3의 행 순서·표기 그대로
MODELS = [("Audio_Only", "Audio_only"),
          ("Audio_VFS", "Audio_VFS"),
          ("Audio_Tabular", "Audio_Tabular"),
          ("Audio_Video", "Audio_Video"),
          ("Audio_Video_VFS", "Audio_Video_VFS"),
          ("Audio_Video_Tabular", "Audio_Video_Tabular"),
          ("Full_Modality", "Full_Modality")]
METRICS = ["AUROC", "AUPRC", "Accuracy", "Recall", "Precision", "F1-Score", "BRIER"]
LABELS = [("보상조음", "Table 2. Model performance for compensatory articulation across modality combinations"),
          ("과다비성", "Table 3. Model performance for hypernasality across modality combinations")]
BASE = "Audio_Only"


def find_threshold_at_recall(y, p, target=0.70, n=256):
    best_t, best_d, best_f = 0.5, np.inf, -1.0
    for t in np.linspace(0, 1, n):
        pred = (p >= t).astype(int)
        d = abs(recall_score(y, pred, zero_division=0) - target)
        f = f1_score(y, pred, zero_division=0)
        if d < best_d or (d == best_d and f > best_f):
            best_t, best_d, best_f = t, d, f
    return best_t


def metrics(y, p, thr):
    pred = (p >= thr).astype(int)
    both = len(np.unique(y)) == 2
    return {
        "AUROC": roc_auc_score(y, p) if both else np.nan,
        "AUPRC": average_precision_score(y, p) if both else np.nan,
        "Accuracy": accuracy_score(y, pred),
        "Recall": recall_score(y, pred, zero_division=0),
        "Precision": precision_score(y, pred, zero_division=0),
        "F1-Score": f1_score(y, pred, zero_division=0),
        "BRIER": brier_score_loss(y, p),
    }


def fmt(point, lo, hi):
    return f"{point:.2f} ({lo:.2f}–{hi:.2f})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_root", default="./RESULT_LOPO_V2")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--out", default=".docs/major_revision/LOPO_Table2_3.md")
    args = ap.parse_args()

    out = ["# LOPO 기반 Table 2 · Table 3 (원고 대응)", "",
           "기존 `mean ± SD`(fold 간)를 `point estimate (95% CI)`로 대체한다.", "",
           "- **점추정**: 34개 fold의 out-of-fold 예측을 pooling(n = 1,254)하여 1회 산출",
           "- **95% CI**: 환자 34명을 복원추출하는 cluster bootstrap 2,000회의 백분위 구간",
           "- **작동점**: pooled 예측에서 recall ≈ 0.70이 되는 threshold를 1회 결정 후 고정 적용",
           "  (Accuracy·Recall·Precision·F1은 이 작동점 기준)", ""]

    for lab, title in LABELS:
        frames = {k: pd.read_csv(os.path.join(args.result_root, k,
                                              f"inference_results_{lab}.csv"))
                  for k, _ in MODELS}
        ref = frames[BASE]
        y = ref["label"].values.astype(int)
        pids = ref["pid"].values
        for k, df in frames.items():
            assert (df["pid"].values == pids).all(), f"{k}: pid 순서 불일치"
            assert (df["label"].values == ref["label"].values).all(), f"{k}: label 불일치"

        upids = np.unique(pids)
        rows_by_pid = {p: np.flatnonzero(pids == p) for p in upids}
        rng = np.random.default_rng(17)
        boot = [np.concatenate([rows_by_pid[p] for p in
                                rng.choice(upids, len(upids), replace=True)])
                for _ in range(args.n_boot)]

        prev = y.mean()
        out += ["---", "", f"## {title}", "",
                f"검출 대상 양성: 환자 {int(pd.Series(y).groupby(pids).max().sum())}/34, "
                f"발화 {y.sum()}/1,254 ({prev*100:.1f}%) · chance AUPRC = {prev:.3f}", ""]
        out.append("| | " + " | ".join(METRICS) + " |")
        out.append("|" + "---|" * (len(METRICS) + 1))

        boot_auroc, n_undef = {}, 0
        for key, disp in MODELS:
            p = frames[key]["predicted_probas"].values
            thr = find_threshold_at_recall(y, p)
            point = metrics(y, p, thr)

            samp = {m: np.empty(args.n_boot) for m in METRICS}
            for b, idx in enumerate(boot):
                mt = metrics(y[idx], p[idx], thr)
                for m in METRICS:
                    samp[m][b] = mt[m]
            boot_auroc[key] = samp["AUROC"]
            n_undef = int(np.isnan(samp["AUROC"]).sum())

            cells = []
            for m in METRICS:
                v = samp[m][~np.isnan(samp[m])]
                lo, hi = np.percentile(v, [2.5, 97.5])
                cells.append(fmt(point[m], lo, hi))
            out.append(f"| {disp} | " + " | ".join(cells) + " |")

        if n_undef:
            out += ["", f"*부트스트랩 {args.n_boot}회 중 {n_undef}회({n_undef/args.n_boot*100:.2f}%)는 "
                        f"재표집 표본에 양성 발화가 없어 AUROC·AUPRC가 정의되지 않아 해당 지표의 "
                        f"구간 산출에서 제외했다.*"]

        # 모달리티 비교
        out += ["", f"### {title.split('.')[0]}b. Audio_only 대비 AUROC 차이", "",
                "동일한 부트스트랩 환자 표본으로 두 모델을 함께 평가한 paired bootstrap. "
                "개별 신뢰구간의 겹침 여부로 비교하는 것은 타당하지 않으므로 사용하지 않았다.", "",
                "| | ΔAUROC (95% CI) | bootstrap *p* |",
                "|---|---|---|"]
        for key, disp in MODELS:
            if key == BASE:
                continue
            d = boot_auroc[key] - boot_auroc[BASE]
            d = d[~np.isnan(d)]
            lo, hi = np.percentile(d, [2.5, 97.5])
            pv = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
            pt = (roc_auc_score(y, frames[key]["predicted_probas"].values)
                  - roc_auc_score(y, frames[BASE]["predicted_probas"].values))
            out.append(f"| {disp} | {pt:+.3f} ({lo:+.3f} – {hi:+.3f}) | {pv:.3f} |")
        out.append("")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
