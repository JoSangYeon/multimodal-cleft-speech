"""
LOPO 기반 ROC / Precision-Recall 곡선 — 원고 Figure 2 대응.

기존 Figure 2는 fold별 곡선을 보간해 평균 ± SD 밴드로 그렸으나, LOPO는 fold당
검증 환자가 1명이라 fold별 곡선이 정의되지 않는다. 대신 34개 fold의 out-of-fold
예측을 모두 모아(n = 1,254) 곡선을 하나씩 그린다.

색은 CVD 검증을 통과한 7색 고정 순서를 쓰고, 인쇄·흑백·색각이상 대비를 위해
선 스타일을 보조 부호화로 함께 사용한다.

사용법: python plot_lopo_curves.py [--result_root ./RESULT_LOPO_V2]
"""

import os
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_curve, precision_recall_curve,
                             roc_auc_score, average_precision_score)

# 원고 Table 2·3의 표기·순서 그대로 (Editor E3: 용어 일관성)
MODELS = [
    ("Audio_Only",          "Audio_only"),
    ("Audio_VFS",           "Audio_VFS"),
    ("Audio_Tabular",       "Audio_Tabular"),
    ("Audio_Video",         "Audio_Video"),
    ("Audio_Video_VFS",     "Audio_Video_VFS"),
    ("Audio_Video_Tabular", "Audio_Video_Tabular"),
    ("Full_Modality",       "Full_Modality"),
]

# CVD 검증 통과 (worst adjacent ΔE 9.1 protan). 고정 순서, 순환하지 않음.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
# tritan 분리가 floor 구간(5.8)이라 보조 부호화 필수
DASHES = [(None, None), (5, 2), (1.5, 1.5), (7, 2, 1.5, 2),
          (3, 1.5), (7, 2, 1.5, 2, 1.5, 2), (2.5, 1.5, 1, 1.5)]

LABELS = [("보상조음", "compensatory articulation"),
          ("과다비성", "hypernasality")]

INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#d8d7d2"
CHANCE = "#8a8981"


def setup_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.unicode_minus": False,
        "axes.edgecolor": GRID,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def load(result_root, model, label):
    return pd.read_csv(os.path.join(result_root, model,
                                    f"inference_results_{label}.csv"))


def style_axes(ax, ymax=1.0):
    ax.grid(True, color=GRID, linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02 * ymax, 1.02 * ymax)
    ax.set_aspect(1.0 / ymax)


def pr_ylim(y, probs, floor_recall=0.05):
    """
    PR 곡선의 y 상한. recall이 극히 낮은 구간의 퇴화 구간(첫 샘플 하나로
    precision=1.0이 되는 지점)을 제외한 실제 최대 precision을 기준으로 잡는다.
    유병률이 5% 수준이라 0~1 전체를 쓰면 곡선이 바닥에 깔려 판독이 불가능하다.
    """
    m = 0.0
    for p in probs:
        pr, rc, _ = precision_recall_curve(y, p)
        sel = rc[:-1] >= floor_recall          # 마지막 원소는 recall=0 sentinel
        if sel.any():
            m = max(m, pr[:-1][sel].max())
    return min(1.0, np.ceil((m * 1.15) * 20) / 20)   # 0.05 단위로 올림


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_root", default="./RESULT_LOPO_V2")
    ap.add_argument("--out_dir", default="./RESULT_LOPO_V2/figures")
    args = ap.parse_args()
    setup_style()
    os.makedirs(args.out_dir, exist_ok=True)

    panel = ["A", "B", "C", "D"]
    saved = []

    for row, (lab, lab_en) in enumerate(LABELS):
        frames = {k: load(args.result_root, k, lab) for k, _ in MODELS}
        ref = frames["Audio_Only"]
        y = ref["label"].values.astype(int)
        prev = y.mean()
        for k, df in frames.items():
            assert (df["pid"].values == ref["pid"].values).all(), f"{k}: pid 순서 불일치"
            assert (df["label"].values == ref["label"].values).all(), f"{k}: label 불일치"

        fig_roc, ax_roc = plt.subplots(figsize=(5.4, 5.0))
        fig_pr, ax_pr = plt.subplots(figsize=(5.4, 5.0))

        for i, (key, disp) in enumerate(MODELS):
            p = frames[key]["predicted_probas"].values
            c, dash = COLORS[i], DASHES[i]

            fpr, tpr, _ = roc_curve(y, p)
            ax_roc.plot(fpr, tpr, color=c, linewidth=2.0, dashes=dash,
                        label=f"{disp}  {roc_auc_score(y, p):.2f}",
                        solid_capstyle="round", dash_capstyle="round")

            pr, rc, _ = precision_recall_curve(y, p)
            ax_pr.plot(rc, pr, color=c, linewidth=2.0, dashes=dash,
                       label=f"{disp}  {average_precision_score(y, p):.2f}",
                       solid_capstyle="round", dash_capstyle="round")

        ax_roc.plot([0, 1], [0, 1], color=CHANCE, linewidth=1.2,
                    dashes=(3, 3), label="Chance  0.50", zorder=0)
        ax_pr.axhline(prev, color=CHANCE, linewidth=1.2, dashes=(3, 3),
                      label=f"Chance  {prev:.3f}", zorder=0)

        ymax = pr_ylim(y, [frames[k]["predicted_probas"].values for k, _ in MODELS])
        style_axes(ax_roc)
        style_axes(ax_pr, ymax=ymax)
        ax_roc.set_xlabel("False positive rate")
        ax_roc.set_ylabel("True positive rate")
        ax_pr.set_xlabel("Recall")
        ax_pr.set_ylabel("Precision")

        # 범례 텍스트는 잉크색으로 (계열색을 글자에 쓰지 않는다)
        for ax, loc in ((ax_roc, "lower right"), (ax_pr, "upper right")):
            leg = ax.legend(loc=loc, fontsize=7.6, labelspacing=0.42,
                            handlelength=2.6, borderpad=0.5)
            for t in leg.get_texts():
                t.set_color(INK)

        n_pos_pt = int(pd.Series(y).groupby(ref["pid"].values).max().sum())
        tag = "compensatory" if row == 0 else "hypernasality"
        for f, ax, kind, pl in ((fig_roc, ax_roc, "roc", panel[row * 2]),
                                (fig_pr, ax_pr, "pr", panel[row * 2 + 1])):
            f.tight_layout()
            for ext in ("png", "pdf"):
                path = os.path.join(args.out_dir,
                                    f"figure2{pl}_{kind}_{tag}.{ext}")
                f.savefig(path, dpi=400, bbox_inches="tight")
                if ext == "png":
                    saved.append(path)
            plt.close(f)

        print(f"[{lab_en}] precision axis truncated at {ymax:.2f} "
              f"(positive {n_pos_pt}/34 patients, {y.sum()} utterances, {prev*100:.1f}%)")

    for path in saved:
        print(f"-> {path}  (+ .pdf)")


if __name__ == "__main__":
    main()
