"""
LOPO(leave-one-patient-out) 평가 — Reviewer 2 코멘트 1 대응.

기존 5-fold의 "fold별 mean ± SD"를 대체한다.

방법
----
1. 34개 fold의 out-of-fold(OOF) 예측을 모두 pooling(n=1,254)하여 지표를 1회 산출.
   -> 검증셋에 양성이 없어 AUROC가 정의되지 않던 fold 문제가 구조적으로 사라짐.
2. 95% CI는 환자 34명을 복원추출하는 cluster bootstrap(B=2000)으로 산출.
   -> 발화가 환자 내에서 군집되어 있다는 지적(R2-1e)을 동시에 해결.
3. 모달리티 간 비교는 매 replicate에서 동일한 환자 표본으로 두 모델을 함께 평가하는
   paired bootstrap으로 ΔAUROC의 CI와 p값을 산출.
   -> 개별 CI의 겹침 여부로 비교하는 것은 타당하지 않으므로 사용하지 않음(R2-1d).

threshold
---------
recall=0.70 지점을 pooled OOF 전체에서 한 번 결정하고, bootstrap replicate에는
그 값을 고정 적용한다. 따라서 precision/recall/F1의 CI는 "고정된 작동점에서의
표본 변동"으로 해석된다.

사용법
------
  python lopo_eval.py [--result_root ./RESULT_LOPO] [--n_boot 2000]
"""

import os
import glob
import argparse

import numpy as np
import pandas as pd

from sklearn.metrics import (roc_auc_score, average_precision_score,
                             accuracy_score, recall_score, precision_score,
                             f1_score, brier_score_loss)

LABELS = ["보상조음", "과다비성"]
BASELINE = "Audio_Only"
# 표에 넣을 순서 (원고 Table 2/3 순서와 동일)
MODEL_ORDER = ["Audio_Only", "Audio_VFS", "Audio_Tabular", "Audio_Video",
               "Audio_Video_VFS", "Audio_Video_Tabular", "Full_Modality"]


def find_threshold_at_recall(y_true, y_prob, target_recall=0.70, sep_n=256):
    """recall이 target_recall에 가장 가까운 threshold. 동점 시 F1이 높은 쪽."""
    best_t, best_diff, best_f1 = 0.5, np.inf, -1.0
    for t in np.linspace(0, 1, sep_n):
        pred = (y_prob >= t).astype(int)
        diff = abs(recall_score(y_true, pred, zero_division=0) - target_recall)
        f1 = f1_score(y_true, pred, zero_division=0)
        if diff < best_diff or (diff == best_diff and f1 > best_f1):
            best_t, best_diff, best_f1 = t, diff, f1
    return best_t


def compute_metrics(y_true, y_prob, threshold):
    """단일 예측 집합에 대한 전체 지표. 한 클래스만 있으면 순위 기반 지표는 NaN."""
    y_pred = (y_prob >= threshold).astype(int)
    both = len(np.unique(y_true)) == 2
    return {
        "AUROC": roc_auc_score(y_true, y_prob) if both else np.nan,
        "AUPRC": average_precision_score(y_true, y_prob) if both else np.nan,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, zero_division=0),
        "Brier": brier_score_loss(y_true, y_prob),
    }


METRIC_NAMES = ["AUROC", "AUPRC", "Accuracy", "Recall", "Precision", "F1-Score", "Brier"]


def load_pooled(result_root, model, label):
    path = os.path.join(result_root, model, f"inference_results_{label}.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def make_bootstrap_indices(pids_per_row, n_boot, seed=17):
    """
    환자 단위 복원추출 인덱스를 미리 생성.
    모든 모델이 동일한 replicate를 쓰도록 한 번만 만들어 재사용한다(=paired).
    """
    unique_pids = np.unique(pids_per_row)
    pid_to_rows = {p: np.flatnonzero(pids_per_row == p) for p in unique_pids}

    rng = np.random.default_rng(seed)
    boot_idx = []
    for _ in range(n_boot):
        sampled = rng.choice(unique_pids, size=len(unique_pids), replace=True)
        boot_idx.append(np.concatenate([pid_to_rows[p] for p in sampled]))
    return boot_idx, len(unique_pids)


def ci_from_samples(vals, alpha=0.05):
    """NaN을 제외한 percentile CI와 제외 개수."""
    v = np.asarray(vals, dtype=float)
    n_bad = int(np.isnan(v).sum())
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return np.nan, np.nan, n_bad
    lo, hi = np.percentile(v, [alpha / 2 * 100, (1 - alpha / 2) * 100])
    return lo, hi, n_bad


def bootstrap_pvalue(deltas):
    """
    부트스트랩 percentile 검정(양측).
    p = 2 * min(P(Δ* <= 0), P(Δ* >= 0)), 1로 절단.
    """
    d = np.asarray(deltas, dtype=float)
    d = d[~np.isnan(d)]
    if len(d) == 0:
        return np.nan
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return min(p, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_root", type=str, default="./RESULT_LOPO")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--target_recall", type=float, default=0.70)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    models = [m for m in MODEL_ORDER
              if os.path.isdir(os.path.join(args.result_root, m))]
    extra = sorted(d for d in os.listdir(args.result_root)
                   if os.path.isdir(os.path.join(args.result_root, d))
                   and not d.startswith("_") and d not in MODEL_ORDER)
    models += extra
    if not models:
        raise SystemExit(f"{args.result_root} 에 결과 디렉토리가 없습니다.")

    print(f"모델 {len(models)}개: {', '.join(models)}")
    print(f"bootstrap {args.n_boot}회, seed={args.seed}\n")

    for label in LABELS:
        # ---- 데이터 로드 및 정렬 일치 확인 ----
        frames = {}
        for m in models:
            df = load_pooled(args.result_root, m, label)
            if df is None:
                print(f"  [skip] {m} / {label}: 결과 없음")
                continue
            frames[m] = df
        if not frames:
            continue

        ref_key = next(iter(frames))
        ref = frames[ref_key]
        for m, df in frames.items():
            assert len(df) == len(ref), f"{m}: 행 수 불일치"
            assert (df["pid"].values == ref["pid"].values).all(), f"{m}: pid 순서 불일치"
            assert (df["label"].values == ref["label"].values).all(), f"{m}: label 불일치"

        y_true = ref["label"].values.astype(int)
        pids = ref["pid"].values
        prevalence = y_true.mean()

        boot_idx, n_pat = make_bootstrap_indices(pids, args.n_boot, args.seed)

        print("=" * 78)
        print(f"[{label}]  n={len(y_true)}  환자={n_pat}  "
              f"양성={y_true.sum()} ({prevalence*100:.1f}%)  "
              f"chance AUPRC={prevalence:.3f}")
        print("=" * 78)

        # ---- 모델별 점추정 + CI ----
        rows, boot_auroc = [], {}
        for m in models:
            if m not in frames:
                continue
            y_prob = frames[m]["predicted_probas"].values

            thr = find_threshold_at_recall(y_true, y_prob, args.target_recall)
            point = compute_metrics(y_true, y_prob, thr)

            samples = {k: np.empty(args.n_boot) for k in METRIC_NAMES}
            for b, idx in enumerate(boot_idx):
                mt = compute_metrics(y_true[idx], y_prob[idx], thr)
                for k in METRIC_NAMES:
                    samples[k][b] = mt[k]
            boot_auroc[m] = samples["AUROC"]

            row = {"Model": m, "threshold": round(thr, 4)}
            for k in METRIC_NAMES:
                lo, hi, n_bad = ci_from_samples(samples[k])
                row[k] = f"{point[k]:.3f} ({lo:.3f}–{hi:.3f})"
                if k == "AUROC":
                    row["_undefined_replicates"] = n_bad
                if k == "AUPRC":
                    # 유병률이 낮아 AUPRC 절대값만으로는 판별력을 읽을 수 없다(R2-1d).
                    # chance 수준(=유병률) 대비 몇 배인지 함께 보고한다.
                    row["AUPRC/chance"] = f"{point[k] / prevalence:.2f}x"
            rows.append(row)

        res = pd.DataFrame(rows)
        res.insert(1, "chance AUPRC", f"{prevalence:.3f}")
        out = os.path.join(args.result_root, f"lopo_result_{label}.csv")
        res.to_csv(out, index=False, encoding="utf-8-sig")
        print(res.to_string(index=False))
        print(f"\n-> {out}")

        # ---- baseline 대비 paired ΔAUROC ----
        if BASELINE in boot_auroc:
            base_point = roc_auc_score(y_true, frames[BASELINE]["predicted_probas"].values)
            comp = []
            for m in models:
                if m not in boot_auroc or m == BASELINE:
                    continue
                point = roc_auc_score(y_true, frames[m]["predicted_probas"].values)
                deltas = boot_auroc[m] - boot_auroc[BASELINE]   # 동일 replicate = paired
                lo, hi, n_bad = ci_from_samples(deltas)
                comp.append({
                    "Model": m,
                    "AUROC": f"{point:.3f}",
                    f"vs {BASELINE}": f"{base_point:.3f}",
                    "ΔAUROC (95% CI)": f"{point - base_point:+.3f} ({lo:+.3f}–{hi:+.3f})",
                    "bootstrap p": f"{bootstrap_pvalue(deltas):.3f}",
                    "CI가 0을 포함": "예" if lo <= 0 <= hi else "아니오",
                })
            cdf = pd.DataFrame(comp)
            cout = os.path.join(args.result_root, f"lopo_comparison_{label}.csv")
            cdf.to_csv(cout, index=False, encoding="utf-8-sig")
            print(f"\n--- {BASELINE} 대비 paired ΔAUROC ---")
            print(cdf.to_string(index=False))
            print(f"\n-> {cout}")
        print()


if __name__ == "__main__":
    main()
