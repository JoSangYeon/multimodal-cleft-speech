import os
import glob
import json
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 한글 폰트 설정 (Pretendard)
fm.fontManager.addfont('/usr/share/fonts/truetype/Pretendard-Regular.ttf')
fm.fontManager.addfont('/usr/share/fonts/truetype/Pretendard-Bold.ttf')
plt.rcParams['font.family'] = 'Pretendard'
plt.rcParams['axes.unicode_minus'] = False

from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import (roc_auc_score, 
                             average_precision_score,
                             accuracy_score,  
                             recall_score,
                             precision_score,
                             f1_score,
                             brier_score_loss,) # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.brier_score_loss.html
from transformers import ViTFeatureExtractor

from utils import (str2bool, make_dirs)
from model import *

import warnings
warnings.filterwarnings(action='ignore')

# ============================================================
# Display name mapping & legend order
# ============================================================
DISPLAY_NAME = {
    'Audio_Only':          'Audio',
    'Audio_Tabular':       'Audio + Tabular',
    'Audio_Video':         'Audio + Video',
    'Audio_VFS':           'Audio + VFS',
    'Audio_Video_Tabular': 'Audio + Video + Tabular',
    'Audio_Video_VFS':     'Audio + Video + VFS',
    'Full_Modality':       'Full (A+V+VFS+T)',
}

# 모달리티 수 기준 정렬 순서
LEGEND_ORDER = [
    'Audio_Only',
    'Audio_Tabular',
    'Audio_Video',
    'Audio_VFS',
    'Audio_Video_Tabular',
    'Audio_Video_VFS',
    'Full_Modality',
]

def _display(model_name):
    """디렉토리명 → legend 표시명."""
    return DISPLAY_NAME.get(model_name, model_name)

def _sort_key(model_name):
    """LEGEND_ORDER 기준 정렬 키."""
    if model_name in LEGEND_ORDER:
        return LEGEND_ORDER.index(model_name)
    return 999

def thresholding(data, sep_n=256):
    threshold_list = np.linspace(0, 1, sep_n)

    results = {
        "threshold": [],
        'AUROC': [],
        'AUPRC': [],
        "ACCURACY": [],
        "RECALL": [],
        "PRECISION": [],
        "F1-Score": [],
        "BRIER": [],
    }

    pbar = tqdm(threshold_list)
    for threshold in pbar:
        pbar.set_description(f"Threshold: {threshold:.4f}")
        
        y_true = data.label.values
        y_prob = data.predicted_probas.values
        y_labe = (y_prob >= threshold).astype(int)
        
        results["threshold"].append(threshold)
        results['AUROC'].append(roc_auc_score(y_true, y_prob))
        results['AUPRC'].append(average_precision_score(y_true, y_prob))
        results["ACCURACY"].append(accuracy_score(y_true, y_labe))
        results["RECALL"].append(recall_score(y_true, y_labe))
        results["PRECISION"].append(precision_score(y_true, y_labe))
        results["F1-Score"].append(f1_score(y_true, y_labe))
        results["BRIER"].append(brier_score_loss(y_true, y_prob))

    results_df = pd.DataFrame(results)
    return results_df

def _find_threshold_at_recall(y_true, y_prob, target_recall=0.70, sep_n=256):
    """
    주어진 데이터에서 recall이 target_recall에 가장 가까운 threshold를 반환.
    동점 시 F1이 높은 threshold 선택.
    """
    thresholds = np.linspace(0, 1, sep_n)
    best_t, best_diff, best_f1 = 0.5, 999.0, -1.0
    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        rec = recall_score(y_true, pred, zero_division=0)
        f1 = f1_score(y_true, pred, zero_division=0)
        diff = abs(rec - target_recall)
        if diff < best_diff or (diff == best_diff and f1 > best_f1):
            best_t, best_diff, best_f1 = t, diff, f1
    return best_t


def kfold_metrics(logits_path, threshold, round_digit=4,
                  target_recall=0.70):
    """
    K-fold 교차 검증 기반 메트릭 계산.
    - AUROC, AUPRC, BRIER: threshold 무관 (확률 기반)
    - ACCURACY, RECALL, PRECISION, F1-Score: 각 fold별로
      recall ≈ target_recall인 threshold를 독립적으로 찾아서 계산
      (fold 간 확률 스케일 차이 문제 방지)
    """
    parent_dir = os.path.dirname(logits_path)
    label_name = os.path.basename(logits_path)[:-4].split("_")[-1]  # 보상조음 or 과다비성

    # fold 디렉토리 자동 탐색
    fold_dirs = sorted(glob.glob(os.path.join(parent_dir, 'fold_*')))

    metrics = ['AUROC', 'AUPRC', 'ACCURACY', 'RECALL', 'PRECISION', 'F1-Score', 'BRIER']
    fold_results = {m: [] for m in metrics}
    fold_thresholds = []

    for fold_dir in fold_dirs:
        fold_path = os.path.join(fold_dir, f'inference_results_{label_name}.csv')
        if not os.path.exists(fold_path):
            continue
        data = pd.read_csv(fold_path)

        y_true = data.label.values
        y_prob = data.predicted_probas.values

        # 단일 클래스 fold는 AUROC/AUPRC 계산 불가 → skip
        if len(np.unique(y_true)) < 2:
            continue

        # fold별 recall ≈ target_recall인 threshold 탐색
        fold_t = _find_threshold_at_recall(y_true, y_prob,
                                           target_recall=target_recall)
        fold_thresholds.append(fold_t)
        y_pred = (y_prob >= fold_t).astype(int)

        fold_results['AUROC'].append(roc_auc_score(y_true, y_prob))
        fold_results['AUPRC'].append(average_precision_score(y_true, y_prob))
        fold_results['ACCURACY'].append(accuracy_score(y_true, y_pred))
        fold_results['RECALL'].append(recall_score(y_true, y_pred, zero_division=0))
        fold_results['PRECISION'].append(precision_score(y_true, y_pred, zero_division=0))
        fold_results['F1-Score'].append(f1_score(y_true, y_pred, zero_division=0))
        fold_results['BRIER'].append(brier_score_loss(y_true, y_prob))

    # Summary: mean ± std across folds
    print(logits_path)
    n_valid_folds = len(fold_results['AUROC'])
    print(f'\t({n_valid_folds}/{len(fold_dirs)} folds with both classes)')
    if fold_thresholds:
        t_arr = np.array(fold_thresholds)
        print(f'\tPer-fold thresholds (target recall={target_recall}): '
              f'{t_arr.mean():.3f} ± {t_arr.std():.3f}  {[round(t,3) for t in fold_thresholds]}')
    summary = {}
    for m in metrics:
        arr = np.array(fold_results[m])
        if len(arr) >= 2:
            summary[m] = f"{arr.mean():.{round_digit}f} ± {arr.std():.{round_digit}f}"
        elif len(arr) == 1:
            summary[m] = f"{arr[0]:.{round_digit}f}"
        else:
            summary[m] = "N/A"
        print(f'\t{m}: {summary[m]}')
    print()

    return summary

def get_opt_threshold(logits_path, sep_n=256, target_metric='RECALL', 
                      target_value=0.70, sort_metric='F1-Score'):
    parent_dir = os.path.dirname(logits_path)
    os.makedirs(parent_dir, exist_ok=True)

    data = pd.read_csv(logits_path)
    result_df = thresholding(data, sep_n=sep_n)

    label_name = os.path.basename(logits_path)[:-4].split("_")[-1]
    result_df.to_csv(os.path.join(parent_dir, f'thresholding_{label_name}.csv'), index=False)

    result_df['metric_diff'] = abs(result_df[target_metric] - target_value)
    sorted_df = result_df.sort_values(by=['metric_diff', sort_metric], ascending=[True, False])
    opt_threshold = float(sorted_df.threshold.values[0])

    return opt_threshold

def ROC_PRC_subplot(logits_path_list, threshold_list,
                    round_digit=4, LABEL_NAME="보상조음",
                    target_recall=0.70, show_std_band=True):
    """
    Draw ROC and PRC curves side-by-side using subplots.
    각 fold별 curve를 interpolation하여 mean curve (± std band) 로 표현.
    """
    mean_fpr = np.linspace(0, 1, 200)
    mean_recall = np.linspace(0, 1, 200)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    metric_df = []
    model_list = []
    last_no_skill = 0.0

    # 수집 후 정렬하기 위해 임시 저장
    curve_data = []

    for logits_path, threshold in zip(logits_path_list, threshold_list):
        model_name = logits_path.split('/')[1]
        label_name = os.path.basename(logits_path)[:-4].split("_")[-1]
        if label_name != LABEL_NAME:
            continue

        result_dict = kfold_metrics(
            logits_path, threshold, round_digit=round_digit,
            target_recall=target_recall)
        metric_df.append(result_dict)
        model_list.append(model_name)

        parent_dir = os.path.dirname(logits_path)
        fold_dirs = sorted(glob.glob(os.path.join(parent_dir, 'fold_*')))

        tpr_list = []
        prec_list = []
        for fold_dir in fold_dirs:
            fold_path = os.path.join(fold_dir, f'inference_results_{label_name}.csv')
            if not os.path.exists(fold_path):
                continue
            data = pd.read_csv(fold_path)
            y_true = data.label.values
            y_prob = data.predicted_probas.values
            if len(np.unique(y_true)) < 2:
                continue

            fpr_i, tpr_i, _ = roc_curve(y_true, y_prob)
            interp_tpr = np.interp(mean_fpr, fpr_i, tpr_i)
            interp_tpr[0] = 0.0
            tpr_list.append(interp_tpr)

            prec_i, rec_i, _ = precision_recall_curve(y_true, y_prob)
            interp_prec = np.interp(mean_recall, rec_i[::-1], prec_i[::-1])
            prec_list.append(interp_prec)

            last_no_skill = np.sum(y_true == 1) / len(y_true)

        if not tpr_list:
            continue

        curve_data.append({
            'model_name': model_name,
            'tpr_list': tpr_list,
            'prec_list': prec_list,
        })

    # LEGEND_ORDER 기준 정렬
    curve_data.sort(key=lambda x: _sort_key(x['model_name']))

    for idx, cd in enumerate(curve_data):
        model_name = cd['model_name']
        display_name = _display(model_name)
        color = f'C{idx}'

        mean_tpr = np.mean(cd['tpr_list'], axis=0)
        std_tpr = np.std(cd['tpr_list'], axis=0)
        axes[0].plot(mean_fpr, mean_tpr, linewidth=2.0, label=display_name, color=color)
        if show_std_band:
            axes[0].fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr,
                                 alpha=0.15, color=color)

        mean_prec = np.mean(cd['prec_list'], axis=0)
        std_prec = np.std(cd['prec_list'], axis=0)
        axes[1].plot(mean_recall, mean_prec, linewidth=2.0, label=display_name, color=color)
        if show_std_band:
            axes[1].fill_between(mean_recall, mean_prec - std_prec, mean_prec + std_prec,
                                 alpha=0.15, color=color)

    # ROC subplot
    std_tag = ' (K-fold mean ± std)' if show_std_band else ' (K-fold mean)'
    axes[0].plot([0, 1], [0, 1], linestyle='--', linewidth=1.5, color='black', label='No Skill')
    axes[0].set_title(f'ROC Curve{std_tag}', fontsize=15)
    axes[0].tick_params(labelsize=13)
    axes[0].set_ylim([-0.00, 1.001])
    axes[0].set_xlim([-0.01, 1])
    axes[0].set_xlabel('False Positive Rate', fontsize=13)
    axes[0].set_ylabel('True Positive Rate', fontsize=13)
    axes[0].legend(loc='lower right', fontsize=8)

    # PRC subplot
    axes[1].axhline(last_no_skill, linestyle='--', linewidth=1.5, color='black', label=f'No Skill ({last_no_skill:.3f})')
    axes[1].set_title(f'Precision-Recall Curve{std_tag}', fontsize=15)
    axes[1].tick_params(labelsize=13)
    axes[1].set_ylim([-0.00, 1.01])
    axes[1].set_xlim([-0.01, 1.00])
    axes[1].set_xlabel('Recall', fontsize=13)
    axes[1].set_ylabel('Precision', fontsize=13)
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    std_suffix = '' if show_std_band else '_no_std'
    save_root = os.path.dirname(os.path.dirname(logits_path))
    plt.savefig(os.path.join(save_root, f'roc_prc_subplot_{LABEL_NAME}{std_suffix}.png'), dpi=300)
    plt.show()
    plt.close()

    # ---- 개별 파일 저장 (ROC / PRC 각각) ----
    for plot_type in ['roc', 'prc']:
        fig_s, ax_s = plt.subplots(1, 1, figsize=(7, 6))

        for idx, cd in enumerate(curve_data):
            display_name = _display(cd['model_name'])
            color = f'C{idx}'

            if plot_type == 'roc':
                mean_vals = np.mean(cd['tpr_list'], axis=0)
                std_vals = np.std(cd['tpr_list'], axis=0)
                ax_s.plot(mean_fpr, mean_vals, linewidth=2.0, label=display_name, color=color)
                if show_std_band:
                    ax_s.fill_between(mean_fpr, mean_vals - std_vals, mean_vals + std_vals,
                                      alpha=0.15, color=color)
            else:
                mean_vals = np.mean(cd['prec_list'], axis=0)
                std_vals = np.std(cd['prec_list'], axis=0)
                ax_s.plot(mean_recall, mean_vals, linewidth=2.0, label=display_name, color=color)
                if show_std_band:
                    ax_s.fill_between(mean_recall, mean_vals - std_vals, mean_vals + std_vals,
                                      alpha=0.15, color=color)

        if plot_type == 'roc':
            ax_s.plot([0, 1], [0, 1], linestyle='--', linewidth=1.5, color='black', label='No Skill')
            ax_s.set_title(f'ROC Curve{std_tag}', fontsize=15)
            ax_s.set_xlabel('False Positive Rate', fontsize=13)
            ax_s.set_ylabel('True Positive Rate', fontsize=13)
            ax_s.set_ylim([-0.00, 1.001])
            ax_s.set_xlim([-0.01, 1])
            ax_s.legend(loc='lower right', fontsize=9)
        else:
            ax_s.axhline(last_no_skill, linestyle='--', linewidth=1.5, color='black',
                         label=f'No Skill ({last_no_skill:.3f})')
            ax_s.set_title(f'Precision-Recall Curve{std_tag}', fontsize=15)
            ax_s.set_xlabel('Recall', fontsize=13)
            ax_s.set_ylabel('Precision', fontsize=13)
            ax_s.set_ylim([-0.00, 1.01])
            ax_s.set_xlim([-0.01, 1.00])
            ax_s.legend(fontsize=9)

        ax_s.tick_params(labelsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(save_root, f'{plot_type}_{LABEL_NAME}{std_suffix}.png'), dpi=300)
        plt.close()

    return metric_df, model_list

def collect_kfold_numeric(logits_path_list, threshold_list, LABEL_NAME="보상조음",
                          target_recall=0.70):
    """
    각 실험의 fold-level 수치를 수집하여 {model: {metric: (mean, std, values)}} 반환.
    fold별 recall≈target_recall인 threshold로 F1 계산.
    """
    results = {}
    for logits_path, threshold in zip(logits_path_list, threshold_list):
        label_name = os.path.basename(logits_path)[:-4].split("_")[-1]
        if label_name != LABEL_NAME:
            continue
        model_name = logits_path.split('/')[1]
        parent_dir = os.path.dirname(logits_path)
        fold_dirs = sorted(glob.glob(os.path.join(parent_dir, 'fold_*')))

        fold_data = {'AUROC': [], 'AUPRC': [], 'F1-Score': []}
        prevalence = None
        for fold_dir in fold_dirs:
            fold_path = os.path.join(fold_dir, f'inference_results_{label_name}.csv')
            if not os.path.exists(fold_path):
                continue
            data = pd.read_csv(fold_path)
            y_true = data.label.values
            y_prob = data.predicted_probas.values
            if len(np.unique(y_true)) < 2:
                continue
            fold_t = _find_threshold_at_recall(y_true, y_prob,
                                               target_recall=target_recall)
            y_pred = (y_prob >= fold_t).astype(int)
            fold_data['AUROC'].append(roc_auc_score(y_true, y_prob))
            fold_data['AUPRC'].append(average_precision_score(y_true, y_prob))
            fold_data['F1-Score'].append(f1_score(y_true, y_pred, zero_division=0))

        # prevalence from aggregated data
        agg = pd.read_csv(logits_path)
        prevalence = agg.label.mean()

        numeric = {}
        for m in fold_data:
            arr = np.array(fold_data[m])
            numeric[m] = (arr.mean(), arr.std(), arr) if len(arr) > 0 else (np.nan, np.nan, arr)
        numeric['prevalence'] = prevalence
        results[model_name] = numeric

    return results


def plot_delta_bar(all_numeric, LABEL_NAME="보상조음", baseline_key="Audio_only", save_dir="RESULT"):
    """
    Audio Only baseline 대비 AUROC/AUPRC 향상폭(delta) bar chart.
    """
    if baseline_key not in all_numeric:
        print(f"[Warning] baseline '{baseline_key}' not found, skipping delta bar chart.")
        return

    base_auroc = all_numeric[baseline_key]['AUROC'][0]
    base_auprc = all_numeric[baseline_key]['AUPRC'][0]

    models = [k for k in all_numeric if k != baseline_key]
    if not models:
        return

    delta_auroc = [all_numeric[m]['AUROC'][0] - base_auroc for m in models]
    delta_auprc = [all_numeric[m]['AUPRC'][0] - base_auprc for m in models]
    std_auroc = [all_numeric[m]['AUROC'][1] for m in models]
    std_auprc = [all_numeric[m]['AUPRC'][1] for m in models]

    # sort by delta AUROC descending
    order = np.argsort(delta_auroc)[::-1]
    models = [models[i] for i in order]
    delta_auroc = [delta_auroc[i] for i in order]
    delta_auprc = [delta_auprc[i] for i in order]
    std_auroc = [std_auroc[i] for i in order]
    std_auprc = [std_auprc[i] for i in order]

    # short display names
    display = [m.replace('Audio_', 'A+').replace('Video', 'V').replace('Tabular', 'T')
                .replace('Full_Modality', 'Full(A+V+D+T)').replace('DSR', 'D')
               for m in models]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(models))
    w = 0.55

    # --- AUROC delta ---
    colors_auroc = ['#2ecc71' if d > 0 else '#e74c3c' for d in delta_auroc]
    axes[0].bar(x, delta_auroc, w, yerr=std_auroc, capsize=4,
                color=colors_auroc, edgecolor='white', linewidth=0.8)
    axes[0].axhline(0, color='black', linewidth=1.0, linestyle='-')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(display, fontsize=11, rotation=20, ha='right')
    axes[0].set_ylabel('$\\Delta$ AUROC (vs Audio Only)', fontsize=13)
    axes[0].set_title(f'{LABEL_NAME} — AUROC Improvement', fontsize=14)
    for i, v in enumerate(delta_auroc):
        axes[0].text(i, v + std_auroc[i] + 0.005 if v >= 0 else v - std_auroc[i] - 0.02,
                     f'{v:+.3f}', ha='center', fontsize=10, fontweight='bold')

    # --- AUPRC delta ---
    colors_auprc = ['#2ecc71' if d > 0 else '#e74c3c' for d in delta_auprc]
    axes[1].bar(x, delta_auprc, w, yerr=std_auprc, capsize=4,
                color=colors_auprc, edgecolor='white', linewidth=0.8)
    axes[1].axhline(0, color='black', linewidth=1.0, linestyle='-')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(display, fontsize=11, rotation=20, ha='right')
    axes[1].set_ylabel('$\\Delta$ AUPRC (vs Audio Only)', fontsize=13)
    axes[1].set_title(f'{LABEL_NAME} — AUPRC Improvement', fontsize=14)
    for i, v in enumerate(delta_auprc):
        axes[1].text(i, v + std_auprc[i] + 0.005 if v >= 0 else v - std_auprc[i] - 0.02,
                     f'{v:+.3f}', ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'delta_bar_{LABEL_NAME}.png'), dpi=300)
    plt.show()
    plt.close()


def plot_lift_bar(all_numeric, LABEL_NAME="보상조음", save_dir="RESULT"):
    """
    AUPRC를 No-Skill(=prevalence) 대비 Lift 배수로 시각화.
    """
    models = list(all_numeric.keys())
    if not models:
        return

    prevalence = all_numeric[models[0]]['prevalence']
    if prevalence is None or prevalence == 0:
        return

    lifts = [all_numeric[m]['AUPRC'][0] / prevalence for m in models]
    aurocs = [all_numeric[m]['AUROC'][0] for m in models]

    # sort by lift descending
    order = np.argsort(lifts)[::-1]
    models = [models[i] for i in order]
    lifts = [lifts[i] for i in order]
    aurocs = [aurocs[i] for i in order]

    display = [m.replace('Audio_', 'A+').replace('Video', 'V').replace('Tabular', 'T')
                .replace('Full_Modality', 'Full(A+V+D+T)').replace('DSR', 'D')
                .replace('only', 'Only')
               for m in models]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(models))
    w = 0.55

    # --- AUPRC Lift ---
    bars = axes[0].bar(x, lifts, w, color='#3498db', edgecolor='white', linewidth=0.8)
    axes[0].axhline(1.0, color='black', linewidth=1.2, linestyle='--', label=f'No Skill (prevalence={prevalence:.3f})')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(display, fontsize=11, rotation=20, ha='right')
    axes[0].set_ylabel('AUPRC Lift (×)', fontsize=13)
    axes[0].set_title(f'{LABEL_NAME} — AUPRC Lift over No Skill', fontsize=14)
    axes[0].legend(fontsize=10)
    for i, v in enumerate(lifts):
        axes[0].text(i, v + 0.1, f'{v:.1f}×', ha='center', fontsize=11, fontweight='bold')

    # --- AUROC (absolute, but with 0.5 baseline) ---
    bars2 = axes[1].bar(x, aurocs, w, color='#9b59b6', edgecolor='white', linewidth=0.8)
    axes[1].axhline(0.5, color='black', linewidth=1.2, linestyle='--', label='Random (0.5)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(display, fontsize=11, rotation=20, ha='right')
    axes[1].set_ylabel('AUROC', fontsize=13)
    axes[1].set_title(f'{LABEL_NAME} — AUROC (K-fold mean)', fontsize=14)
    axes[1].set_ylim([0.0, 1.0])
    axes[1].legend(fontsize=10)
    for i, v in enumerate(aurocs):
        axes[1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'lift_bar_{LABEL_NAME}.png'), dpi=300)
    plt.show()
    plt.close()


def save_result(metric_df: list, model_list: list, LABEL_NAME="보상조음"):
    result = {}
    for d in metric_df:
        for key, value in d.items():
            if key not in result:
                result[key] = []
            result[key].append(value)

    result_df = pd.DataFrame(result, index=model_list)
    result_df.to_csv(os.path.join('RESULT',
                                  f'result_{LABEL_NAME}.csv'))

def plot_vfs_ablation(logits_path_list, threshold_list, round_digit=4,
                      target_recall=0.70, save_dir="RESULT", show_std_band=True):
    """
    VFS K ablation: Full_Modality (K=8), Full_Modality_K16, Full_Modality_K32 비교.
    ROC/PRC curve + metric bar chart를 label별로 생성.
    """
    dsr_keys = {'Full_Modality': 'K=8', 'Full_Modality_K16': 'K=16', 'Full_Modality_K32': 'K=32'}

    # DSR ablation에 해당하는 path만 필터
    filtered_paths = []
    filtered_thresholds = []
    for p, t in zip(logits_path_list, threshold_list):
        model_name = p.split('/')[1]
        if model_name in dsr_keys:
            filtered_paths.append(p)
            filtered_thresholds.append(t)

    if not filtered_paths:
        print("[DSR Ablation] No DSR ablation results found. Skipping.")
        return

    ablation_dir = os.path.join(save_dir, 'vfs_ablation')
    os.makedirs(ablation_dir, exist_ok=True)

    mean_fpr = np.linspace(0, 1, 200)
    mean_recall_grid = np.linspace(0, 1, 200)

    for LABEL_NAME in ['보상조음', '과다비성']:
        # ---- ROC / PRC subplot ----
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        metric_summary = {}
        last_no_skill = 0.0

        colors = {'Full_Modality': '#e74c3c', 'Full_Modality_K16': '#3498db', 'Full_Modality_K32': '#2ecc71'}

        for logits_path, threshold in zip(filtered_paths, filtered_thresholds):
            label_name = os.path.basename(logits_path)[:-4].split("_")[-1]
            if label_name != LABEL_NAME:
                continue
            model_name = logits_path.split('/')[1]
            display_name = dsr_keys.get(model_name, model_name)
            color = colors.get(model_name, 'gray')

            parent_dir = os.path.dirname(logits_path)
            fold_dirs = sorted(glob.glob(os.path.join(parent_dir, 'fold_*')))

            tpr_list, prec_list = [], []
            fold_metrics = {m: [] for m in
                            ['AUROC', 'AUPRC', 'ACCURACY', 'RECALL', 'PRECISION', 'F1-Score', 'BRIER']}

            for fold_dir in fold_dirs:
                fold_path = os.path.join(fold_dir, f'inference_results_{label_name}.csv')
                if not os.path.exists(fold_path):
                    continue
                data = pd.read_csv(fold_path)
                y_true = data.label.values
                y_prob = data.predicted_probas.values
                if len(np.unique(y_true)) < 2:
                    continue

                fpr_i, tpr_i, _ = roc_curve(y_true, y_prob)
                interp_tpr = np.interp(mean_fpr, fpr_i, tpr_i)
                interp_tpr[0] = 0.0
                tpr_list.append(interp_tpr)

                prec_i, rec_i, _ = precision_recall_curve(y_true, y_prob)
                interp_prec = np.interp(mean_recall_grid, rec_i[::-1], prec_i[::-1])
                prec_list.append(interp_prec)

                fold_t = _find_threshold_at_recall(y_true, y_prob, target_recall=target_recall)
                y_pred = (y_prob >= fold_t).astype(int)

                fold_metrics['AUROC'].append(roc_auc_score(y_true, y_prob))
                fold_metrics['AUPRC'].append(average_precision_score(y_true, y_prob))
                fold_metrics['ACCURACY'].append(accuracy_score(y_true, y_pred))
                fold_metrics['RECALL'].append(recall_score(y_true, y_pred, zero_division=0))
                fold_metrics['PRECISION'].append(precision_score(y_true, y_pred, zero_division=0))
                fold_metrics['F1-Score'].append(f1_score(y_true, y_pred, zero_division=0))
                fold_metrics['BRIER'].append(brier_score_loss(y_true, y_prob))

                last_no_skill = np.sum(y_true == 1) / len(y_true)

            if not tpr_list:
                continue

            metric_summary[display_name] = {
                m: (np.mean(fold_metrics[m]), np.std(fold_metrics[m]))
                for m in fold_metrics
            }

            mt = np.mean(tpr_list, axis=0)
            st = np.std(tpr_list, axis=0)
            axes[0].plot(mean_fpr, mt, linewidth=2.0, label=display_name, color=color)
            if show_std_band:
                axes[0].fill_between(mean_fpr, mt - st, mt + st, alpha=0.15, color=color)

            mp = np.mean(prec_list, axis=0)
            sp = np.std(prec_list, axis=0)
            axes[1].plot(mean_recall_grid, mp, linewidth=2.0, label=display_name, color=color)
            if show_std_band:
                axes[1].fill_between(mean_recall_grid, mp - sp, mp + sp, alpha=0.15, color=color)

        std_tag = ' (K-fold mean ± std)' if show_std_band else ' (K-fold mean)'
        label_en = 'Hypernasality' if LABEL_NAME == '과다비성' else 'Compensatory Articulation'

        axes[0].plot([0, 1], [0, 1], linestyle='--', linewidth=1.5, color='black', label='No Skill')
        axes[0].set_title(f'{label_en} — ROC (VFS K Ablation){std_tag}', fontsize=14)
        axes[0].set_xlabel('False Positive Rate', fontsize=13)
        axes[0].set_ylabel('True Positive Rate', fontsize=13)
        axes[0].set_xlim([-0.01, 1]); axes[0].set_ylim([0, 1.001])
        axes[0].legend(fontsize=10)

        axes[1].axhline(last_no_skill, linestyle='--', linewidth=1.5, color='black',
                        label=f'No Skill ({last_no_skill:.3f})')
        axes[1].set_title(f'{label_en} — PRC (VFS K Ablation){std_tag}', fontsize=14)
        axes[1].set_xlabel('Recall', fontsize=13)
        axes[1].set_ylabel('Precision', fontsize=13)
        axes[1].set_xlim([-0.01, 1]); axes[1].set_ylim([0, 1.01])
        axes[1].legend(fontsize=10)

        plt.tight_layout()
        std_suffix = '' if show_std_band else '_no_std'
        plt.savefig(os.path.join(ablation_dir, f'vfs_ablation_roc_prc_{label_en}{std_suffix}.png'), dpi=300)
        plt.close()

        # ---- 개별 파일 저장 (ROC / PRC 각각) ----
        # ablation에서 수집된 curve data를 재활용
        ablation_curves = []
        for lp, _ in zip(filtered_paths, filtered_thresholds):
            ln = os.path.basename(lp)[:-4].split("_")[-1]
            if ln != LABEL_NAME:
                continue
            mn = lp.split('/')[1]
            dn = dsr_keys.get(mn, mn)
            cl = colors.get(mn, 'gray')
            pd_dir = os.path.dirname(lp)
            fd = sorted(glob.glob(os.path.join(pd_dir, 'fold_*')))
            tl, pl = [], []
            for fd_i in fd:
                fp = os.path.join(fd_i, f'inference_results_{ln}.csv')
                if not os.path.exists(fp):
                    continue
                d = pd.read_csv(fp)
                yt, yp = d.label.values, d.predicted_probas.values
                if len(np.unique(yt)) < 2:
                    continue
                fi, ti, _ = roc_curve(yt, yp)
                it = np.interp(mean_fpr, fi, ti); it[0] = 0.0; tl.append(it)
                pi, ri, _ = precision_recall_curve(yt, yp)
                ip = np.interp(mean_recall_grid, ri[::-1], pi[::-1]); pl.append(ip)
            if tl:
                ablation_curves.append({'dn': dn, 'cl': cl, 'tl': tl, 'pl': pl})

        for plot_type in ['roc', 'prc']:
            fig_s, ax_s = plt.subplots(1, 1, figsize=(7, 6))
            for ac in ablation_curves:
                if plot_type == 'roc':
                    mv = np.mean(ac['tl'], axis=0); sv = np.std(ac['tl'], axis=0)
                    ax_s.plot(mean_fpr, mv, linewidth=2.0, label=ac['dn'], color=ac['cl'])
                    if show_std_band:
                        ax_s.fill_between(mean_fpr, mv - sv, mv + sv, alpha=0.15, color=ac['cl'])
                else:
                    mv = np.mean(ac['pl'], axis=0); sv = np.std(ac['pl'], axis=0)
                    ax_s.plot(mean_recall_grid, mv, linewidth=2.0, label=ac['dn'], color=ac['cl'])
                    if show_std_band:
                        ax_s.fill_between(mean_recall_grid, mv - sv, mv + sv, alpha=0.15, color=ac['cl'])
            if plot_type == 'roc':
                ax_s.plot([0, 1], [0, 1], linestyle='--', linewidth=1.5, color='black', label='No Skill')
                ax_s.set_title(f'{label_en} — ROC (VFS K Ablation){std_tag}', fontsize=14)
                ax_s.set_xlabel('False Positive Rate', fontsize=13)
                ax_s.set_ylabel('True Positive Rate', fontsize=13)
                ax_s.set_ylim([0, 1.001]); ax_s.set_xlim([-0.01, 1])
                ax_s.legend(loc='lower right', fontsize=10)
            else:
                ax_s.axhline(last_no_skill, linestyle='--', linewidth=1.5, color='black',
                             label=f'No Skill ({last_no_skill:.3f})')
                ax_s.set_title(f'{label_en} — PRC (VFS K Ablation){std_tag}', fontsize=14)
                ax_s.set_xlabel('Recall', fontsize=13)
                ax_s.set_ylabel('Precision', fontsize=13)
                ax_s.set_ylim([0, 1.01]); ax_s.set_xlim([-0.01, 1])
                ax_s.legend(fontsize=10)
            ax_s.tick_params(labelsize=13)
            plt.tight_layout()
            plt.savefig(os.path.join(ablation_dir, f'vfs_ablation_{plot_type}_{label_en}{std_suffix}.png'), dpi=300)
            plt.close()

        # ---- Metric bar chart ----
        if not metric_summary:
            continue

        k_labels = list(metric_summary.keys())
        all_metrics = ['AUROC', 'AUPRC', 'RECALL', 'PRECISION', 'F1-Score', 'BRIER']
        n_metrics = len(all_metrics)
        fig, axes_bar = plt.subplots(2, 3, figsize=(18, 10))
        axes_bar = axes_bar.flatten()
        x = np.arange(len(k_labels))
        w = 0.5
        bar_colors = ['#e74c3c', '#3498db', '#2ecc71'][:len(k_labels)]

        for mi, metric in enumerate(all_metrics):
            means = [metric_summary[k][metric][0] for k in k_labels]
            stds = [metric_summary[k][metric][1] for k in k_labels]
            axes_bar[mi].bar(x, means, w, yerr=stds, capsize=5,
                             color=bar_colors, edgecolor='white', linewidth=0.8)
            axes_bar[mi].set_xticks(x)
            axes_bar[mi].set_xticklabels(k_labels, fontsize=12)
            axes_bar[mi].set_ylabel(metric, fontsize=13)
            axes_bar[mi].set_title(f'{LABEL_NAME} — {metric}', fontsize=14)
            for i, (m, s) in enumerate(zip(means, stds)):
                axes_bar[mi].text(i, m + s + 0.01, f'{m:.4f}', ha='center', fontsize=10, fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(ablation_dir, f'vfs_ablation_metrics_{LABEL_NAME}.png'), dpi=300)
        plt.close()

        # ---- CSV 저장 (mean ± std) ----
        csv_metrics = ['AUROC', 'AUPRC', 'ACCURACY', 'RECALL', 'PRECISION', 'F1-Score', 'BRIER']
        csv_rows = []
        for k in k_labels:
            row = {'DSR_K': k}
            for m in csv_metrics:
                mean_val, std_val = metric_summary[k][m]
                row[m] = f"{mean_val:.{round_digit}f} ± {std_val:.{round_digit}f}"
            csv_rows.append(row)
        csv_df = pd.DataFrame(csv_rows)
        csv_path = os.path.join(ablation_dir, f'vfs_ablation_result_{LABEL_NAME}.csv')
        csv_df.to_csv(csv_path, index=False, encoding='utf-8-sig')

        print(f"  [DSR Ablation] {LABEL_NAME}:")
        for k in k_labels:
            s = metric_summary[k]
            print(f"    {k}: AUROC={s['AUROC'][0]:.4f}±{s['AUROC'][1]:.4f}  "
                  f"AUPRC={s['AUPRC'][0]:.4f}±{s['AUPRC'][1]:.4f}  "
                  f"F1={s['F1-Score'][0]:.4f}±{s['F1-Score'][1]:.4f}")
        print(f"    → CSV saved: {csv_path}")

    print(f"\n  DSR ablation plots saved to: {ablation_dir}/")


def parse_args():
    parser = argparse.ArgumentParser(description='CP-Speech #2 Plotting')

    parser.add_argument('--sep_n', type=int, default=256, help='Thresholding 횟수')
    parser.add_argument('--target_metric', type=str, default='RECALL', help='Thresholding 기준 지표(metric)')
    parser.add_argument('--target_value', type=float, default=0.70, help='Thresholding 기준 지표의 값(value)')
    parser.add_argument('--sort_metric', type=str, default='F1-Score', help='최종 Thresholding시 정렬 기준 지표(metric)')

    parser.add_argument('--round_digit', type=int, default=4, help='결과 반올림 자리수')
    parser.add_argument('--vfs_ablation', action='store_true', help='VFS K ablation만 플로팅 (Full_Modality K=8/16/32)')
    parser.add_argument('--no_std_band', action='store_true', help='ROC/PRC에서 std 음영 제거')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    sep_n = args.sep_n
    target_metric = args.target_metric
    target_value = args.target_value
    sort_metric = args.sort_metric
    round_digit = args.round_digit

    make_dirs(os.path.join('.', 'RESULT'))

    all_logits_paths = sorted(glob.glob(os.path.join('RESULT', '*', 'inference_results_*.csv')))

    # 메인 플로팅용: LEGEND_ORDER에 있는 모델만 (K16, K32 등 ablation 전용 제외)
    logits_path_list = [p for p in all_logits_paths if p.split('/')[1] in LEGEND_ORDER]
    # VFS ablation용: 전체 경로 유지
    all_threshold_list = [get_opt_threshold(p, sep_n=sep_n,
                                            target_metric=target_metric,
                                            target_value=target_value,
                                            sort_metric=sort_metric) for p in all_logits_paths]
    threshold_list = [get_opt_threshold(p, sep_n=sep_n,
                                        target_metric=target_metric,
                                        target_value=target_value,
                                        sort_metric=sort_metric) for p in logits_path_list]
    
    show_std_band = not args.no_std_band

    if args.vfs_ablation:
        plot_vfs_ablation(all_logits_paths, all_threshold_list,
                          round_digit=round_digit,
                          target_recall=target_value,
                          save_dir="RESULT",
                          show_std_band=show_std_band)
        return

    for label in ['보상조음', '과다비성']:
        # 기존: ROC/PRC curve
        metric_df, model_list = ROC_PRC_subplot(logits_path_list, threshold_list,
                                                round_digit=round_digit,
                                                LABEL_NAME=label,
                                                target_recall=target_value,
                                                show_std_band=show_std_band)
        save_result(metric_df, model_list, label)

        # 추가: delta bar chart + lift chart
        all_numeric = collect_kfold_numeric(logits_path_list, threshold_list,
                                            LABEL_NAME=label,
                                            target_recall=target_value)
        plot_delta_bar(all_numeric, LABEL_NAME=label, baseline_key="Audio_only", save_dir="RESULT")
        plot_lift_bar(all_numeric, LABEL_NAME=label, save_dir="RESULT")
    


if __name__ == "__main__":
    main()