"""
Video Modality Contribution Analysis
=====================================
1. AttnPool weight → spatial/temporal attention heatmap
2. Case study → Audio_only vs Audio+Video 예측 비교
3. Grad-CAM → class-discriminative spatial maps

Usage:
    CUDA_VISIBLE_DEVICES=1 python ablation.py --fold_idx 0 --epochs 8

Results: ./RESULT/video_ablation/
"""

import os
import glob
import argparse
import subprocess
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import AutoProcessor

from dataset import Mydataset
from model import Baseline
from learning import train, evaluate
from utils import set_SEED, make_dirs, load_jsonl, get_patient_kfold_splits

import warnings
warnings.filterwarnings('ignore')

matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/truetype/Pretendard-Regular.ttf')
fm.fontManager.addfont('/usr/share/fonts/truetype/Pretendard-Bold.ttf')
matplotlib.rcParams['font.family'] = 'Pretendard'

DATA_PATH = os.path.join("_DATA", "DATA_video_clip_npy_audio_clip_tabular.jsonl")
RESULT_ROOT = "RESULT"


# ============================================================
# Utils
# ============================================================

def _load_video_frames(video_path, n_frames=8):
    """ffmpeg로 동영상에서 n_frames개의 프레임을 균일 추출."""
    if not os.path.exists(video_path):
        return None
    try:
        probe = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
             '-show_entries', 'stream=width,height,nb_frames',
             '-of', 'csv=p=0', video_path],
            capture_output=True, text=True, timeout=10)
        parts = probe.stdout.strip().split(',')
        w, h = int(parts[0]), int(parts[1])
        total = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 30

        indices = np.linspace(0, total - 1, n_frames, dtype=int)

        frames = []
        for idx in indices:
            result = subprocess.run(
                ['ffmpeg', '-i', video_path,
                 '-vf', f'select=eq(n\\,{idx})',
                 '-frames:v', '1',
                 '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                 '-v', 'error', '-'],
                capture_output=True, timeout=10)
            if len(result.stdout) == h * w * 3:
                frame = np.frombuffer(result.stdout, dtype=np.uint8).reshape(h, w, 3)
                frames.append(frame)
            else:
                frames.append(None)
        return frames
    except Exception:
        return None


# ============================================================
# 1. Attention Weight Extraction
# ============================================================

def extract_video_attention(model, dataloader, device):
    """
    AttnPool attention weights 추출.
    Returns:
        token_weights: (N, 8, 256) — spatial attention per frame
        frame_weights: (N, 8)      — temporal attention across frames
        logits:        (N, 2)
        targets:       (N, 2)
    """
    model.eval()
    all_tw, all_fw, all_logits, all_targets = [], [], [], []

    for batch in tqdm(dataloader, desc="Extracting attention"):
        ((audio_in, attn_mask), video_emb, dsr_emb, (cate, nume)), target = batch

        tw_buf, fw_buf = [None], [None]

        def _token_hook(m, inp, out, buf=tw_buf):
            w = out.squeeze(-1)                       # (B, F, T)
            buf[0] = torch.softmax(w, dim=2).detach().cpu()

        def _frame_hook(m, inp, out, buf=fw_buf):
            w = out.squeeze(-1)                       # (B, F)
            buf[0] = torch.softmax(w, dim=1).detach().cpu()

        h1 = model.video.token_pool.score.register_forward_hook(_token_hook)
        h2 = model.video.frame_pool.score.register_forward_hook(_frame_hook)

        with torch.no_grad():
            logits = model(
                audio_input_values=audio_in.to(device),
                audio_attention_mask=attn_mask.to(device),
                video_x=video_emb.to(device))

        h1.remove()
        h2.remove()

        all_tw.append(tw_buf[0])
        all_fw.append(fw_buf[0])
        all_logits.append(logits.cpu())
        all_targets.append(target)

    return (torch.cat(all_tw), torch.cat(all_fw),
            torch.cat(all_logits), torch.cat(all_targets))


def plot_frame_attention(frame_weights, targets, save_dir):
    """Positive vs Negative별 temporal attention bar chart."""
    fw = frame_weights.numpy()
    tgt = targets.numpy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for li, lname in enumerate(['보상조음', '과다비성']):
        ax = axes[li]
        pos = tgt[:, li] == 1
        neg = ~pos
        x = np.arange(8)
        w = 0.35

        for mask, label, color, offset in [
            (pos, f'Positive (n={pos.sum()})', '#e74c3c', -w / 2),
            (neg, f'Negative (n={neg.sum()})', '#3498db',  w / 2),
        ]:
            if mask.sum() == 0:
                continue
            m = fw[mask].mean(0)
            s = fw[mask].std(0)
            ax.bar(x + offset, m, w, yerr=s, label=label,
                   color=color, alpha=0.8, capsize=3)

        ax.set_xlabel('Frame Index')
        ax.set_ylabel('Attention Weight')
        ax.set_title(f'{lname} — Frame Attention')
        ax.set_xticks(x)
        ax.legend()

    plt.tight_layout()
    path = os.path.join(save_dir, 'frame_attention_bar.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_spatial_attention_mean(token_weights, targets, save_dir):
    """Positive vs Negative의 평균 spatial attention heatmap (16×16)."""
    tw = token_weights.numpy()
    tgt = targets.numpy()
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    for li, lname in enumerate(['보상조음', '과다비성']):
        pos = tgt[:, li] == 1
        neg = ~pos
        for ci, (mask, cname) in enumerate([(pos, 'Positive'), (neg, 'Negative')]):
            ax = axes[li][ci]
            if mask.sum() == 0:
                ax.set_title(f'{lname} — {cname} (n=0)')
                ax.axis('off')
                continue
            mean_attn = tw[mask].mean(axis=(0, 1)).reshape(16, 16)
            im = ax.imshow(mean_attn, cmap='jet', interpolation='bilinear')
            ax.set_title(f'{lname} — {cname} (n={mask.sum()})')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle('Average Spatial Attention (16×16 ViT patches)', fontsize=14)
    plt.tight_layout()
    path = os.path.join(save_dir, 'spatial_attention_mean.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_spatial_attention_overlay(token_weights, valid_data, save_dir,
                                   sample_indices, tag="attn"):
    """개별 샘플의 spatial attention을 video frame 위에 overlay."""
    subdir = os.path.join(save_dir, 'spatial_attention')
    make_dirs(subdir)
    tw = token_weights.numpy()

    for idx in sample_indices:
        sample = valid_data[idx]
        attn = tw[idx]                          # (8, 256)
        name = sample.get('name', 'unk')
        word = sample.get('word', 'unk')
        video_path = sample.get('video_clip_file', '')

        frames = _load_video_frames(video_path, n_frames=8)

        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle(
            f'{name} — "{word}" '
            f'(보상조음={sample["보상조음"]}, 과다비성={sample["과다비성"]})',
            fontsize=13)

        for fi in range(8):
            ax = axes[fi // 4][fi % 4]
            hm = attn[fi].reshape(16, 16)
            hm_norm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)

            if frames and frames[fi] is not None:
                frame = frames[fi]
                ax.imshow(frame, aspect='auto')
                hm_resized = np.array(
                    Image.fromarray((hm_norm * 255).astype(np.uint8)).resize(
                        (frame.shape[1], frame.shape[0]), Image.BILINEAR)
                ) / 255.0
                ax.imshow(hm_resized, alpha=0.5, cmap='jet', aspect='auto')
            else:
                ax.imshow(hm_norm, cmap='jet')

            ax.set_title(f'Frame {fi}', fontsize=10)
            ax.axis('off')

        plt.tight_layout()
        fname = f'{tag}_{idx}_{name}_{word}.png'.replace(' ', '_')
        plt.savefig(os.path.join(subdir, fname), dpi=200, bbox_inches='tight')
        plt.close()

    print(f"  Saved {len(sample_indices)} spatial overlays → {subdir}/")


# ============================================================
# 2. Case Study — Video가 예측을 교정한 샘플
# ============================================================

def build_case_study(folds, save_dir,
                     ao_dir=os.path.join(RESULT_ROOT, 'Audio_Only'),
                     av_dir=os.path.join(RESULT_ROOT, 'Audio_Video')):
    """
    Audio_only와 Audio_Video의 fold별 예측 비교.
    improvement > 0 → Video가 올바른 방향으로 확률을 이동시킨 샘플.
    """
    rows = []
    for label_name in ['보상조음', '과다비성']:
        for fi, (_, vdata) in enumerate(folds):
            ao_p = os.path.join(ao_dir, f'fold_{fi}',
                                f'inference_results_{label_name}.csv')
            av_p = os.path.join(av_dir, f'fold_{fi}',
                                f'inference_results_{label_name}.csv')
            if not os.path.exists(ao_p) or not os.path.exists(av_p):
                continue

            ao = pd.read_csv(ao_p)
            av = pd.read_csv(av_p)

            for i in range(len(ao)):
                label = int(ao.label.iloc[i])
                ao_prob = float(ao.predicted_probas.iloc[i])
                av_prob = float(av.predicted_probas.iloc[i])
                direction = 1 if label == 1 else -1
                improvement = (av_prob - ao_prob) * direction

                rows.append({
                    'label_name': label_name,
                    'fold': fi,
                    'idx_in_fold': i,
                    'name': vdata[i]['name'],
                    'word': vdata[i]['word'],
                    'true_label': label,
                    'audio_only_prob': round(ao_prob, 4),
                    'audio_video_prob': round(av_prob, 4),
                    'prob_delta': round(av_prob - ao_prob, 4),
                    'improvement': round(improvement, 4),
                    'video_clip_file': vdata[i].get('video_clip_file', ''),
                })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(save_dir, 'case_study_all.csv'),
              index=False, encoding='utf-8-sig')

    print("\n  === Case Study Summary ===")
    for ln in ['보상조음', '과다비성']:
        sub = df[df.label_name == ln]
        n_imp = (sub.improvement > 0).sum()
        n_deg = (sub.improvement < 0).sum()
        print(f"  {ln}: improved {n_imp}/{len(sub)} "
              f"({n_imp/len(sub)*100:.1f}%), "
              f"degraded {n_deg}/{len(sub)} "
              f"({n_deg/len(sub)*100:.1f}%), "
              f"mean Δ={sub.improvement.mean():+.4f}")
    return df


LABEL_EN = {'보상조음': 'Compensatory Articulation', '과다비성': 'Hypernasality'}


def plot_case_study(case_df, save_dir):
    """Probability shift scatter + improvement histogram."""
    # ---- Scatter: Audio_only prob vs Audio_Video prob (합본) ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for li, lname in enumerate(['보상조음', '과다비성']):
        ax = axes[li]
        label_en = LABEL_EN[lname]
        sub = case_df[case_df.label_name == lname]
        pos = sub[sub.true_label == 1]
        neg = sub[sub.true_label == 0]

        ax.scatter(neg.audio_only_prob, neg.audio_video_prob,
                   alpha=0.3, s=15, c='#3498db', label=f'Negative (n={len(neg)})')
        ax.scatter(pos.audio_only_prob, pos.audio_video_prob,
                   alpha=0.7, s=40, c='#e74c3c', label=f'Positive (n={len(pos)})')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='No change')
        ax.set_xlabel('Audio Only Predicted Probability')
        ax.set_ylabel('Audio + Video Predicted Probability')
        ax.set_title(f'{label_en} — Probability Shift')
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=9)
        ax.set_aspect('equal')

    plt.tight_layout()
    path = os.path.join(save_dir, 'case_study_probability_shift.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # ---- Scatter: 개별 파일 저장 ----
    for lname in ['보상조음', '과다비성']:
        label_en = LABEL_EN[lname]
        sub = case_df[case_df.label_name == lname]
        pos = sub[sub.true_label == 1]
        neg = sub[sub.true_label == 0]

        fig_s, ax_s = plt.subplots(1, 1, figsize=(7, 6))
        ax_s.scatter(neg.audio_only_prob, neg.audio_video_prob,
                     alpha=0.3, s=15, c='#3498db', label=f'Negative (n={len(neg)})')
        ax_s.scatter(pos.audio_only_prob, pos.audio_video_prob,
                     alpha=0.7, s=40, c='#e74c3c', label=f'Positive (n={len(pos)})')
        ax_s.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='No change')
        ax_s.set_xlabel('Audio Only Predicted Probability', fontsize=13)
        ax_s.set_ylabel('Audio + Video Predicted Probability', fontsize=13)
        ax_s.set_title(f'{label_en} — Probability Shift', fontsize=14)
        ax_s.set_xlim(-0.02, 1.02)
        ax_s.set_ylim(-0.02, 1.02)
        ax_s.legend(fontsize=10)
        ax_s.set_aspect('equal')
        plt.tight_layout()
        path = os.path.join(save_dir, f'case_study_probability_shift_{label_en}.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {path}")

    # ---- Histogram: improvement distribution (합본) ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for li, lname in enumerate(['보상조음', '과다비성']):
        ax = axes[li]
        label_en = LABEL_EN[lname]
        sub = case_df[case_df.label_name == lname]
        pos = sub[sub.true_label == 1]
        neg = sub[sub.true_label == 0]

        bins = np.linspace(-0.5, 0.5, 41)
        if len(pos) > 0:
            ax.hist(pos.improvement, bins=bins, alpha=0.7,
                    color='#e74c3c', label=f'Positive (n={len(pos)})')
        if len(neg) > 0:
            ax.hist(neg.improvement, bins=bins, alpha=0.5,
                    color='#3498db', label=f'Negative (n={len(neg)})')
        ax.axvline(0, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('Improvement (>0 = Video helped)')
        ax.set_ylabel('Count')
        ax.set_title(f'{label_en} — Improvement Distribution')
        ax.legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(save_dir, 'case_study_improvement_dist.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # ---- Histogram: 개별 파일 저장 ----
    for lname in ['보상조음', '과다비성']:
        label_en = LABEL_EN[lname]
        sub = case_df[case_df.label_name == lname]
        pos = sub[sub.true_label == 1]
        neg = sub[sub.true_label == 0]

        fig_s, ax_s = plt.subplots(1, 1, figsize=(7, 5))
        bins = np.linspace(-0.5, 0.5, 41)
        if len(pos) > 0:
            ax_s.hist(pos.improvement, bins=bins, alpha=0.7,
                      color='#e74c3c', label=f'Positive (n={len(pos)})')
        if len(neg) > 0:
            ax_s.hist(neg.improvement, bins=bins, alpha=0.5,
                      color='#3498db', label=f'Negative (n={len(neg)})')
        ax_s.axvline(0, color='k', linestyle='--', alpha=0.5)
        ax_s.set_xlabel('Improvement (>0 = Video helped)', fontsize=13)
        ax_s.set_ylabel('Count', fontsize=13)
        ax_s.set_title(f'{label_en} — Improvement Distribution', fontsize=14)
        ax_s.legend(fontsize=10)
        plt.tight_layout()
        path = os.path.join(save_dir, f'case_study_improvement_dist_{label_en}.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {path}")


# ============================================================
# 3. Grad-CAM
# ============================================================

def compute_grad_cam(model, dataloader, device, target_class=0):
    """
    Grad-CAM on VideoModel.proj output.
    Returns: cam (N, 8, 256), targets (N, 2)
    """
    model.eval()
    all_cams, all_targets = [], []

    for batch in tqdm(dataloader, desc=f"Grad-CAM (class={target_class})"):
        ((audio_in, attn_mask), video_emb, dsr_emb, (cate, nume)), target = batch

        act_buf = [None]

        def _fwd_hook(m, inp, out, buf=act_buf):
            out.retain_grad()
            buf[0] = out

        handle = model.video.proj.register_forward_hook(_fwd_hook)

        logits = model(
            audio_input_values=audio_in.to(device),
            audio_attention_mask=attn_mask.to(device),
            video_x=video_emb.to(device))

        model.zero_grad()
        logits[:, target_class].sum().backward()
        handle.remove()

        act = act_buf[0]               # (B, F, T, d_model)
        grad = act.grad                # (B, F, T, d_model)

        # Grad-CAM: GAP of gradients → channel weights → weighted activation sum
        weights = grad.mean(dim=(1, 2))                         # (B, d_model)
        cam = torch.relu(
            (act * weights[:, None, None, :]).sum(dim=-1))      # (B, F, T)

        # Normalize per sample
        flat = cam.view(cam.size(0), -1)
        cmin = flat.min(dim=1, keepdim=True)[0].unsqueeze(-1)
        cmax = flat.max(dim=1, keepdim=True)[0].unsqueeze(-1)
        cam = (cam - cmin) / (cmax - cmin + 1e-8)

        all_cams.append(cam.detach().cpu())
        all_targets.append(target)

    return torch.cat(all_cams), torch.cat(all_targets)


def plot_grad_cam_mean(cam_0, cam_1, targets, save_dir):
    """Aggregated Grad-CAM: Positive vs Negative 평균 (16×16)."""
    tgt = targets.numpy()
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    for row, (cam, cname) in enumerate(
            [(cam_0.numpy(), '보상조음'), (cam_1.numpy(), '과다비성')]):
        label_idx = 0 if cname == '보상조음' else 1
        pos = tgt[:, label_idx] == 1
        neg = ~pos
        for col, (mask, mname) in enumerate(
                [(pos, 'Positive'), (neg, 'Negative')]):
            ax = axes[row][col]
            if mask.sum() == 0:
                ax.set_title(f'{cname} — {mname} (n=0)')
                ax.axis('off')
                continue
            mean_cam = cam[mask].mean(axis=(0, 1)).reshape(16, 16)
            im = ax.imshow(mean_cam, cmap='hot', interpolation='bilinear')
            ax.set_title(f'{cname} — {mname} (n={mask.sum()})')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle('Average Grad-CAM (16×16)', fontsize=14)
    plt.tight_layout()
    path = os.path.join(save_dir, 'grad_cam_mean.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_grad_cam_samples(cam_0, cam_1, targets, valid_data, save_dir,
                          sample_indices):
    """Per-sample Grad-CAM: video frame + 보상조음 CAM + 과다비성 CAM."""
    subdir = os.path.join(save_dir, 'grad_cam')
    make_dirs(subdir)
    c0 = cam_0.numpy()
    c1 = cam_1.numpy()

    for idx in sample_indices:
        sample = valid_data[idx]
        name = sample.get('name', 'unk')
        word = sample.get('word', 'unk')
        video_path = sample.get('video_clip_file', '')
        frames = _load_video_frames(video_path, n_frames=8)

        fig, axes = plt.subplots(3, 8, figsize=(24, 9))
        fig.suptitle(
            f'{name} — "{word}" '
            f'(보상조음={sample["보상조음"]}, 과다비성={sample["과다비성"]})',
            fontsize=14)

        for fi in range(8):
            # Row 0: Video frame
            if frames and frames[fi] is not None:
                axes[0][fi].imshow(frames[fi], aspect='auto')
            axes[0][fi].set_title(f'Frame {fi}', fontsize=9)
            axes[0][fi].axis('off')

            # Row 1: 보상조음 Grad-CAM
            axes[1][fi].imshow(
                c0[idx, fi].reshape(16, 16),
                cmap='hot', vmin=0, vmax=1, interpolation='bilinear')
            axes[1][fi].axis('off')

            # Row 2: 과다비성 Grad-CAM
            axes[2][fi].imshow(
                c1[idx, fi].reshape(16, 16),
                cmap='hot', vmin=0, vmax=1, interpolation='bilinear')
            axes[2][fi].axis('off')

        axes[0][0].set_ylabel('Frame', fontsize=11, rotation=90, labelpad=15)
        axes[1][0].set_ylabel('보상조음', fontsize=11, rotation=90, labelpad=15)
        axes[2][0].set_ylabel('과다비성', fontsize=11, rotation=90, labelpad=15)

        plt.tight_layout()
        fname = f'gradcam_{idx}_{name}_{word}.png'.replace(' ', '_')
        plt.savefig(os.path.join(subdir, fname), dpi=200, bbox_inches='tight')
        plt.close()

    print(f"  Saved {len(sample_indices)} Grad-CAM plots → {subdir}/")


# ============================================================
# 5. Matched Pair Comparison (교수님 발표용)
# ============================================================

def _overlay_cam_on_frame(frame, cam_16x16, alpha=0.45):
    """16×16 Grad-CAM을 프레임 위에 overlay하여 blended image 반환."""
    from PIL import Image as PILImage
    h, w = frame.shape[:2]
    cam_norm = (cam_16x16 - cam_16x16.min()) / (cam_16x16.max() - cam_16x16.min() + 1e-8)
    cam_resized = np.array(
        PILImage.fromarray((cam_norm * 255).astype(np.uint8)).resize(
            (w, h), PILImage.BILINEAR)) / 255.0
    # jet colormap
    cmap = plt.cm.jet
    cam_color = cmap(cam_resized)[..., :3]  # (H, W, 3)
    blended = (1 - alpha) * (frame / 255.0) + alpha * cam_color
    blended = np.clip(blended, 0, 1)
    return blended, cam_resized


def find_matched_pairs(valid_data, targets):
    """
    같은 단어를 발화한 양성 vs 음성 샘플 쌍을 찾는다.
    Returns: list of (pos_idx, neg_idx, word) tuples
    """
    tgt = targets.numpy()
    # 어느 label이든 양성이면 positive로 간주
    is_pos = (tgt[:, 0] == 1) | (tgt[:, 1] == 1)

    word_to_pos = {}
    word_to_neg = {}
    for i, sample in enumerate(valid_data):
        word = sample.get('word', 'unk')
        if is_pos[i]:
            word_to_pos.setdefault(word, []).append(i)
        else:
            word_to_neg.setdefault(word, []).append(i)

    pairs = []
    for word in word_to_pos:
        if word in word_to_neg:
            # 각 단어에서 하나씩 선택
            pairs.append((word_to_pos[word][0], word_to_neg[word][0], word))

    return pairs


def plot_matched_comparison(cam_0, cam_1, targets, valid_data,
                            save_dir, max_pairs=6):
    """
    Matched Pair: 같은 단어를 말하는 양성 vs 음성 환자의
    Grad-CAM overlay를 한 장에 나란히 배치.

    Layout per pair (1 row):
      [Positive frame+CAM] [Negative frame+CAM]

    보상조음과 과다비성 Grad-CAM 중 해당 label이 양성인 쪽을 사용.
    """
    subdir = os.path.join(save_dir, 'matched_pairs')
    make_dirs(subdir)

    pairs = find_matched_pairs(valid_data, targets)
    if not pairs:
        print("  No matched pairs found (양성/음성이 같은 단어를 발화한 쌍 없음)")
        return

    pairs = pairs[:max_pairs]
    tgt = targets.numpy()
    c0, c1 = cam_0.numpy(), cam_1.numpy()

    n_pairs = len(pairs)
    fig, axes = plt.subplots(n_pairs, 4, figsize=(20, 5 * n_pairs))
    if n_pairs == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle('Matched Pair Comparison — 같은 단어, 다른 환자\n'
                 '(Grad-CAM overlay: 모델이 주목하는 영역)',
                 fontsize=16, y=1.01)

    for row, (pos_idx, neg_idx, word) in enumerate(pairs):
        pos_sample = valid_data[pos_idx]
        neg_sample = valid_data[neg_idx]

        # 각 샘플에서 attention이 가장 높은 프레임(frame_weights 없으므로 CAM 합 기준)
        for col_offset, (idx, sample, label_tag) in enumerate([
            (pos_idx, pos_sample, 'Positive'),
            (neg_idx, neg_sample, 'Negative'),
        ]):
            # 해당 샘플이 양성인 label의 CAM 사용 (없으면 보상조음 기본)
            if tgt[idx, 0] == 1:
                cam = c0[idx]  # (F, T)
                cam_label = '보상조음'
            elif tgt[idx, 1] == 1:
                cam = c1[idx]
                cam_label = '과다비성'
            else:
                cam = c0[idx]  # negative → 보상조음 CAM 사용
                cam_label = '보상조음'

            # 프레임별 CAM 강도 → 가장 높은 프레임 선택
            frame_scores = cam.sum(axis=1)  # (F,)
            best_frame = int(np.argmax(frame_scores))

            cam_16 = cam[best_frame].reshape(16, 16)

            frames = _load_video_frames(sample.get('video_clip_file', ''), n_frames=8)

            # Col 1: 원본 프레임
            ax_orig = axes[row, col_offset * 2]
            # Col 2: Grad-CAM overlay
            ax_cam = axes[row, col_offset * 2 + 1]

            name = sample.get('name', 'unk')

            if frames and frames[best_frame] is not None:
                frame = frames[best_frame]
                ax_orig.imshow(frame)
                blended, _ = _overlay_cam_on_frame(frame, cam_16)
                ax_cam.imshow(blended)
            else:
                ax_orig.text(0.5, 0.5, 'No frame', ha='center', va='center',
                            transform=ax_orig.transAxes, fontsize=12)
                ax_cam.imshow(cam_16, cmap='jet', interpolation='bilinear')

            ax_orig.set_title(f'{label_tag}: {name}\n"{word}" (frame {best_frame})',
                             fontsize=12)
            ax_cam.set_title(f'Grad-CAM ({cam_label})', fontsize=12)
            ax_orig.axis('off')
            ax_cam.axis('off')

    plt.tight_layout()
    path = os.path.join(subdir, 'matched_pairs_comparison.png')
    plt.savefig(path, dpi=250, bbox_inches='tight')
    plt.close()
    print(f"  Saved matched pair comparison: {path}")

    # 개별 pair도 고해상도로 저장
    for pos_idx, neg_idx, word in pairs:
        pos_sample = valid_data[pos_idx]
        neg_sample = valid_data[neg_idx]

        fig, axes_i = plt.subplots(1, 4, figsize=(20, 5))

        for col_offset, (idx, sample, label_tag) in enumerate([
            (pos_idx, pos_sample, 'Positive'),
            (neg_idx, neg_sample, 'Negative'),
        ]):
            if tgt[idx, 0] == 1:
                cam = c0[idx]; cam_label = '보상조음'
            elif tgt[idx, 1] == 1:
                cam = c1[idx]; cam_label = '과다비성'
            else:
                cam = c0[idx]; cam_label = '보상조음'

            frame_scores = cam.sum(axis=1)
            best_frame = int(np.argmax(frame_scores))
            cam_16 = cam[best_frame].reshape(16, 16)
            frames = _load_video_frames(sample.get('video_clip_file', ''), n_frames=8)
            name = sample.get('name', 'unk')

            ax_orig = axes_i[col_offset * 2]
            ax_cam = axes_i[col_offset * 2 + 1]

            if frames and frames[best_frame] is not None:
                frame = frames[best_frame]
                ax_orig.imshow(frame)
                blended, _ = _overlay_cam_on_frame(frame, cam_16)
                ax_cam.imshow(blended)
            else:
                ax_orig.text(0.5, 0.5, 'No frame', ha='center', va='center',
                            transform=ax_orig.transAxes)
                ax_cam.imshow(cam_16, cmap='jet', interpolation='bilinear')

            ax_orig.set_title(f'{label_tag}: {name}', fontsize=13)
            ax_cam.set_title(f'Grad-CAM ({cam_label})', fontsize=13)
            ax_orig.axis('off')
            ax_cam.axis('off')

        fig.suptitle(f'단어: "{word}"', fontsize=15)
        plt.tight_layout()
        pos_name = pos_sample.get('name', 'unk')
        neg_name = neg_sample.get('name', 'unk')
        fname = f'pair_{word}_{pos_name}_vs_{neg_name}.png'.replace(' ', '_')
        plt.savefig(os.path.join(subdir, fname), dpi=250, bbox_inches='tight')
        plt.close()

    print(f"  Saved {len(pairs)} individual pair plots → {subdir}/")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Video Modality Contribution Analysis')
    parser.add_argument('--fold_idx', type=int, default=0,
                        help='분석에 사용할 fold index')
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--n_vis_samples', type=int, default=8,
                        help='Per-sample 시각화 샘플 수')
    parser.add_argument('--SEED', type=int, default=17)
    args = parser.parse_args()

    SAVE_DIR = os.path.join(RESULT_ROOT, 'video_ablation')
    make_dirs(SAVE_DIR)
    set_SEED(args.SEED)

    # ---- Load data & folds ----
    data = load_jsonl(DATA_PATH)
    folds = get_patient_kfold_splits(data, n_splits=5, SEED=args.SEED)
    train_data, valid_data = folds[args.fold_idx]

    print(f"Fold {args.fold_idx}: "
          f"train {len(train_data)} samples, valid {len(valid_data)} samples")

    # ==========================================================
    # Step 1: Case Study (기존 결과 활용, 학습 불필요)
    # ==========================================================
    print("\n" + "=" * 60)
    print("Step 1: Case Study — Audio_only vs Audio_Video")
    print("=" * 60)

    case_df = build_case_study(folds, SAVE_DIR)
    plot_case_study(case_df, SAVE_DIR)

    # ==========================================================
    # Step 2: Audio+Video 모델 학습 (단일 fold)
    # ==========================================================
    print("\n" + "=" * 60)
    print(f"Step 2: Training Audio+Video model (fold {args.fold_idx})")
    print("=" * 60)

    audio_model_name = "Kkonjeong/wav2vec2-base-korean"
    processor = AutoProcessor.from_pretrained(audio_model_name)
    processor.feature_extractor.return_attention_mask = True

    train_dataset = Mydataset(train_data, processor, DSR_K=8)
    valid_dataset = Mydataset(valid_data, processor, DSR_K=8)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=4)
    eval_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Baseline(
        active_modality={
            'audio': True, 'video': True,
            'dsr': False, 'tabular': False},
        audio_model_name=audio_model_name,
        unfreeze_last_n_layers=2,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-3)
    pos_weight = torch.tensor([20.28, 14.67]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # train/evaluate에 필요한 최소 args
    train_args = argparse.Namespace(
        label_frequency_0=0.0323,
        label_frequency_1=0.0638,
    )

    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")
        train(train_args, device, model, optimizer,
              criterion, train_loader, accum_step=1)

    print("\nTraining complete.")

    # ==========================================================
    # Step 3: Attention Weight Extraction & Visualization
    # ==========================================================
    print("\n" + "=" * 60)
    print("Step 3: Extracting Attention Weights")
    print("=" * 60)

    token_weights, frame_weights, logits, targets = \
        extract_video_attention(model, eval_loader, device)

    plot_frame_attention(frame_weights, targets, SAVE_DIR)
    plot_spatial_attention_mean(token_weights, targets, SAVE_DIR)

    # 시각화할 샘플 선택: positive 우선
    tgt_np = targets.numpy()
    pos_idx = np.where((tgt_np[:, 0] == 1) | (tgt_np[:, 1] == 1))[0]
    neg_idx = np.where((tgt_np[:, 0] == 0) & (tgt_np[:, 1] == 0))[0]
    vis_indices = list(pos_idx[:min(args.n_vis_samples, len(pos_idx))])
    remaining = args.n_vis_samples - len(vis_indices)
    if remaining > 0 and len(neg_idx) > 0:
        vis_indices += list(neg_idx[
            np.linspace(0, len(neg_idx) - 1, remaining, dtype=int)])

    plot_spatial_attention_overlay(
        token_weights, valid_data, SAVE_DIR, vis_indices)

    # ==========================================================
    # Step 4: Grad-CAM
    # ==========================================================
    print("\n" + "=" * 60)
    print("Step 4: Grad-CAM")
    print("=" * 60)

    cam_0, _ = compute_grad_cam(model, eval_loader, device, target_class=0)
    cam_1, tgt2 = compute_grad_cam(model, eval_loader, device, target_class=1)

    plot_grad_cam_mean(cam_0, cam_1, tgt2, SAVE_DIR)
    plot_grad_cam_samples(
        cam_0, cam_1, tgt2, valid_data, SAVE_DIR, vis_indices)

    # ==========================================================
    # Step 5: Matched Pair Comparison (발표용)
    # ==========================================================
    print("\n" + "=" * 60)
    print("Step 5: Matched Pair Comparison")
    print("=" * 60)

    plot_matched_comparison(
        cam_0, cam_1, tgt2, valid_data, SAVE_DIR, max_pairs=6)

    # ==========================================================
    # Summary
    # ==========================================================
    print("\n" + "=" * 60)
    print(f"All results saved to: {SAVE_DIR}/")
    print("=" * 60)
    print("  case_study_all.csv")
    print("  case_study_probability_shift.png")
    print("  case_study_improvement_dist.png")
    print("  frame_attention_bar.png")
    print("  spatial_attention_mean.png")
    print("  spatial_attention/  (per-sample overlays)")
    print("  grad_cam_mean.png")
    print("  grad_cam/           (per-sample Grad-CAM)")
    print("  matched_pairs/      (발표용 matched pair comparison)")


if __name__ == "__main__":
    main()

"""
CUDA_VISIBLE_DEVICES=1 python ablation.py \
--fold_idx 0 \
--epochs 8 \
--batch_size 32 \
--learning_rate 1e-5 \
--n_vis_samples 8
"""
