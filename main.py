import os
import math
import time
import argparse
from platform import processor
import numpy as np
import pandas as pd

from tqdm import tqdm
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel, AutoProcessor

from dataset import *
from model import *
from learning import *
from utils import *

import warnings
warnings.filterwarnings(action='ignore')


def main(args):
    ### Args Settings
    SEED = args.SEED; set_SEED(SEED)

    SAVE_ROOT = args.save_root
    SAVE_DIR = os.path.join(SAVE_ROOT, args.save_dir)
    make_dirs(SAVE_DIR)

    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.learning_rate
    ACCUM_STEP = args.accum_step
    DSR_K = args.dsr_k

    label_frequency_0 = args.label_frequency_0
    label_frequency_1 = args.label_frequency_1
    
    is_audio_active = args.is_audio_active
    is_video_active = args.is_video_active
    is_dsr_active = args.is_dsr_active
    is_tabular_active = args.is_tabular_active

    active_modality = {
        'audio': is_audio_active,
        'video': is_video_active,
        'dsr': is_dsr_active,
        'tabular': is_tabular_active,
    }

    audio_model_name = args.audio_model_name
    audio_d_out = args.audio_d_out
    audio_dropout = args.audio_dropout
    freeze_feature_extractor = args.freeze_feature_extractor
    unfreeze_last_n_layers = args.unfreeze_last_n_layers

    video_input_dim = args.video_input_dim
    video_d_model = args.video_d_model
    video_d_out = args.video_d_out
    video_dropout = args.video_dropout

    dsr_input_dim = args.dsr_input_dim
    dsr_d_model = args.dsr_d_model
    dsr_d_small = args.dsr_d_small
    dsr_dropout = args.dsr_dropout
    dsr_alpha = args.dsr_alpha

    nume_input_dim = args.nume_input_dim
    nume_hidden_dim = args.nume_hidden_dim
    nume_output_dim = args.nume_output_dim
    cate_input_dims = args.cate_input_dims
    cate_emb_dims = args.cate_emb_dims
    tab_final_output_dim = args.tab_final_output_dim
    tab_dropout = args.tab_dropout

    processor = AutoProcessor.from_pretrained(audio_model_name)
    processor.feature_extractor.return_attention_mask = True  # ✅ 강제 ON

    ### Dataset Settings
    data_path = os.path.join("_DATA", "DATA_video_clip_npy_audio_clip_tabular.jsonl")
    data = load_jsonl(data_path)
    if args.cv_scheme == 'lopo':
        folds = get_lopo_splits(data)
    elif args.cv_scheme == 'mccv':
        folds = get_montecarlo_splits(data, n_splits=args.n_splits,
                                      test_size=args.test_size, SEED=SEED)
    else:
        folds = get_patient_kfold_splits(data, n_splits=args.n_folds, SEED=SEED)
    if args.max_folds is not None:
        folds = folds[:args.max_folds]
        print(f"[max_folds] 앞 {args.max_folds}개 fold만 실행 — 집계 결과는 불완전함")
    N_FOLDS = len(folds)
    print(f"CV scheme: {args.cv_scheme} | {N_FOLDS} folds")

    # VFS 임베딩은 환자 단위 파일(35개)을 1,254개 샘플이 공유하므로 미리 캐시한다.
    # DataLoader 워커가 fork되기 전(=여기)에 채워야 copy-on-write로 공유된다.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if is_dsr_active:
        preload_dsr_cache(data, DSR_K)
        if args.dsr_gpu_cache and device.type == 'cuda':
            build_dsr_gpu_cache(device)

    ### K-Fold Cross Validation
    fold_elapsed = []
    for fold_idx, (train_data, valid_data) in enumerate(folds):
        fold_t0 = time.time()
        print(f"\n{'='*60}")
        print(f"=== Fold {fold_idx + 1} / {N_FOLDS} ===")
        print(f"{'='*60}")

        fold_dir = os.path.join(SAVE_DIR, f'fold_{fold_idx}')
        # 34-fold LOPO는 중단/재개가 잦으므로 완료된 fold는 건너뛴다.
        # fold별로 재시딩하므로 재개해도 결과가 연속 실행과 동일하다.
        if all(os.path.exists(os.path.join(fold_dir, f"inference_results_{ln}.csv"))
               for ln in ["보상조음", "과다비성"]):
            print(f"  이미 완료됨 — 건너뜀")
            continue
        make_dirs(fold_dir)
        set_SEED(SEED + fold_idx)

        train_df = pd.DataFrame(train_data)
        valid_df = pd.DataFrame(valid_data)
        print(f"Train: {len(train_data)} samples, {train_df['pid'].nunique()} patients | "
              f"보상조음 {train_df['보상조음'].mean():.4f} | 과다비성 {train_df['과다비성'].mean():.4f}")
        print(f"Valid: {len(valid_data)} samples, {valid_df['pid'].nunique()} patients | "
              f"보상조음 {valid_df['보상조음'].mean():.4f} | 과다비성 {valid_df['과다비성'].mean():.4f}")

        # dbg_load_all: 데이터로더 최적화가 수치에 영향을 주지 않는지 검증하기 위한 플래그.
        # 모델의 active_modality는 그대로 두고 로딩만 예전처럼 전부 수행한다.
        ds_modality = None if args.dbg_load_all else active_modality
        train_dataset = Mydataset(train_data, processor, DSR_K, ds_modality)
        valid_dataset = Mydataset(valid_data, processor, DSR_K, ds_modality)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

        ### model, optimizer, criterion (fresh per fold)
        model = Baseline(
            active_modality=active_modality,

            audio_model_name=audio_model_name,
            audio_d_out=audio_d_out,
            audio_dropout=audio_dropout,
            freeze_feature_extractor=freeze_feature_extractor,
            unfreeze_last_n_layers=unfreeze_last_n_layers,

            video_input_dim=video_input_dim,
            video_d_model=video_d_model,
            video_d_out=video_d_out,
            video_dropout=video_dropout,

            dsr_input_dim=dsr_input_dim,
            dsr_d_model=dsr_d_model,
            dsr_d_small=dsr_d_small,
            dsr_dropout=dsr_dropout,
            dsr_alpha=dsr_alpha,

            nume_input_dim=nume_input_dim,
            nume_hidden_dim=nume_hidden_dim,
            nume_output_dim=nume_output_dim,
            cate_input_dims=cate_input_dims,
            cate_emb_dims=cate_emb_dims,
            tab_final_output_dim=tab_final_output_dim,
            tab_dropout=tab_dropout,
        ); model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)

        # pos_weight는 반드시 해당 fold의 '학습셋'에서 계산해야 한다.
        # 전역값을 고정 사용하면 (a) 전체 클래스 비율이 학습에 누출되고,
        # (b) 양성 환자를 hold-out한 fold에서 positive가 최대 2배 과소가중되어
        #     그 fold의 예측이 체계적으로 낮아진다(관측된 스케일 드리프트의 원인).
        if args.pos_weight_mode == 'global':
            pos_weight = torch.tensor([20.28, 14.67], dtype=torch.float32)
        else:
            n_pos = train_df[["보상조음", "과다비성"]].values.sum(axis=0)
            n_neg = len(train_df) - n_pos
            pos_weight = torch.tensor(n_neg / np.maximum(n_pos, 1), dtype=torch.float32)
        print(f"  pos_weight ({args.pos_weight_mode}): "
              f"보상조음 {pos_weight[0]:.2f}, 과다비성 {pos_weight[1]:.2f}")
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

        ### Training & Validation Loop
        train_history_list = []
        valid_history_list = []
        for epoch in range(1, EPOCHS + 1):
            print(f"\n--- Fold {fold_idx + 1}, Epoch: {epoch} / {EPOCHS} ---")

            train_history = train(
                args, device, model, optimizer, criterion, train_loader, accum_step=ACCUM_STEP
            )

            valid_history = evaluate(
                args, device, model, criterion, valid_loader, is_inference=False
            )

            train_history_list.append(train_history)
            valid_history_list.append(valid_history)

        ### Save Training Logs
        pd.DataFrame(train_history_list).to_csv(
            os.path.join(fold_dir, "train_history.csv"), index=False, encoding='utf-8-sig')
        pd.DataFrame(valid_history_list).to_csv(
            os.path.join(fold_dir, "valid_history.csv"), index=False, encoding='utf-8-sig')

        ### Save Model Weights
        # LOPO는 fold가 34개라 체크포인트가 ~88GB에 달하고, 평가에는 OOF 예측만
        # 필요하므로 기본적으로 저장하지 않는다.
        if args.save_model:
            MODEL_DIR = os.path.join("MODEL", args.save_dir, f"fold_{fold_idx}")
            make_dirs(MODEL_DIR)
            model_path = os.path.join(MODEL_DIR, "model.pt")
            torch.save(model.state_dict(), model_path)
            print(f"  Model saved: {model_path}")

        ### Inference
        print(f"\n=== Fold {fold_idx + 1} Inference ===")
        (inference_history, (predicted_probas_0, predicted_labels_0, labels_0,
                             predicted_probas_1, predicted_labels_1, labels_1)) = evaluate(
            args, device, model, criterion, valid_loader, is_inference=True)

        # valid_loader는 shuffle=False이므로 예측 순서가 valid_data 순서와 일치한다.
        # 환자 단위 cluster bootstrap을 위해 pid/word를 함께 저장.
        for label_name, pp, pl, lb in [
            ("보상조음", predicted_probas_0, predicted_labels_0, labels_0),
            ("과다비성", predicted_probas_1, predicted_labels_1, labels_1),
        ]:
            assert len(pp) == len(valid_df), \
                f"예측 {len(pp)}개 != 검증 샘플 {len(valid_df)}개 (fold {fold_idx})"
            pd.DataFrame({
                "pid": valid_df["pid"].values,
                "word": valid_df["word"].values,
                "predicted_probas": pp.tolist(),
                "predicted_labels": pl.tolist(),
                "label": lb.tolist(),
            }).to_csv(os.path.join(fold_dir, f"inference_results_{label_name}.csv"),
                      index=False, encoding='utf-8-sig')

        dt = time.time() - fold_t0
        fold_elapsed.append(dt)
        remain = (N_FOLDS - fold_idx - 1) * np.mean(fold_elapsed)
        print(f"  Fold {fold_idx + 1} 소요 {dt/60:.1f}분 | "
              f"평균 {np.mean(fold_elapsed)/60:.1f}분 | 남은 예상 {remain/3600:.1f}시간")

    ### Aggregate all fold results
    print(f"\n{'='*60}")
    print(f"=== Aggregating {N_FOLDS} Folds ===")
    print(f"총 소요 {np.sum(fold_elapsed)/3600:.2f}시간 (fold당 평균 {np.mean(fold_elapsed)/60:.1f}분)")
    print(f"{'='*60}")
    for label_name in ["보상조음", "과다비성"]:
        all_dfs, missing = [], []
        for fi in range(N_FOLDS):
            fold_path = os.path.join(SAVE_DIR, f'fold_{fi}', f'inference_results_{label_name}.csv')
            if not os.path.exists(fold_path):
                missing.append(fi)
                continue
            fold_df = pd.read_csv(fold_path)
            fold_df['fold'] = fi
            all_dfs.append(fold_df)
        if missing:
            print(f"  [경고] {label_name}: fold {missing} 누락 — 집계 결과가 불완전합니다")
        aggregated = pd.concat(all_dfs, ignore_index=True)
        agg_path = os.path.join(SAVE_DIR, f"inference_results_{label_name}.csv")
        aggregated.to_csv(agg_path, index=False, encoding='utf-8-sig')
        print(f"  {label_name}: {len(aggregated)} samples -> {agg_path}")


def parse_args():
    parser = argparse.ArgumentParser(description='CP-Speech #2 Training')

    # 사용자 설정
    parser.add_argument('--SEED', type=int, default=17)

    parser.add_argument('--save_root', type=str, default='./RESULT', help="Path to results root directory")
    parser.add_argument('--save_dir', type=str, default='./baseline_save', help="Directory to save model checkpoints and logs")

    parser.add_argument('--cv_scheme', type=str, default='kfold', choices=['kfold', 'lopo', 'mccv'],
                        help="kfold: patient-level StratifiedGroupKFold | lopo: leave-one-patient-out | mccv: Monte-Carlo CV (양성 포함 제약)")
    parser.add_argument('--n_splits', type=int, default=20, help="Number of Monte-Carlo CV splits (cv_scheme=mccv)")
    parser.add_argument('--test_size', type=int, default=8, help="Validation patients per Monte-Carlo split (cv_scheme=mccv)")
    parser.add_argument('--n_folds', type=int, default=5, help="Number of K-fold cross-validation folds (patient-level, cv_scheme=kfold only)")
    parser.add_argument('--max_folds', type=int, default=None, help="Run only the first N folds (timing/debug)")
    parser.add_argument('--save_model', type=str2bool, default=True, help="Save per-fold checkpoints (~382MB each; disable for LOPO)")
    parser.add_argument('--dbg_load_all', type=str2bool, default=False, help="Debug: load all modalities in the dataset regardless of active_modality")
    parser.add_argument('--dsr_gpu_cache', type=str2bool, default=True, help="VFS 임베딩 전체(2.4GB)를 GPU에 상주시켜 배치별 전송 제거")
    parser.add_argument('--pos_weight_mode', type=str, default='fold', choices=['fold', 'global'],
                        help="fold: fold별 학습셋에서 계산(정상) | global: 전체 데이터 기준 고정값(기존 동작, 재현용)")
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=3e-5)
    parser.add_argument('--accum_step', type=int, default=1)
    parser.add_argument('--dsr_k', type=int, default=8, help="DSR K value (e.g., 8, 16, 32)")

    parser.add_argument('--label_frequency_0', type=float, default=0.0323, help="Frequency of label 0 (보상조음)")
    parser.add_argument('--label_frequency_1', type=float, default=0.0638, help="Frequency of label 1 (과디비성)")

    parser.add_argument('--is_audio_active', type=str2bool, default=True, help="Active Audio to use")
    parser.add_argument('--is_video_active', type=str2bool, default=True, help="Active Video to use")
    parser.add_argument('--is_dsr_active', type=str2bool, default=True, help="Active DSR to use")
    parser.add_argument('--is_tabular_active', type=str2bool, default=True, help="Active Tabular to use")

    # ======================
    # Audio
    # ======================
    parser.add_argument("--audio_model_name", type=str, default="Kkonjeong/wav2vec2-base-korean", help="Pretrained wav2vec2 model name")
    parser.add_argument("--audio_d_out", type=int, default=256, help="Output dimension of AudioModel")
    parser.add_argument("--audio_dropout", type=float, default=0.1, help="Dropout rate for AudioModel head")
    parser.add_argument("--freeze_feature_extractor", action="store_true", default=True, help="Freeze wav2vec2 CNN feature extractor")
    parser.add_argument("--unfreeze_last_n_layers", type=int, default=2, help="Unfreeze last N transformer layers of wav2vec2 (e.g., 2, 4). None means fully frozen")

    # ======================
    # Video
    # ======================
    parser.add_argument("--video_input_dim", type=int, default=1024, help="Input dimension of video embeddings")
    parser.add_argument("--video_d_model", type=int, default=512, help="Hidden dimension in VideoModel")
    parser.add_argument("--video_d_out", type=int, default=256, help="Output dimension of VideoModel")
    parser.add_argument("--video_dropout", type=float, default=0.1, help="Dropout rate for VideoModel")

    # ======================
    # DSR (CT)
    # ======================
    parser.add_argument("--dsr_input_dim", type=int, default=768, help="Input dimension of DSR embeddings")
    parser.add_argument("--dsr_d_model", type=int, default=256, help="Hidden dimension in DSRModel")
    parser.add_argument("--dsr_d_small", type=int, default=64, help="Bottleneck dimension of DSRModel")
    parser.add_argument("--dsr_dropout", type=float, default=0.3, help="Dropout rate for DSRModel bottleneck")
    parser.add_argument("--dsr_alpha", type=float, default=0.2, help="Scaling factor for DSR contribution")

    # ======================
    # Tabular
    # ======================
    parser.add_argument("--nume_input_dim", type=int, default=4, help="Number of numerical tabular features")
    parser.add_argument("--nume_hidden_dim", type=int, default=64, help="Hidden dimension for numerical features")
    parser.add_argument("--nume_output_dim", type=int, default=16, help="Output dimension for numerical embedding")
    parser.add_argument("--cate_input_dims",type=int, nargs="+", default=[2, 3, 2, 2], help="Cardinalities of categorical features")
    parser.add_argument("--cate_emb_dims", type=int, nargs="+", default=[16, 16, 16, 16], help="Embedding dimensions of categorical features")
    parser.add_argument("--tab_final_output_dim", type=int, default=32, help="Final output dimension of TabularModel")
    parser.add_argument("--tab_dropout", type=float, default=0.3, help="Dropout rate for TabularModel")

    # ======================
    # Fusion Head
    # ======================
    parser.add_argument("--fusion_hidden", type=int, default=256, help="Hidden dimension of fusion classifier")
    parser.add_argument("--fusion_dropout", type=float, default=0.1, help="Dropout rate of fusion classifier")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)

"""
# ============================================
# K-Fold CV (patient-level split) 실험
# ============================================

# 1) Audio Only
CUDA_VISIBLE_DEVICES=1 python main.py \
--is_audio_active True \
--is_video_active False \
--is_dsr_active False \
--is_tabular_active False \
--save_root ./RESULT \
--save_dir ./Audio_Only \
--n_folds 5 \
--epochs 8 \
--batch_size 32 \
--learning_rate 5e-6 \
--accum_step 1 \
--dsr_k 8

# 2) Audio + Tabular
CUDA_VISIBLE_DEVICES=1 python main.py \
--is_audio_active True \
--is_video_active False \
--is_dsr_active False \
--is_tabular_active True \
--save_root ./RESULT \
--save_dir ./Audio_Tabular \
--n_folds 5 \
--epochs 8 \
--batch_size 32 \
--learning_rate 5e-6 \
--accum_step 1 \
--dsr_k 8

# 3) Audio + Video
CUDA_VISIBLE_DEVICES=1 python main.py \
--is_audio_active True \
--is_video_active True \
--is_dsr_active False \
--is_tabular_active False \
--save_root ./RESULT \
--save_dir ./Audio_Video \
--n_folds 5 \
--epochs 8 \
--batch_size 32 \
--learning_rate 1e-5 \
--accum_step 1 \
--dsr_k 8

# 4) Audio + VFS
CUDA_VISIBLE_DEVICES=1 python main.py \
--is_audio_active True \
--is_video_active False \
--is_dsr_active True \
--is_tabular_active False \
--save_root ./RESULT \
--save_dir ./Audio_VFS \
--n_folds 5 \
--epochs 8 \
--batch_size 32 \
--learning_rate 1e-5 \
--accum_step 1 \
--dsr_k 8

# 5) Audio + Video + Tabular
CUDA_VISIBLE_DEVICES=1 python main.py \
--is_audio_active True \
--is_video_active True \
--is_dsr_active False \
--is_tabular_active True \
--save_root ./RESULT \
--save_dir ./Audio_Video_Tabular \
--n_folds 5 \
--epochs 8 \
--batch_size 32 \
--learning_rate 1e-5 \
--accum_step 1 \
--dsr_k 8

# 6) Audio + Video + VFS
CUDA_VISIBLE_DEVICES=1 python main.py \
--is_audio_active True \
--is_video_active True \
--is_dsr_active True \
--is_tabular_active False \
--save_root ./RESULT \
--save_dir ./Audio_Video_VFS \
--n_folds 5 \
--epochs 8 \
--batch_size 32 \
--learning_rate 1e-5 \
--accum_step 1 \
--dsr_k 8

# 7) Full Modality (Audio + Video + VFS + Tabular)
CUDA_VISIBLE_DEVICES=1 python main.py \
--save_root ./RESULT \
--save_dir ./Full_Modality \
--n_folds 5 \
--epochs 8 \
--batch_size 32 \
--learning_rate 1e-5 \
--accum_step 1 \
--dsr_k 8

# 7-1) Full Modality — VFS K=16
CUDA_VISIBLE_DEVICES=0 python main.py \
--save_root ./RESULT \
--save_dir ./Full_Modality_K16 \
--n_folds 5 \
--epochs 8 \
--batch_size 32 \
--learning_rate 2e-6 \
--accum_step 1 \
--dsr_k 16

# 7-2) Full Modality — VFS K=32
CUDA_VISIBLE_DEVICES=1 python main.py \
--save_root ./RESULT \
--save_dir ./Full_Modality_K32 \
--n_folds 5 \
--epochs 8 \
--batch_size 32 \
--learning_rate 3e-6 \
--accum_step 1 \
--dsr_k 32
"""