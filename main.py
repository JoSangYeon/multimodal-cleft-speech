import os
import math
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
    N_FOLDS = args.n_folds
    folds = get_patient_kfold_splits(data, n_splits=N_FOLDS, SEED=SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ### K-Fold Cross Validation
    for fold_idx, (train_data, valid_data) in enumerate(folds):
        print(f"\n{'='*60}")
        print(f"=== Fold {fold_idx + 1} / {N_FOLDS} ===")
        print(f"{'='*60}")

        train_df = pd.DataFrame(train_data)
        valid_df = pd.DataFrame(valid_data)
        print(f"Train: {len(train_data)} samples, {train_df['pid'].nunique()} patients | "
              f"보상조음 {train_df['보상조음'].mean():.4f} | 과다비성 {train_df['과다비성'].mean():.4f}")
        print(f"Valid: {len(valid_data)} samples, {valid_df['pid'].nunique()} patients | "
              f"보상조음 {valid_df['보상조음'].mean():.4f} | 과다비성 {valid_df['과다비성'].mean():.4f}")

        fold_dir = os.path.join(SAVE_DIR, f'fold_{fold_idx}')
        make_dirs(fold_dir)

        train_dataset = Mydataset(train_data, processor, DSR_K)
        valid_dataset = Mydataset(valid_data, processor, DSR_K)

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
        pos_weight = torch.tensor([20.28, 14.67], dtype=torch.float32).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

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

        ### Inference
        print(f"\n=== Fold {fold_idx + 1} Inference ===")
        (inference_history, (predicted_probas_0, predicted_labels_0, labels_0,
                             predicted_probas_1, predicted_labels_1, labels_1)) = evaluate(
            args, device, model, criterion, valid_loader, is_inference=True)

        for label_name, pp, pl, lb in [
            ("보상조음", predicted_probas_0, predicted_labels_0, labels_0),
            ("과다비성", predicted_probas_1, predicted_labels_1, labels_1),
        ]:
            pd.DataFrame({
                "predicted_probas": pp.tolist(),
                "predicted_labels": pl.tolist(),
                "label": lb.tolist(),
            }).to_csv(os.path.join(fold_dir, f"inference_results_{label_name}.csv"),
                      index=False, encoding='utf-8-sig')

    ### Aggregate all fold results
    print(f"\n{'='*60}")
    print(f"=== Aggregating {N_FOLDS} Folds ===")
    print(f"{'='*60}")
    for label_name in ["보상조음", "과다비성"]:
        all_dfs = []
        for fi in range(N_FOLDS):
            fold_path = os.path.join(SAVE_DIR, f'fold_{fi}', f'inference_results_{label_name}.csv')
            fold_df = pd.read_csv(fold_path)
            fold_df['fold'] = fi
            all_dfs.append(fold_df)
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

    parser.add_argument('--n_folds', type=int, default=5, help="Number of K-fold cross-validation folds (patient-level)")
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