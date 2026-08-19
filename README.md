# Detection of Compensatory Articulation and Hypernasality in Cleft Patients using a Multimodal Deep Learning Model

Source code for the study *"Multimodal Deep Learning for Detection of Compensatory Articulation and Hypernasality in Patients with Cleft Palate"*.

## Abstract

Clinical speech evaluation for patients with cleft palate relies primarily on perceptual assessment by speech-language pathologists, which may vary according to evaluator experience and clinical setting. This study aimed to evaluate a multimodal machine learning model integrating speech audio, facial video, videofluoroscopic (VFS) imaging, and clinical variables for automated detection of compensatory articulation and hypernasality. Speech data from 34 Korean patients with repaired cleft palate were retrospectively analyzed, yielding 1,254 word-level samples from 30 target words. Audiovisual recordings were synchronized at the word level, whereas VFS was incorporated as complementary patient-level information. Performance was evaluated using leave-one-patient-out cross-validation with patient-level bootstrap confidence intervals. For compensatory articulation, the full multimodal model yielded the highest AUROC point estimate (0.738; 95% CI, 0.666–0.868), whereas for hypernasality, Audio+VFS yielded the highest AUROC point estimate (0.682; 95% CI, 0.578–0.752). However, neither configuration showed a statistically significant improvement over audio alone. Modality contribution analysis suggested task-dependent patterns, with facial video showing a greater probability shift for compensatory articulation than for hypernasality. Multimodal machine learning integrating complementary functional and anatomical information may support the assessment of cleft-related speech abnormalities and provide adjunctive information for clinical decision-making.

## Project Structure

```
.
├── main.py                      # Training & evaluation driver (LOPO / K-fold)
├── model.py                     # Model architecture (Audio, Video, VFS, Tabular encoders + fusion)
├── dataset.py                   # Dataset & DataLoader (with VFS embedding cache)
├── learning.py                  # Training & evaluation loops
├── utils.py                     # Cross-validation splits and utilities
│
├── run_lopo.sh                  # Runs all 7 modality configurations under LOPO
├── lopo_eval.py                 # Pooled out-of-fold metrics + patient-level bootstrap CIs
├── make_manuscript_tables.py    # Generates Tables 2 and 3 in manuscript format
├── plot_lopo_curves.py          # Generates Figure 2 (ROC / precision–recall panels)
│
├── plotting.py                  # Result visualization for the K-fold analysis
├── ablation.py                  # Video modality contribution analysis (Grad-CAM, attention)
│
├── _DATA/                       # Data directory (not included; see Data Availability)
└── RESULT_LOPO_V2/              # LOPO experiment outputs
```

## Requirements

- Python 3.12+
- CUDA 12.8+ (for GPU training)

```bash
pip install -r requirements.txt
```

### Key Dependencies

| Package | Version |
|---------|---------|
| torch | 2.8.0+cu128 |
| torchaudio | 2.8.0+cu128 |
| transformers | 4.52.4 |
| scikit-learn | 1.7.2 |
| scipy | 1.16.2 |
| pandas | 2.3.2 |
| numpy | 2.3.3 |
| matplotlib | 3.10.6 |

## Reproducing the published results

The reported results use **leave-one-patient-out (LOPO) cross-validation**. Each of the 34 iterations
holds out all utterances from one patient; out-of-fold predictions are then pooled across all
34 iterations and each metric is computed once on the pooled set. Confidence intervals are obtained
by resampling the 34 patients with replacement (2,000 replicates), which also accounts for the
clustering of multiple utterances within patients.

```bash
# 1. Train all 7 modality configurations under LOPO (34 folds each).
#    Splits the workload across two GPUs; outputs to ./RESULT_LOPO_V2
bash run_lopo.sh 0    # GPU 0
bash run_lopo.sh 1    # GPU 1

# 2. Pooled metrics with patient-level cluster bootstrap CIs,
#    plus paired bootstrap comparisons against the audio-only model
python lopo_eval.py --result_root ./RESULT_LOPO_V2 --n_boot 2000

# 3. Tables 2 and 3 in manuscript format
python make_manuscript_tables.py --result_root ./RESULT_LOPO_V2

# 4. Figure 2 (ROC and precision–recall panels)
python plot_lopo_curves.py --result_root ./RESULT_LOPO_V2
```

### Notes on the implementation

- **Class weighting.** `pos_weight` for the loss is computed from each fold's *training* set
  (`--pos_weight_mode fold`, the default). Using a single global value causes systematic
  probability-scale drift across folds when the positive class is concentrated in a few patients.
  `--pos_weight_mode global` reproduces the earlier behaviour.
- **Operating point.** Threshold-dependent metrics (accuracy, precision, recall, F1) use a single
  threshold determined on the pooled out-of-fold predictions at a recall of 0.70. Because each LOPO
  iteration contains one patient — and for most patients no positive utterance — the threshold
  cannot be determined per iteration.
- **VFS input.** VFS embeddings are per patient, so `dataset.py` caches them
  (35 files, ~2.4 GB) and keeps them resident on the GPU, avoiding repeated transfer of the same
  arrays. This is a performance optimisation only; predictions are bit-identical.
- **VFS temporal sampling.** `--dsr_k K` selects `K` frames per view (lateral and frontal), so
  `K = 8` yields 16 frames in total. Frame features are aggregated by attention pooling rather than
  an explicit temporal architecture.

### K-fold analysis (earlier version)

The five-fold analysis is retained for reference and reproduces the modality contribution
analysis in Table 4.

```bash
python main.py --cv_scheme kfold --n_folds 5 \
  --is_audio_active True --is_video_active True \
  --is_dsr_active True --is_tabular_active True \
  --save_root ./RESULT --save_dir Full_Modality \
  --epochs 8 --batch_size 32 --learning_rate 5e-6

python plotting.py                  # ROC / PRC curves
python plotting.py --vfs_ablation   # VFS temporal sampling (K = 8, 16, 32)

CUDA_VISIBLE_DEVICES=0 python ablation.py \
  --fold_idx 0 --epochs 8 --batch_size 32 --learning_rate 1e-5
```

## Pretrained Model Weights

| Model | Download |
|-------|----------|
| Full Modality (Audio + Video + VFS + Tabular) | [Link](https://drive.google.com/drive/folders/1P6pJnJNE3RGkYUvYjTSzUo9vH7i7Of7d?usp=sharing) |

These checkpoints are from the five-fold analysis. LOPO training does not retain per-fold
checkpoints, since 34 folds × 7 configurations would exceed 80 GB and only the out-of-fold
predictions are required for evaluation.

## Data Availability

Patient data (audio recordings, facial video, and VFS imaging) are not publicly available due to
privacy and ethical considerations, as they contain identifiable facial information. Requests may be
directed to the corresponding author.

## Citation

*Paper citation will be added upon publication.*
