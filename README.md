# Detection of Compensatory Articulation and Hypernasality in Cleft Patients using a Multimodal Deep Learning Model

## Abstract

**Objective:** Clinical speech evaluation for cleft patients relies primarily on perceptual evaluation by speech-language pathologists, which may vary according to experience and clinical setting. The purpose of this study was to develop a multimodal deep learning model integrating audio recordings, facial video, videofluoroscopic (VFS) imaging, and clinical variables for detection of compensatory articulation and hypernasality.

**Methods:** Speech evaluation data from 34 Korean patients with repaired cleft palate were retrospectively analyzed. Each patient underwent a standardized articulation test consisting of 30 target words, yielding 1,254 word-level speech samples with synchronized audiovisual recordings. Imaging data from VFS and structured clinical variables, including sex, Veau classification, cleft width, and age at primary palatoplasty, were incorporated. Multimodal deep learning models were trained using combinations of audio, facial video, imaging, and clinical variables. Model performance was evaluated using patient-level five-fold cross-validation.

**Results:** For compensatory articulation detection, the multimodal model integrating audio, facial video, and VFS imaging achieved the highest performance (AUROC 0.76), outperforming the audio-only model (AUROC 0.71). In contrast, hypernasality detection showed greater improvement with the inclusion of VFS imaging and clinical variables, with the full-modality model achieving the highest AUROC (0.67). Modality contribution analysis demonstrated a stronger influence of visual articulatory information for compensatory articulation, whereas hypernasality-related abnormalities were more closely associated with anatomical and clinical factors.

**Conclusion:** Multimodal deep learning integrating complementary functional and anatomical information improves automated detection of cleft-related speech abnormalities. These findings support the potential role of multimodal AI as an objective adjunct to clinical speech evaluation.

## Project Structure

```
.
├── main.py          # Training & evaluation (K-fold CV)
├── model.py         # Model architecture (Audio, Video, VFS, Tabular encoders + Fusion)
├── dataset.py       # Dataset & DataLoader
├── learning.py      # Training & evaluation loops
├── plotting.py      # Result visualization (ROC/PRC curves, ablation plots)
├── ablation.py      # Video modality contribution analysis (Grad-CAM, attention)
├── utils.py         # Utility functions
├── _DATA/           # Data directory (not included, see Data Availability)
└── RESULT/          # Experiment results
```

## Requirements

- Python 3.12+
- CUDA 12.8+ (for GPU training)

### Installation

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
| pandas | 2.3.2 |
| numpy | 2.3.3 |
| matplotlib | 3.10.6 |

## Pretrained Model Weights

Trained model weights are available at the following link:

| Model | Download |
|-------|----------|
| Audio Only | [Link]() |
| Audio + Tabular | [Link]() |
| Audio + Video | [Link]() |
| Audio + VFS | [Link]() |
| Audio + Video + Tabular | [Link]() |
| Audio + Video + VFS | [Link]() |
| Full Modality (A+V+VFS+T) | [Link]() |

## Usage

### Training

```bash
# 1) Audio Only
CUDA_VISIBLE_DEVICES=0 python main.py \
  --is_audio_active True \
  --is_video_active False \
  --is_dsr_active False \
  --is_tabular_active False \
  --save_root ./RESULT \
  --save_dir ./Audio_Only \
  --n_folds 5 --epochs 8 --batch_size 32 --learning_rate 5e-6

# 7) Full Modality (Audio + Video + VFS + Tabular)
CUDA_VISIBLE_DEVICES=0 python main.py \
  --save_root ./RESULT \
  --save_dir ./Full_Modality \
  --n_folds 5 --epochs 8 --batch_size 32 --learning_rate 1e-5
```

See `main.py` for all 7 modality configurations.

### Evaluation & Plotting

```bash
# ROC/PRC curves for all modality combinations
python plotting.py

# Without std shading
python plotting.py --no_std_band

# VFS temporal sampling ablation (K=8, 16, 32)
python plotting.py --vfs_ablation
```

### Video Modality Contribution Analysis

```bash
CUDA_VISIBLE_DEVICES=0 python ablation.py \
  --fold_idx 0 --epochs 8 --batch_size 32 --learning_rate 1e-5
```

## Data Availability

Patient data (audio recordings, facial video, and VFS imaging) are not publicly available due to privacy and ethical considerations, as they contain identifiable facial information. Access may be granted upon reasonable request and with appropriate institutional approval.

## Citation

*Paper citation will be added upon publication.*
