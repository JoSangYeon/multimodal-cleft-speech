import os
import glob
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchaudio

from PIL import Image
from torchvision.transforms import v2

def load_and_preprocess_audio(
    file_path,
    processor=None,
    target_sr=16000,
    max_len_sec=2.5,   # 예: 5초
):
    speech, sr = torchaudio.load(file_path)   # (C, L)

    # mono
    if speech.shape[0] > 1:
        speech = speech.mean(dim=0, keepdim=True)

    # resample
    if sr != target_sr:
        speech = torchaudio.transforms.Resample(sr, target_sr)(speech)

    speech = speech.squeeze(0)  # (L,)
    max_len = int(target_sr * max_len_sec)

    length = speech.shape[0]

    if length < max_len:
        pad_len = max_len - length
        speech = torch.nn.functional.pad(speech, (0, pad_len))
        attention_mask = torch.cat([
            torch.ones(length),
            torch.zeros(pad_len)
        ])
    else:
        speech = speech[:max_len]
        attention_mask = torch.ones(max_len)

    if processor is not None:
        speech = processor.feature_extractor(speech, sampling_rate=target_sr).input_values[0]  # (max_len,)

    return speech, attention_mask

class Mydataset(Dataset):
    def __init__(self, data, audio_processor, DSR_K=16):
        super(Mydataset, self).__init__()
        self.data = data
        self.audio_processor = audio_processor

        self.DSR_K = DSR_K

        self.tabular_cols = [
            'sex', 'age_at_test_in_years', 
            'age at primary palatoplasty', 
            'veau class', 'Cleft GAP SIZE(mm)', 
            'Mean preop palatal length(mm)', 
            'Palatal fistula', 'Syndrome'
        ]
        self.cate_cols = [
            'sex', 'veau class', 'Palatal fistula', 'Syndrome'
        ]
        self.nume_cols = [
            'age_at_test_in_years', 'age at primary palatoplasty', 
            'Cleft GAP SIZE(mm)', 'Mean preop palatal length(mm)'
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        ### Audio
        audio_clip_file = sample['audio_clip_file']
        audio_inputs, attention_mask = load_and_preprocess_audio(audio_clip_file, self.audio_processor, 
                                                                 target_sr=16000, max_len_sec=2.5)
        
        ### Video
        video_clip_file = sample['video_clip_npy']
        video_clip_embed = np.load(video_clip_file)
        video_clip_embed = torch.tensor(video_clip_embed) # (8, 256, 1024)

        ### DSR(CT)
        dsr_root_path = sample['dsr_path']
        dsr_embed_path_list = glob.glob(os.path.join(dsr_root_path, f"*_K{self.DSR_K}.npy"))
        dsr_embed_list = [np.load(dsr_embed_path) for dsr_embed_path in dsr_embed_path_list]
        dsr_embed = np.concatenate(dsr_embed_list, axis=0)
        dsr_embed = torch.tensor(dsr_embed) # (2K, 1369, 768)

        ### Tabular
        cate_tabular = torch.tensor([sample[col] for col in self.cate_cols], dtype=torch.long)
        nume_tabular = torch.tensor([sample[col] for col in self.nume_cols], dtype=torch.float)
        tabular = (cate_tabular, nume_tabular)

        ### Label(보상조음, 과다비성)
        label = torch.tensor([sample["보상조음"], sample["과다비성"]], dtype=torch.long)

        return ((audio_inputs, attention_mask), video_clip_embed, dsr_embed, (cate_tabular, nume_tabular)), label

def main():
    pass

if __name__ == "__main__":
    main()