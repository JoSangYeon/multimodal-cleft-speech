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


# --- VFS(DSR) 임베딩 캐시 ---------------------------------------------------
# VFS는 환자 단위 파일이라 1,254개 샘플이 35개 파일을 공유한다. 캐시가 없으면
# 같은 67MB 배열을 epoch마다 ~36번씩 다시 읽어 1 epoch당 80.5GB를 읽게 된다.
# 전부 캐시해도 35 x 67.3MB = 2.3GB이므로 메모리 부담이 없다.
#
# 모듈 레벨에 두어 (1) train/valid 두 Dataset이 공유하고, (2) DataLoader 워커가
# fork되기 전에 채워져 copy-on-write로 공유되도록 한다. 워커 안에서 채우면
# 워커 수만큼 사본이 생기고 epoch마다 다시 읽는다.
_DSR_CACHE = {}

# GPU 상주 캐시. VFS는 배치당 2.15GB(32 x 67.3MB)라 워커->메인->GPU 전송이
# 학습 시간의 대부분을 차지한다(fold당 654GB). 전체 캐시가 2.36GB뿐이므로
# GPU에 통째로 올려두고 Dataset은 인덱스만 반환하면 이 전송이 사라진다.
_DSR_GPU = None          # (n_files, 2K, T, D) 텐서
_DSR_KEY2IDX = {}        # (dsr_path, K) -> row index


def preload_dsr_cache(data, dsr_k, verbose=True):
    """data에 등장하는 모든 dsr_path의 임베딩을 미리 읽어 캐시한다."""
    paths = sorted({s['dsr_path'] for s in data})
    todo = [p for p in paths if (p, dsr_k) not in _DSR_CACHE]
    if not todo:
        return
    it = tqdm(todo, desc='VFS 캐시 로딩', ncols=100) if verbose else todo
    for p in it:
        files = sorted(glob.glob(os.path.join(p, f"*_K{dsr_k}.npy")))
        arr = np.concatenate([np.load(f) for f in files], axis=0)
        _DSR_CACHE[(p, dsr_k)] = torch.from_numpy(arr)
    if verbose:
        total = sum(v.numel() * v.element_size() for v in _DSR_CACHE.values())
        print(f"  VFS 캐시: {len(_DSR_CACHE)}개 파일, {total/1e9:.2f} GB")


def build_dsr_gpu_cache(device, verbose=True):
    """CPU 캐시를 하나의 GPU 텐서로 올리고 key->row 매핑을 만든다.
    이후 Dataset은 실제 배열 대신 row index를 반환한다."""
    global _DSR_GPU, _DSR_KEY2IDX
    if not _DSR_CACHE:
        raise RuntimeError("preload_dsr_cache()를 먼저 호출해야 합니다")
    keys = sorted(_DSR_CACHE.keys())
    shapes = {tuple(_DSR_CACHE[k].shape) for k in keys}
    if len(shapes) != 1:
        raise RuntimeError(f"VFS 배열 shape이 균일하지 않습니다: {shapes}")
    _DSR_KEY2IDX = {k: i for i, k in enumerate(keys)}
    _DSR_GPU = torch.stack([_DSR_CACHE[k] for k in keys]).to(device)
    if verbose:
        print(f"  VFS GPU 캐시: {tuple(_DSR_GPU.shape)}, "
              f"{_DSR_GPU.numel()*_DSR_GPU.element_size()/1e9:.2f} GB on {device}")


def resolve_dsr(x, device):
    """DataLoader가 넘긴 값을 실제 VFS 텐서로 바꾼다.
    GPU 캐시 모드에서는 (B,) int64 인덱스가 오므로 GPU에서 gather한다."""
    if _DSR_GPU is not None and x.dtype == torch.int64 and x.dim() == 1:
        return _DSR_GPU[x.to(_DSR_GPU.device)]
    return x.to(device)

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
    def __init__(self, data, audio_processor, DSR_K=16, active_modality=None):
        """
        active_modality: {'audio','video','dsr','tabular'} -> bool.
        비활성 모달리티는 로드하지 않고 빈 텐서를 반환한다. 모델 forward가
        해당 텐서를 사용하지 않으므로 결과는 동일하며, 샘플당 최대 75.7MB
        (video 8.4MB + VFS 67.3MB)의 불필요한 디스크 I/O를 제거한다.
        None이면 전부 로드한다(기존 동작).
        """
        super(Mydataset, self).__init__()
        self.data = data
        self.audio_processor = audio_processor

        self.DSR_K = DSR_K
        self.active_modality = active_modality or {}
        self.load_video = self.active_modality.get('video', True)
        self.load_dsr = self.active_modality.get('dsr', True)

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
        if self.load_video:
            video_clip_file = sample['video_clip_npy']
            video_clip_embed = np.load(video_clip_file)
            video_clip_embed = torch.tensor(video_clip_embed) # (8, 256, 1024)
        else:
            video_clip_embed = torch.empty(0)

        ### DSR(CT)
        if self.load_dsr:
            key = (sample['dsr_path'], self.DSR_K)
            if _DSR_GPU is not None:
                # GPU 캐시 모드: 67MB 배열 대신 row index만 넘긴다.
                # 실제 gather는 학습 루프의 resolve_dsr()에서 GPU 내부에서 일어난다.
                dsr_embed = torch.tensor(_DSR_KEY2IDX[key], dtype=torch.long)
            else:
                dsr_embed = _DSR_CACHE.get(key)
                if dsr_embed is None:  # 캐시 미적재 시 폴백 (기존 동작)
                    files = sorted(glob.glob(os.path.join(key[0], f"*_K{self.DSR_K}.npy")))
                    dsr_embed = torch.from_numpy(
                        np.concatenate([np.load(f) for f in files], axis=0))
                    _DSR_CACHE[key] = dsr_embed
                # 캐시 텐서는 읽기 전용으로만 쓰이므로 복사하지 않는다
        else:
            dsr_embed = torch.empty(0)

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