import os
import glob
import json
import wandb
import shutil
import random
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def split_train_test(data, test_ratio=0.2, SEED=17):
    random.shuffle(data)
    sep_idx = int(len(data) * (1 - test_ratio))
    return data[:sep_idx], data[sep_idx:]

def get_patient_kfold_splits(data, n_splits=5, SEED=17):
    """
    환자(pid) 기준 Stratified Group K-Fold split.
    같은 환자의 샘플이 train/test에 동시에 나타나지 않도록 보장.
    두 라벨(보상조음, 과다비성)의 환자 단위 조합으로 stratification.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    df = pd.DataFrame(data)

    # 환자 단위 라벨 조합으로 stratification
    patient_labels = df.groupby('pid').agg({'보상조음': 'max', '과다비성': 'max'})
    pid_to_strat = (patient_labels['보상조음'].astype(str) + '_' + patient_labels['과다비성'].astype(str)).to_dict()

    groups = df['pid'].values
    strat_labels = df['pid'].map(pid_to_strat).values

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    folds = []
    for train_idx, test_idx in sgkf.split(df, strat_labels, groups):
        train_data = [data[i] for i in train_idx]
        test_data = [data[i] for i in test_idx]
        folds.append((train_data, test_data))

    return folds

def save_to_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # 빈 줄은 건너뛰기
            data.append(json.loads(line))
    # print(f"Loaded {len(data)} items")
    return data

def set_SEED(SEED):
    random.seed(SEED) #  Python의 random 라이브러리가 제공하는 랜덤 연산이 항상 동일한 결과를 출력하게끔
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

def make_dirs(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)
    return directory
    
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')