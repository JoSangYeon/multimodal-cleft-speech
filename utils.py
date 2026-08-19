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

def get_lopo_splits(data):
    """
    환자(pid) 기준 Leave-One-Patient-Out split.
    환자 1명을 검증셋으로 사용하므로 fold 수 = 환자 수.

    fold당 검증 환자가 1명이라 대부분의 fold는 양성 샘플이 0개다.
    따라서 fold별 AUROC/AUPRC는 정의되지 않으며, 모든 fold의
    out-of-fold 예측을 pooling한 뒤 지표를 1회 산출해야 한다.
    신뢰구간은 환자 단위 cluster bootstrap으로 얻는다.
    """
    from sklearn.model_selection import LeaveOneGroupOut

    df = pd.DataFrame(data)
    groups = df['pid'].values

    logo = LeaveOneGroupOut()

    folds = []
    for train_idx, test_idx in logo.split(df, groups=groups):
        train_data = [data[i] for i in train_idx]
        test_data = [data[i] for i in test_idx]
        folds.append((train_data, test_data))

    return folds

def get_montecarlo_splits(data, n_splits=20, test_size=8,
                          min_pos=None, max_pos=None, SEED=17, max_tries=10000):
    """
    환자 단위 Monte-Carlo 교차검증 (반복 무작위 검증셋).

    K-fold와 달리 검증셋이 데이터를 분할하지 않으므로
      - 반복 횟수가 양성 환자 수에 제약받지 않고 (K <= 4 제약 없음),
      - 검증셋 크기를 학습셋 크기와 독립적으로 정할 수 있다.

    각 검증셋이 min_pos에 지정된 만큼의 양성 '환자'를 반드시 포함하도록
    제약을 걸어, AUROC가 정의되지 않는 fold가 발생하지 않는다.

    제약은 구성적으로 채우지 않고 기각표집(rejection sampling)으로 만족시킨다.
    양성 환자를 먼저 뽑고 나머지를 채우면 조건부 분포가 왜곡되기 때문이다.

    Parameters
    ----------
    n_splits : 반복 횟수
    test_size : 검증셋 환자 수
    min_pos : {라벨명: 최소 양성 환자 수}. 기본 {'보상조음': 1, '과다비성': 2}
    max_pos : {라벨명: 최대 양성 환자 수}. 기본 {'보상조음': 1}
        보상조음 양성 환자가 4명뿐이라, 검증셋에 2명 이상 들어가면 학습셋에
        남는 양성 환자가 2명 이하로 줄어 모델이 크게 열화된다(pos_weight 분석
        참조). 검증셋에 정확히 1명만 두어 학습셋이 항상 3명을 유지하도록 한다.
    """
    if min_pos is None:
        min_pos = {'보상조음': 1, '과다비성': 2}
    if max_pos is None:
        max_pos = {'보상조음': 1}

    df = pd.DataFrame(data)
    label_cols = sorted(set(min_pos) | set(max_pos))
    patient_labels = df.groupby('pid').agg({k: 'max' for k in label_cols})
    pids = patient_labels.index.to_numpy()
    if test_size >= len(pids):
        raise ValueError(f"test_size({test_size})가 환자 수({len(pids)})보다 작아야 합니다")

    rng = np.random.default_rng(SEED)
    pid_to_rows = {p: np.flatnonzero(df['pid'].values == p) for p in pids}

    splits, tries = [], 0
    while len(splits) < n_splits:
        tries += 1
        if tries > max_tries:
            raise RuntimeError(
                f"{max_tries}회 시도 후에도 제약을 만족하는 분할을 "
                f"{len(splits)}/{n_splits}개만 찾았습니다. "
                f"test_size를 늘리거나 min_pos를 낮추세요.")
        test_pids = rng.choice(pids, size=test_size, replace=False)
        sel = patient_labels.loc[test_pids]
        if any(int(sel[k].sum()) < v for k, v in min_pos.items()):
            continue
        if any(int(sel[k].sum()) > v for k, v in max_pos.items()):
            continue
        test_idx = np.concatenate([pid_to_rows[p] for p in test_pids])
        mask = np.ones(len(df), dtype=bool)
        mask[test_idx] = False
        train_idx = np.flatnonzero(mask)
        splits.append(([data[i] for i in train_idx], [data[i] for i in test_idx]))

    print(f"  Monte-Carlo CV: {n_splits}개 분할 (검증 환자 {test_size}명, "
          f"최소 {min_pos}, 최대 {max_pos}), "
          f"기각표집 시도 {tries}회 (수용률 {n_splits/tries:.1%})")
    return splits

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