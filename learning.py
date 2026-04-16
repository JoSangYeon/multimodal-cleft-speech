import os
import math
import numpy as np
import pandas as pd

from tqdm import tqdm
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import (roc_auc_score, 
                             average_precision_score,
                             accuracy_score,  
                             recall_score,
                             precision_score,
                             f1_score,
                             brier_score_loss,) # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.brier_score_loss.html


@torch.no_grad()
def multilabel_accuracy(logits, targets, threshold=0.5):
    """
    logits: (B, 2)
    targets: (B, 2) in {0,1}

    returns:
      acc0: label 0 accuracy (보상조음)
      acc1: label 1 accuracy (과다비성)
      acc_mean: (acc0 + acc1) / 2
    """
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).long()
    t = targets.long()

    acc0 = (preds[:, 0] == t[:, 0]).float().mean()
    acc1 = (preds[:, 1] == t[:, 1]).float().mean()
    acc_mean = (acc0 + acc1) / 2.0

    return acc0, acc1, acc_mean


def train(args, device, model, optimizer, criterion, train_loader, accum_step=1):
    model.train()

    running_loss = 0.0
    running_acc0 = 0.0
    running_acc1 = 0.0
    running_acc_mean = 0.0
    n_batches = 0

    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(train_loader, desc='Training', ncols=140)
    for batch_idx, (((audio_inputs, attention_mask), video_clip_embed, dsr_embed, (cate_tabular, nume_tabular)), target) in enumerate(pbar):        
        data = {
            "audio_input_values": audio_inputs.to(device),
            "audio_attention_mask": attention_mask.to(device),
            "video_x": video_clip_embed.to(device),
            "dsr_x": dsr_embed.to(device),
            "cate_tabular": cate_tabular.to(device),
            "nume_tabular": nume_tabular.to(device),
        }
        target = target.to(device)

        logits = model(**data)  # (B, 2)

        loss = criterion(logits, target.float())
        loss = loss / accum_step
        loss.backward()

        if (batch_idx + 1) % accum_step == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            running_loss += loss.item() * accum_step

            acc0, acc1, acc_mean = multilabel_accuracy(logits, target)
            running_acc0 += acc0.item()
            running_acc1 += acc1.item()
            running_acc_mean += acc_mean.item()
            n_batches += 1

            pbar.set_postfix({
                "loss": round(running_loss / n_batches, 6),
                "보상조음": round(running_acc0 / n_batches, 4),
                "과다비성": round(running_acc1 / n_batches, 4),
                "평균": round(running_acc_mean / n_batches, 4),
            })

    history = {
        "loss": running_loss / max(n_batches, 1),
        "acc_label0": running_acc0 / max(n_batches, 1),
        "acc_label1": running_acc1 / max(n_batches, 1),
        "acc_mean": running_acc_mean / max(n_batches, 1),
    }
    return history

def evaluate(args, device, model, criterion, data_loader, is_inference=False):
    model.eval()
    
    predicted_0 = torch.tensor([]).to(device)
    labels_0 = torch.tensor([]).to(device)

    predicted_1 = torch.tensor([]).to(device)
    labels_1 = torch.tensor([]).to(device)

    with torch.no_grad():
        running_loss = 0.0
        running_acc0 = 0.0
        running_acc1 = 0.0
        running_acc_mean = 0.0
        n_batches = 0

        pbar = tqdm(data_loader, desc='Inferencing' if is_inference else 'Evaluating', ncols=140)
        for batch_idx, (((audio_inputs, attention_mask), video_clip_embed, dsr_embed, (cate_tabular, nume_tabular)), target) in enumerate(pbar):        
            data = {
                "audio_input_values": audio_inputs.to(device),
                "audio_attention_mask": attention_mask.to(device),
                "video_x": video_clip_embed.to(device),
                "dsr_x": dsr_embed.to(device),
                "cate_tabular": cate_tabular.to(device),
                "nume_tabular": nume_tabular.to(device),
            }
            target = target.to(device)

            logits = model(**data)  # (B, 2)

            loss = criterion(logits, target.float())

            running_loss += loss.item()

            acc0, acc1, acc_mean = multilabel_accuracy(logits, target)
            running_acc0 += acc0.item()
            running_acc1 += acc1.item()
            running_acc_mean += acc_mean.item()
            n_batches += 1


            predicted_0 = torch.concat([predicted_0, logits[:, 0]], dim=0)
            labels_0 = torch.concat([labels_0, target[:, 0]], dim=0)
            predicted_1 = torch.concat([predicted_1, logits[:, 1]], dim=0)
            labels_1 = torch.concat([labels_1, target[:, 1]], dim=0)

            pbar.set_postfix({
                "loss": round(running_loss / n_batches, 6),
                "보상조음": round(running_acc0 / n_batches, 4),
                "과다비성": round(running_acc1 / n_batches, 4),
                "평균": round(running_acc_mean / n_batches, 4),
            })

    predicted_probas_0 = torch.sigmoid(predicted_0)
    predicted_labels_0 = torch.where(predicted_0 >= args.label_frequency_0 , 1, 0)
    predicted_probas_1 = torch.sigmoid(predicted_1)
    predicted_labels_1 = torch.where(predicted_1 >= args.label_frequency_1 , 1, 0)

    predicted_probas_0 = predicted_probas_0.detach().cpu().numpy()
    predicted_labels_0 = predicted_labels_0.detach().cpu().numpy()
    predicted_probas_1 = predicted_probas_1.detach().cpu().numpy()
    predicted_labels_1 = predicted_labels_1.detach().cpu().numpy()
    labels_0 = labels_0.detach().cpu().numpy()
    labels_1 = labels_1.detach().cpu().numpy()

    history = {"loss": running_loss / max(n_batches, 1)}

    for suffix, y_true, y_prob, y_pred in [
        ("보상조음", labels_0, predicted_probas_0, predicted_labels_0),
        ("과다비성", labels_1, predicted_probas_1, predicted_labels_1),
    ]:
        has_both_classes = len(np.unique(y_true)) >= 2
        history[f"{suffix}_Accuracy"] = accuracy_score(y_true, y_pred)
        history[f"{suffix}_Recall"] = recall_score(y_true, y_pred, zero_division=0)
        history[f"{suffix}_Precision"] = precision_score(y_true, y_pred, zero_division=0)
        history[f"{suffix}_F1"] = f1_score(y_true, y_pred, zero_division=0)
        history[f"{suffix}_AUROC"] = roc_auc_score(y_true, y_prob) if has_both_classes else float('nan')
        history[f"{suffix}_AUPRC"] = average_precision_score(y_true, y_prob) if has_both_classes else float('nan')
        history[f"{suffix}_Brier"] = brier_score_loss(y_true, y_prob)

    if is_inference:
        return history, (predicted_probas_0, predicted_labels_0, labels_0, predicted_probas_1, predicted_labels_1, labels_1)
    else:
        return history