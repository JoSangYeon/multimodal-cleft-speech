import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchaudio

from transformers import AutoModel, AutoProcessor

def masked_mean_pooling(hidden_states, attention_mask):
    """
    hidden_states: (B, T, H)
    attention_mask: (B, T)  (1=valid, 0=pad)
    """
    mask = attention_mask.unsqueeze(-1).type_as(hidden_states)  # (B, T, 1)
    hidden_states = hidden_states * mask
    denom = mask.sum(dim=1).clamp(min=1.0)
    return hidden_states.sum(dim=1) / denom  # (B, H)

class AudioModel(nn.Module):
    def __init__(
        self,
        model_name="Kkonjeong/wav2vec2-base-korean",
        d_out=256,
        dropout=0.1,
        freeze_feature_extractor=True,
        unfreeze_last_n_layers=None,  # e.g., 4 (last 4 transformer blocks trainable)
    ):
        super().__init__()

        self.wav2vec2 = AutoModel.from_pretrained(model_name)  # Wav2Vec2Model
        hidden_size = self.wav2vec2.config.hidden_size

        # (선택) CNN feature extractor freeze (소규모 데이터에서 강추)
        if freeze_feature_extractor:
            self.wav2vec2.feature_extractor._freeze_parameters()

        # (선택) transformer encoder 부분 unfreeze
        if unfreeze_last_n_layers is not None:
            # 전체를 freeze 후 마지막 N개만 trainable로
            for p in self.wav2vec2.parameters():
                p.requires_grad = False
            if not freeze_feature_extractor:
                # feature extractor까지 열고 싶으면 별도 처리 (보통은 닫는 게 안전)
                for p in self.wav2vec2.feature_extractor.parameters():
                    p.requires_grad = True

            enc_layers = self.wav2vec2.encoder.layers
            for layer in enc_layers[-unfreeze_last_n_layers:]:
                for p in layer.parameters():
                    p.requires_grad = True

            # LayerNorm / projector도 열어주는 게 보통 유리
            for p in self.wav2vec2.encoder.layer_norm.parameters():
                p.requires_grad = True

        # projection head (fusion-friendly)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, d_out),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, input_values, attention_mask=None):
        # attention_mask가 없으면 "전부 유효"라고 가정
        if attention_mask is None:
            attention_mask = torch.ones_like(input_values, dtype=torch.long)

        out = self.wav2vec2(input_values=input_values, attention_mask=attention_mask)
        hs = out.last_hidden_state  # (B, T, H)

        # input length mask -> feature length mask (T)로 변환
        feat_mask = self.wav2vec2._get_feature_vector_attention_mask(hs.shape[1], attention_mask)

        pooled = masked_mean_pooling(hs, feat_mask)
        return self.head(pooled)

class AttnPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, h, dim_pool):
        # h: (..., N, dim)
        w = self.score(h).squeeze(-1)          # (..., N)
        w = torch.softmax(w, dim=dim_pool)     # (..., N)
        return (h * w.unsqueeze(-1)).sum(dim=dim_pool)


class VideoModel(nn.Module):
    def __init__(self, input_dim=1024, d_model=512, d_out=256, dropout=0.2):
        super().__init__()
        self.in_norm = nn.LayerNorm(input_dim)
        self.proj = nn.Linear(input_dim, d_model)

        self.token_pool = AttnPool(d_model)   # pool over 256 tokens
        self.frame_pool = AttnPool(d_model)   # pool over 8 frames

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_out),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """
        x: (B, F=8, T=256, D=1024)
        return: (B, d_out)
        """
        x = self.in_norm(x)
        h = self.proj(x)           # (B, F, T, d_model)

        # token pooling per frame: (B, F, d_model)
        hf = self.token_pool(h, dim_pool=2)

        # frame pooling: (B, d_model)
        h0 = self.frame_pool(hf, dim_pool=1)

        return self.head(h0)

class DSRModel(nn.Module):
    def __init__(self, input_dim:int=768, d_model=256, d_small=64,
                 patch_pool="attn", view_pool="attn",
                 dropout=0.5, alpha=0.2):
        super().__init__()
        self.alpha = alpha

        self.in_norm = nn.LayerNorm(input_dim)
        self.proj = nn.Linear(input_dim, d_model)

        # patch pooling
        if patch_pool == "attn":
            self.patch_attn = nn.Linear(d_model, 1)  # token score
        else:
            self.patch_attn = None

        # view pooling
        if view_pool == "attn":
            self.view_attn = nn.Linear(d_model, 1)
        else:
            self.view_attn = None

        # bottleneck (capacity control)
        self.bottleneck = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_small),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # optional: gate (learned but bounded)
        self.gate = nn.Sequential(
            nn.Linear(d_small, d_small),
            nn.Sigmoid()
        )

    def _attn_pool(self, h, scorer, dim):
        # h: (..., N, d), scorer: d->1
        w = scorer(h).squeeze(-1)          # (..., N)
        w = torch.softmax(w, dim=dim)      # (..., N)
        return (h * w.unsqueeze(-1)).sum(dim=dim)

    def forward(self, x):
        """
        x: (B, 2K, 1369, input_dim)
        return: (B, d_small)  (fusion용 representation)
        """
        B, V, T, Din = x.shape  # V=2K, T=1369
        x = self.in_norm(x)
        h = self.proj(x)  # (B, V, T, d_model)

        # ---- Patch pooling: (B, V, d_model)
        if self.patch_attn is not None:
            hv = self._attn_pool(h, self.patch_attn, dim=2)
        else:
            hv = h.mean(dim=2)

        # ---- View pooling: (B, d_model)
        if self.view_attn is not None:
            h0 = self._attn_pool(hv, self.view_attn, dim=1)
        else:
            h0 = hv.mean(dim=1)

        # ---- Bottleneck + gate + scaled output
        z = self.bottleneck(h0)           # (B, d_small)
        z = z * self.gate(z)              # gated (optional)
        return self.alpha * z             # 영향력 제한

class TabularModel(nn.Module):
    def __init__(self, nume_input_dim:int=4, nume_hidden_dim:int=64, nume_output_dim:int=16, 
                 cate_input_dims:list[int]=[2, 3, 2, 2], cate_emb_dims:list[int]=[16, 16, 16, 16], 
                 final_output_dim:int=32, dropout:float=0.8):
        super().__init__()

        ### Categorical Cols ###
        self.cate_embs = nn.ModuleList([
            nn.Embedding(card, dim)
            for card, dim in zip(cate_input_dims, cate_emb_dims)
        ])

        ### Numerical Cols ###
        self.nume_embs = nn.Sequential(
            nn.Linear(nume_input_dim, nume_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(nume_hidden_dim, nume_output_dim),
        )

        ### Final MLP ###
        self.mlp = nn.Sequential(
            nn.Linear(sum(cate_emb_dims) + nume_output_dim, sum(cate_emb_dims) + nume_output_dim//4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(sum(cate_emb_dims) + nume_output_dim//4, final_output_dim),
        )
    def forward(self, cate_tabular, nume_tabular) -> torch.Tensor:
        # cate_tabular: (B, num_cate)
        # nume_tabular: (B, num_nume)

        # 1. Categorical Embedding
        embs = []
        for i, emb in enumerate(self.cate_embs):
            # cate_tabular[:, i] -> (B,)
            embs.append(emb(cate_tabular[:, i]))

        # (B, sum(cate_emb_dims))
        cate_h = torch.cat(embs, dim=1)

        # 2. Numerical Embedding
        nume_h = self.nume_embs(nume_tabular)  # (B, nume_output_dim)

        # 3. Final MLP
        x = torch.cat([cate_h, nume_h], dim=1)  # (B, total_dim)
        x = self.mlp(x)                         # (B, final_output_dim)

        return x

# class Baseline(nn.Module):
#     """
#     Baseline: modality representations -> concat -> MLP -> 2 logits
#     Targets:
#       - 보상조음: 0/1
#       - 과다비성: 0/1
#     => multi-label binary classification (use BCEWithLogitsLoss)
#     """
#     def __init__(
#         self,
#         # audio
#         audio_model_name="Kkonjeong/wav2vec2-base-korean",
#         audio_d_out=256,
#         audio_dropout=0.1,
#         freeze_feature_extractor=True,
#         unfreeze_last_n_layers=None,

#         # video
#         video_input_dim=1024,
#         video_d_model=512,
#         video_d_out=256,
#         video_dropout=0.2,

#         # dsr
#         dsr_input_dim=768,
#         dsr_d_model=256,
#         dsr_d_small=64,
#         dsr_dropout=0.5,
#         dsr_alpha=0.2,

#         # tabular
#         nume_input_dim=4,
#         nume_hidden_dim=64,
#         nume_output_dim=16,
#         cate_input_dims=(2, 3, 2, 2),
#         cate_emb_dims=(16, 16, 16, 16),
#         tab_final_output_dim=32,
#         tab_dropout=0.8,

#         # fusion head
#         fusion_hidden=256,
#         fusion_dropout=0.3,
#     ):
#         super().__init__()

#         ### Audio ###
#         self.audio = AudioModel(
#             model_name=audio_model_name,
#             d_out=audio_d_out,
#             dropout=audio_dropout,
#             freeze_feature_extractor=freeze_feature_extractor,
#             unfreeze_last_n_layers=unfreeze_last_n_layers,
#         )

#         ### Video ###
#         self.video = VideoModel(
#             input_dim=video_input_dim,
#             d_model=video_d_model,
#             d_out=video_d_out,
#             dropout=video_dropout,
#         )

#         ### DSR(CT) ###
#         self.dsr = DSRModel(
#             input_dim=dsr_input_dim,
#             d_model=dsr_d_model,
#             d_small=dsr_d_small,
#             dropout=dsr_dropout,
#             alpha=dsr_alpha,
#         )

#         ### Tabular ###
#         self.tabular = TabularModel(
#             nume_input_dim=nume_input_dim,
#             nume_hidden_dim=nume_hidden_dim,
#             nume_output_dim=nume_output_dim,
#             cate_input_dims=list(cate_input_dims),
#             cate_emb_dims=list(cate_emb_dims),
#             final_output_dim=tab_final_output_dim,
#             dropout=tab_dropout,
#         )

#         fusion_in_dim = audio_d_out + video_d_out + dsr_d_small + tab_final_output_dim

#         # (선택) modality별 스케일 차이를 줄이기 위한 LN
#         self.fusion_norm = nn.LayerNorm(fusion_in_dim)

#         # 최종 분류 head (2 logits)
#         self.classifier = nn.Sequential(
#             nn.Dropout(fusion_dropout),
#             nn.Linear(fusion_in_dim, fusion_hidden),
#             nn.GELU(),
#             nn.Dropout(fusion_dropout),
#             nn.Linear(fusion_hidden, 2),  # [보상조음_logit, 과다비성_logit]
#         )

#     def forward(
#         self,
#         audio_input_values: torch.Tensor,
#         audio_attention_mask: torch.Tensor | None,
#         video_x: torch.Tensor,
#         dsr_x: torch.Tensor,
#         cate_tabular: torch.Tensor,
#         nume_tabular: torch.Tensor,
#     ) -> torch.Tensor:
#         """
#         audio_input_values: (B, L)
#         audio_attention_mask: (B, L) or None
#         video_x: (B, 8, 256, 1024)
#         dsr_x: (B, 2K, 1369, 768)
#         cate_tabular: (B, num_cate)
#         nume_tabular: (B, num_nume)

#         return:
#           logits: (B, 2)  (use BCEWithLogitsLoss)
#         """
#         a = self.audio(audio_input_values, audio_attention_mask)  # (B, audio_d_out)
#         v = self.video(video_x)                                   # (B, video_d_out)
#         d = self.dsr(dsr_x)                                       # (B, dsr_d_small)
#         t = self.tabular(cate_tabular, nume_tabular)              # (B, tab_final_output_dim)z

#         fused = torch.cat([a, v, d, t], dim=1)                    # (B, fusion_in_dim)
#         fused = self.fusion_norm(fused)

#         logits = self.classifier(fused)                           # (B, 2)
#         return logits

class Baseline(nn.Module):
    """
    Baseline: modality representations -> concat -> MLP -> 2 logits
    Targets:
      - 보상조음: 0/1
      - 과다비성: 0/1
    => multi-label binary classification (use BCEWithLogitsLoss)
    
    Supports ablation via active_modality:
    - Audio is always required (active_modality['audio'] must be True)
    - Other modalities can be individually disabled
    """
    def __init__(
        self,
        # active modalities (Audio is mandatory)
        active_modality: dict | None = None,  # e.g., {'audio': True, 'video': True, 'dsr': False, 'tabular': False}

        # audio
        audio_model_name="Kkonjeong/wav2vec2-base-korean",
        audio_d_out=256,
        audio_dropout=0.1,
        freeze_feature_extractor=True,
        unfreeze_last_n_layers=None,

        # video
        video_input_dim=1024,
        video_d_model=512,
        video_d_out=256,
        video_dropout=0.2,

        # dsr
        dsr_input_dim=768,
        dsr_d_model=256,
        dsr_d_small=64,
        dsr_dropout=0.5,
        dsr_alpha=0.2,

        # tabular
        nume_input_dim=4,
        nume_hidden_dim=64,
        nume_output_dim=16,
        cate_input_dims=(2, 3, 2, 2),
        cate_emb_dims=(16, 16, 16, 16),
        tab_final_output_dim=32,
        tab_dropout=0.8,

        # fusion head
        fusion_hidden=256,
        fusion_dropout=0.3,
    ):
        super().__init__()

        # Set default active_modality (all True)
        if active_modality is None:
            active_modality = {
                'audio': True,
                'video': True,
                'dsr': True,
                'tabular': True,
            }

        # Audio is mandatory
        if not active_modality.get('audio', False):
            raise ValueError("Audio modality must be active (audio=True)")

        self.active_modality = active_modality

        # Store output dimensions
        self.audio_d_out = audio_d_out
        self.video_d_out = video_d_out
        self.dsr_d_small = dsr_d_small
        self.tab_final_output_dim = tab_final_output_dim

        ### Audio (MANDATORY) ###
        self.audio = AudioModel(
            model_name=audio_model_name,
            d_out=audio_d_out,
            dropout=audio_dropout,
            freeze_feature_extractor=freeze_feature_extractor,
            unfreeze_last_n_layers=unfreeze_last_n_layers,
        )

        ### Video ###
        if active_modality.get('video', False):
            self.video = VideoModel(
                input_dim=video_input_dim,
                d_model=video_d_model,
                d_out=video_d_out,
                dropout=video_dropout,
            )
        else:
            self.video = None

        ### DSR(CT) ###
        if active_modality.get('dsr', False):
            self.dsr = DSRModel(
                input_dim=dsr_input_dim,
                d_model=dsr_d_model,
                d_small=dsr_d_small,
                dropout=dsr_dropout,
                alpha=dsr_alpha,
            )
        else:
            self.dsr = None

        ### Tabular ###
        if active_modality.get('tabular', False):
            self.tabular = TabularModel(
                nume_input_dim=nume_input_dim,
                nume_hidden_dim=nume_hidden_dim,
                nume_output_dim=nume_output_dim,
                cate_input_dims=list(cate_input_dims),
                cate_emb_dims=list(cate_emb_dims),
                final_output_dim=tab_final_output_dim,
                dropout=tab_dropout,
            )
        else:
            self.tabular = None

        # ---- Calculate fusion input dimension based on active modalities ----
        fusion_in_dim = audio_d_out
        if active_modality.get('video', False):
            fusion_in_dim += video_d_out
        if active_modality.get('dsr', False):
            fusion_in_dim += dsr_d_small
        if active_modality.get('tabular', False):
            fusion_in_dim += tab_final_output_dim

        # (선택) modality별 스케일 차이를 줄이기 위한 LN
        self.fusion_norm = nn.LayerNorm(fusion_in_dim)

        # 최종 분류 head (2 logits) - fusion_in_dim에 맞춰 동적으로 생성
        self.classifier = nn.Sequential(
            nn.Dropout(fusion_dropout),
            nn.Linear(fusion_in_dim, fusion_hidden),
            nn.GELU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(fusion_hidden, 2),  # [보상조음_logit, 과다비성_logit]
        )

    def forward(
        self,
        audio_input_values: torch.Tensor,
        audio_attention_mask: torch.Tensor | None = None,
        video_x: torch.Tensor | None = None,
        dsr_x: torch.Tensor | None = None,
        cate_tabular: torch.Tensor | None = None,
        nume_tabular: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        audio_input_values: (B, L) - mandatory
        audio_attention_mask: (B, L) or None
        video_x: (B, 8, 256, 1024) or None (required if video is active)
        dsr_x: (B, 2K, 1369, 768) or None (required if dsr is active)
        cate_tabular: (B, num_cate) or None (required if tabular is active)
        nume_tabular: (B, num_nume) or None (required if tabular is active)

        return:
          logits: (B, 2)  (use BCEWithLogitsLoss)
        """
        fused_representations = []

        # ---- Audio (MANDATORY) ----
        a = self.audio(audio_input_values, audio_attention_mask)  # (B, audio_d_out)
        fused_representations.append(a)

        # ---- Video ----
        if self.active_modality.get('video', False):
            if video_x is None:
                raise ValueError("video_x is required when video modality is active")
            v = self.video(video_x)  # (B, video_d_out)
            fused_representations.append(v)

        # ---- DSR ----
        if self.active_modality.get('dsr', False):
            if dsr_x is None:
                raise ValueError("dsr_x is required when dsr modality is active")
            d = self.dsr(dsr_x)  # (B, dsr_d_small)
            fused_representations.append(d)

        # ---- Tabular ----
        if self.active_modality.get('tabular', False):
            if cate_tabular is None or nume_tabular is None:
                raise ValueError("Both cate_tabular and nume_tabular are required when tabular modality is active")
            t = self.tabular(cate_tabular, nume_tabular)  # (B, tab_final_output_dim)
            fused_representations.append(t)

        # ---- Concatenate active modalities ----
        fused = torch.cat(fused_representations, dim=1)  # (B, fusion_in_dim)
        fused = self.fusion_norm(fused)

        logits = self.classifier(fused)  # (B, 2)
        return logits


# -------------------------
# Example instantiation (pseudo)
# -------------------------
if __name__ == "__main__":
    # All modalities active
    model_all = Baseline(active_modality={'audio': True, 'video': True, 'dsr': True, 'tabular': True})
    
    # Only Audio + Video
    model_av = Baseline(active_modality={'audio': True, 'video': True, 'dsr': False, 'tabular': False})
    
    # Only Audio
    model_a = Baseline(active_modality={'audio': True, 'video': False, 'dsr': False, 'tabular': False})
    pass