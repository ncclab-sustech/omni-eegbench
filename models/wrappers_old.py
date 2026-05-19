

from __future__ import annotations
from typing import Optional, Sequence, Literal, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    from einops import rearrange
except ImportError as e:
    raise ImportError("pip install einops") from e

PoolMode = Literal["cls", "mean"]
HeadType = Literal["linear", "mlp"]


# -----------------------------
# Normalization
# -----------------------------
class IdentityNorm(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class PerTrialZScore(nn.Module):
    """x: [B,C,T], z-score over T for each (B,C)."""
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(var + self.eps)


# -----------------------------
# Unified probe head
# -----------------------------
class ProbeHead(nn.Module):
    """
    Unified head template for fairness.
    - linear: Linear(in_dim -> num_classes)
    - mlp:    Linear(in_dim -> hidden -> num_classes), hidden defaults to in_dim
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        head_type: HeadType = "linear",
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        act: Literal["gelu", "relu", "elu"] = "gelu",
    ):
        super().__init__()
        if act == "gelu":
            activation = nn.GELU()
        elif act == "relu":
            activation = nn.ReLU()
        elif act == "elu":
            activation = nn.ELU()
        else:
            raise ValueError(f"Unknown act={act}")

        if head_type == "linear":
            self.net = nn.Linear(in_dim, out_dim)
        elif head_type == "mlp":
            h = int(hidden_dim) if hidden_dim is not None else int(in_dim)
            layers = [nn.Linear(in_dim, h), activation]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(h, out_dim))
            self.net = nn.Sequential(*layers)
        else:
            raise ValueError(f"Unknown head_type={head_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -----------------------------
# Backbone feature extractors
# -----------------------------
class LaBraMFeatureExtractor(nn.Module):
    """
    Accept x: [B,C,T]
    -> reshape [B,C,P,S] with S=patch_size (default 200)
    -> LaBraM(return_all_tokens=True) -> tokens [B,1+N,D]
    -> pool -> feat [B,D]
    """
    def __init__(
        self,
        backbone: nn.Module,
        pool: PoolMode = "mean",
        patch_size: int = 200,
        norm: Optional[nn.Module] = None,
        input_chans: Optional[Sequence[int]] = None,
        ch_use: Optional[np.ndarray] = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.pool = pool
        self.patch_size = patch_size
        self.norm = norm if norm is not None else IdentityNorm()
        self.input_chans = list(input_chans) if input_chans is not None else None

        # strip LaBraM head if present to avoid accidental usage
        if hasattr(self.backbone, "head"):
            self.backbone.head = nn.Identity()

        # infer embed dim
        self.embed_dim = None
        for attr in ("embed_dim", "num_features", "feature_dim"):
            if hasattr(self.backbone, attr):
                self.embed_dim = int(getattr(self.backbone, attr))
                break
        if self.embed_dim is None:
            raise AttributeError("Cannot infer LaBraM embed dim from backbone attrs.")
        self.ch_use = ch_use

    def _to_4d(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected x [B,C,T], got {tuple(x.shape)}")
        _, _, t = x.shape
        if t % self.patch_size != 0:
            raise ValueError(f"T={t} must be divisible by patch_size={self.patch_size}")
        p = t // self.patch_size
        return rearrange(x, "b c (p s) -> b c p s", p=p, s=self.patch_size)

    def _pool_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"Expected tokens [B,N,D], got {tuple(tokens.shape)}")
        if self.pool == "cls":
            return tokens[:, 0]
        if self.pool == "mean":
            return tokens[:, 1:].mean(dim=1) if tokens.size(1) >= 2 else tokens.mean(dim=1)
        raise ValueError(f"Unknown pool={self.pool}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x4 = self._to_4d(x)
        if self.input_chans is not None:
            input_chans = torch.tensor(self.input_chans, dtype=torch.long, device=x4.device)
        else:
            input_chans = None
        if self.ch_use is not None:
            x4 = x4[:, self.ch_use, :, :]
        tokens = self.backbone(x4, input_chans=input_chans, return_all_tokens=True)
        return self._pool_tokens(tokens)  # [B,D]


class BIOTFeatureExtractor(nn.Module):
    """
    Official-like BIOT feature extractor.

    Recommended usage:
      - pass BIOTClassifier as `model_or_encoder`
      - it will use `.biot` encoder and force `.classifier = Identity()`

    Input:  x [B,C,T]
    Output: feat [B,D]
    """
    def __init__(
        self,
        model_or_encoder: nn.Module,
        norm: Optional[nn.Module] = None,
        channel_select: Optional[Sequence[int]] = None,
        n_channel_offset: int = 0,
        expected_channels: Optional[int] = 18,   
        strict_channels: bool = True,            
    ):
        super().__init__()
        self.model = model_or_encoder
        self.norm = norm if norm is not None else IdentityNorm()
        self.channel_select = list(channel_select) if channel_select is not None else None
        self.n_channel_offset = int(n_channel_offset)
        self.expected_channels = expected_channels
        self.strict_channels = strict_channels

        # --- Prefer official container: BIOTClassifier ---
        # If passed BIOTClassifier, it has `.biot` and `.classifier`
        if hasattr(self.model, "classifier"):
            self.model.classifier = nn.Identity()  # ✅ official classifier head removed

        # encoder must be `.biot` if exists; otherwise assume it's already an encoder
        self.encoder = getattr(self.model, "biot") if hasattr(self.model, "biot") else self.model

        # infer embedding dim
        if hasattr(self.encoder, "channel_tokens") and isinstance(self.encoder.channel_tokens, nn.Embedding):
            self.embed_dim = int(self.encoder.channel_tokens.embedding_dim)
            # if encoder defines channel tokens length, we can trust it
            if self.expected_channels is None:
                self.expected_channels = int(self.encoder.channel_tokens.num_embeddings)
        else:
            self.embed_dim = 256

    def _adapt_channels(self, x: torch.Tensor) -> torch.Tensor:
        """
        Channel selection / re-ordering only.
        NOTE: This does NOT create bipolar montage. For BIOT-18 montage, do it in dataset/transform.
        """
        if x.ndim != 3:
            raise ValueError(f"Expected x [B,C,T], got {tuple(x.shape)}")

        if self.channel_select is not None:
            x = x[:, self.channel_select, :]

        if self.expected_channels is not None:
            c = x.shape[1]
            if c != int(self.expected_channels):
                msg = (
                    f"BIOT input channel mismatch: got C={c}, expected C={self.expected_channels}. "
                    f"With EEG-six-datasets-18-channels.ckpt you should feed BIOT-18 montage ([B,18,T]). "
                    f"Fix by providing correct channel_select (if already montage), or build montage in dataset."
                )
                if self.strict_channels:
                    raise ValueError(msg)
                else:
                    print(f"⚠️ {msg}")

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self._adapt_channels(x)
        return self.encoder(x, n_channel_offset=self.n_channel_offset)  # [B,D]



class CBraModFeatureExtractor(nn.Module):
    """
    Accept x: [B,C,T] -> [B,C,P,200] -> CBraMod(proj_out stripped) -> feats [B,C,P,D]
    -> pool over (C,P) -> feat [B,D]
    """
    def __init__(
        self,
        backbone: nn.Module,
        patch_size: int = 200,
        norm: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.patch_size = patch_size
        self.norm = norm if norm is not None else IdentityNorm()

        if self.patch_size != 200:
            raise ValueError("CBraModFeatureExtractor requires patch_size=200 in this implementation.")

        if hasattr(self.backbone, "proj_out"):
            self.backbone.proj_out = nn.Identity()

        # infer d_model
        self.embed_dim = None
        for attr in ("d_model", "model_dim", "hidden_dim", "dim"):
            if hasattr(self.backbone, attr):
                self.embed_dim = int(getattr(self.backbone, attr))
                break
        if self.embed_dim is None:
            self.embed_dim = 200

    def _to_4d(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected x [B,C,T], got {tuple(x.shape)}")
        _, _, t = x.shape
        if t % self.patch_size != 0:
            raise ValueError(f"T={t} must be divisible by patch_size={self.patch_size}")
        p = t // self.patch_size
        return rearrange(x, "b c (p s) -> b c p s", p=p, s=self.patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x4 = self._to_4d(x)
        feats = self.backbone(x4)  # [B,C,P,D]
        if feats.ndim != 4:
            raise ValueError(f"Expected CBraMod feats [B,C,P,D], got {tuple(feats.shape)}") 
        return feats.mean(dim=(1, 2))  # [B,D]


class BrainOmniFeatureExtractor(nn.Module):
    """
    Input:  x [B, C, T]
    Output: feat [B, D]
    """
    def __init__(
        self,
        model: nn.Module,
        static_pos: torch.Tensor,
        feature_dim:int,
        static_sensor_type: torch.Tensor,
        pool: PoolMode = "mean",
        output_num_tokens: int = None,
        norm: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.model = model
        self.pool = pool
        self.norm = norm if norm is not None else IdentityNorm()
        if static_pos.ndim == 2:
             self.register_buffer("static_pos", static_pos.clone().detach())
        else:
            raise ValueError(f"static_pos shape error: {static_pos.shape}")
            
        if static_sensor_type.ndim == 1:
            self.register_buffer("static_sensor_type", static_sensor_type.clone().detach())
        else:
             raise ValueError(f"static_sensor_type shape error: {static_sensor_type.shape}")
        self.num_channels = static_pos.shape[0]
        self.single_dim = feature_dim
        if output_num_tokens is not None:
            self.final_tokens = output_num_tokens
        else:
            self.final_tokens = self.num_channels

        self.embed_dim = self.final_tokens * self.single_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x: [B, C, T]
        x = self.norm(x)
        B, C, T = x.shape
        # pos: [C, 6] -> [B, C, 6]
        pos = self.static_pos.unsqueeze(0).expand(B, -1, -1)
        # sensor: [C] -> [B, C]
        sensor_type = self.static_sensor_type.unsqueeze(0).expand(B, -1)
        feats = self.model.encode(x, pos, sensor_type)
        # based on brainomni downstream
        if feats.ndim == 4: # [B, C, W, D]
            # 只在 Window 维度 (dim=2) 做平均
            feats = feats.mean(dim=2)  # 变成 [B, C, D]
        feats = feats.contiguous().view(B, -1)
        return feats  # 输出 [B, C*D]
    
# models/wrappers.py


TUEG_PAIRS = [
    ("FP1", "F7"), ("F7", "T3"), ("T3", "T5"), ("T5", "O1"),
    ("FP2", "F8"), ("F8", "T4"), ("T4", "T6"), ("T6", "O2"),
    ("T3", "C3"), ("C3", "CZ"), ("CZ", "C4"), ("C4", "T4"),
    ("FP1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1"),
    ("FP2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2"),
    ("A1", "T3"), ("T4", "A2"),
]
class FembaFeatureExtractor(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        input_freq: int,
        input_chn_names: list, 
        norm: nn.Module = None,
    ):
        super().__init__()
        self.model = model
        self.input_freq = input_freq
        self.target_freq = 200 
        self.norm = norm if norm is not None else nn.Identity()

        # add coord info
        pt_path="data/standard_coords.pt"
        data = torch.load(pt_path, map_location='cpu')
        self.STANDARD_COORDS_NORM = {
        name.upper(): pos[:3].numpy() 
        for name, pos in zip(data["ch_names"], data["pos"])
    }
   
        #[22, Input_Channels]
        self.transform_matrix = self._build_spatial_matrix(input_chn_names)
        
        grid_h = self.model.patch_embed.grid_size[0]
        self.embed_dim = grid_h * self.model.embed_dim


    def _get_coord(self, name):
        key = name.strip().upper()
        
        if key in self.STANDARD_COORDS_NORM:
            return self.STANDARD_COORDS_NORM[key]

        print(f"⚠️ Warning: Channel {name} not found in standard coords, using dummy.")
        return np.array([0.0, 0.0, 1.0])
    
    def _build_spatial_matrix(self, input_names):
        num_inputs = len(input_names)
        num_targets = len(TUEG_PAIRS)
        in_coords = np.stack([self._get_coord(n) for n in input_names])
        in_coords = torch.tensor(in_coords, dtype=torch.float32)
        needed_names = sorted(list(set([p[0] for p in TUEG_PAIRS] + [p[1] for p in TUEG_PAIRS])))
        num_needed = len(needed_names)
        
        needed_coords = np.stack([self._get_coord(n) for n in needed_names])
        needed_coords = torch.tensor(needed_coords, dtype=torch.float32)
        dists = torch.cdist(needed_coords, in_coords)

        epsilon = 1e-6
        is_match = (dists < 1e-4).float()

        weights = 1.0 / (dists + epsilon)
   
        has_match = is_match.sum(dim=1, keepdim=True) > 0
        final_weights = torch.where(has_match, is_match, weights)

        M_interp = final_weights / final_weights.sum(dim=1, keepdim=True)

        M_diff = torch.zeros(num_targets, num_needed)
        for i, (pos_name, neg_name) in enumerate(TUEG_PAIRS):
            idx_pos = needed_names.index(pos_name)
            idx_neg = needed_names.index(neg_name)
            M_diff[i, idx_pos] = 1.0
            M_diff[i, idx_neg] = -1.0
        M_final = torch.matmul(M_diff, M_interp)
        
        return nn.Parameter(M_final, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        
        if self.input_freq != self.target_freq:
            orig_T = x.shape[-1]
            new_T = int(orig_T * self.target_freq / self.input_freq)
            x = F.interpolate(x, size=new_T, mode='linear', align_corners=False)
        # x: [B, 19, T] -> [B, 22, T]
        x = torch.matmul(self.transform_matrix.to(x.device), x)

        tokens = self.model.patch_embed(x) 
        pos_embed = self.model.pos_embed 
        
        # 调整 Pos Embed
        if tokens.shape[1] != pos_embed.shape[1]:
            pos_embed = pos_embed.permute(0, 2, 1)
            pos_embed = F.interpolate(pos_embed, size=tokens.shape[1], mode='linear', align_corners=False)
            pos_embed = pos_embed.permute(0, 2, 1)
        
        tokens = tokens + pos_embed
        
        for blk, ln in zip(self.model.mamba_blocks, self.model.norm_layers):
            tokens = ln(tokens + blk(tokens))
            
        emb = tokens.mean(dim=1)
        return emb
    

NEUROGPT_CHANNELS = [
    'FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'T1', 'T3', 'C3', 
    'CZ', 'C4', 'T4', 'T2', 'T5', 'P3', 'PZ', 'P4', 'T6', 'O1', 'OZ', 'O2'
]

class NeuroGPTFeatureExtractor(nn.Module):
    def __init__(self, encoder, input_chns,norm=None, embed_dim=40, chunk_len=500, stride=500, input_freq=200):
        super().__init__()
        self.encoder = encoder
        self.chunk_len = chunk_len
        self.stride = stride
        self.input_freq = input_freq
        self.target_freq = 250
        self.embed_dim = embed_dim
        # add coord info
        pt_path="data/standard_coords.pt"
        data = torch.load(pt_path, map_location='cpu')
        self.STANDARD_COORDS_NORM = {
        name.upper(): pos[:3].numpy() 
        for name, pos in zip(data["ch_names"], data["pos"])
    }
        self.proj_mat = self._build_spatial_matrix(input_chns)
        self.norm = norm if norm is not None else nn.Identity()


    def _get_coord(self, name):
        key = name.strip().upper()
        
        if key in self.STANDARD_COORDS_NORM:
            return self.STANDARD_COORDS_NORM[key]

        print(f"⚠️ Warning: Channel {name} not found in standard coords, using dummy.")
        return np.array([0.0, 0.0, 1.0])
    
    def _build_spatial_matrix(self, input_names):
            in_coords = np.stack([self._get_coord(n) for n in input_names]) # [N_in, 3]
            in_coords = torch.tensor(in_coords, dtype=torch.float32)

            target_coords = np.stack([self._get_coord(n) for n in NEUROGPT_CHANNELS]) # [22, 3]
            target_coords = torch.tensor(target_coords, dtype=torch.float32)

            dists = torch.cdist(target_coords, in_coords)

            epsilon = 1e-6
            is_match = (dists < 1e-4).float()
            
            weights = 1.0 / (dists + epsilon)
            has_match = is_match.sum(dim=1, keepdim=True) > 0
            final_weights = torch.where(has_match, is_match, weights)

            M_interp = final_weights / final_weights.sum(dim=1, keepdim=True)
            
            return nn.Parameter(M_interp, requires_grad=False)
    
    def forward(self, x):
        # x: [Batch, Channel, Time] 
        x = self.norm(x)
        x = torch.einsum('oc,bct->bot', self.proj_mat.to(x.device), x)
        if self.input_freq != self.target_freq:
            orig_T = x.shape[-1]
            new_T = int(orig_T * self.target_freq / self.input_freq)
            x = F.interpolate(x, size=new_T, mode='linear', align_corners=False) 

        B, C, T = x.shape 
        if T < self.chunk_len:
             x = torch.nn.functional.pad(x, (0, self.chunk_len - T))
        # [B, 22, T] -> [B, 22, N, L]
        chunks = x.unfold(2, self.chunk_len, self.stride)
        # [Batch, Num_Chunks, Channel, Chunk_Len]
        chunks = chunks.permute(0, 2, 1, 3).contiguous()
        # Output: [Batch * Num_Chunks, Features]
        features = self.encoder(chunks)

        # [B*N, Dim]
        features = features.view(B, -1, features.shape[-1]) # [B, N, Dim]
        final_feat = features.mean(dim=1) # [B, Dim]
        
        return final_feat

class LengthAdapter(nn.Module):
    """
    Normalize EEG length to a fixed number of samples.

    Input:  x [B, C, T]
    Output: x [B, C, target_samples]
    """
    def __init__(
        self,
        target_samples: int,
        mode: Literal["center_crop", "left_crop", "right_crop"] = "center_crop",
        pad_mode: Literal["zero", "repeat", "reflect"] = "zero",
    ):
        super().__init__()
        self.target_samples = int(target_samples)
        self.mode = mode
        self.pad_mode = pad_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected x [B,C,T], got {x.shape}")

        B, C, T = x.shape
        tgt = self.target_samples

        # ---------- Crop ----------
        if T > tgt:
            if self.mode == "center_crop":
                start = (T - tgt) // 2
            elif self.mode == "left_crop":
                start = 0
            elif self.mode == "right_crop":
                start = T - tgt
            else:
                raise ValueError(f"Unknown crop mode {self.mode}")

            return x[:, :, start:start + tgt]

        # ---------- Pad ----------
        if T < tgt:
            pad_len = tgt - T

            if self.pad_mode == "zero":
                pad = torch.zeros(B, C, pad_len, device=x.device, dtype=x.dtype)
                return torch.cat([x, pad], dim=-1)

            if self.pad_mode == "repeat":
                reps = (pad_len + T - 1) // T
                x_rep = x.repeat(1, 1, reps)
                return x_rep[:, :, :tgt]

            if self.pad_mode == "reflect":
                pad = F.pad(x, (0, pad_len), mode="reflect")
                return pad

            raise ValueError(f"Unknown pad_mode {self.pad_mode}")

        # ---------- Exact ----------
        return x


class EEGMambaFeatureExtractor(nn.Module):
    """
    EEGMamba feature extractor with length normalization.
    """
    def __init__(
        self,
        model: nn.Module,
        patch_size: int,
        target_samples: int = 6000,
        norm: Optional[nn.Module] = None,
        crop_mode: Literal["center_crop", "left_crop", "right_crop"] = "center_crop",
        pad_mode: Literal["zero", "repeat", "reflect"] = "zero",
    ):
        super().__init__()
        self.model = model
        self.patch_size = patch_size
        self.norm = norm if norm is not None else IdentityNorm()

        self.length_adapter = LengthAdapter(
            target_samples=target_samples,
            mode=crop_mode,
            pad_mode=pad_mode,
        )

        if hasattr(self.model, "proj_out"):
            self.embed_dim = self.model.proj_out[0].out_features
        else:
            raise AttributeError("EEGMamba must have proj_out")

    def _to_4d(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        L = T // self.patch_size
        return rearrange(x, "b c (l s) -> b c l s", l=L, s=self.patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T_any]
        x = self.norm(x)
        x = self.length_adapter(x)       # ✅ ALWAYS [B,C,6000]
        x4 = self._to_4d(x)              # → [B,C,30,200]

        # 🚨 Mamba kernels are NOT AMP-safe
        with torch.amp.autocast("cuda", enabled=False):
            feats = self.model(x4)  # [B,C,L,D]

        return feats.mean(dim=(1, 2))    # [B,D]




class NeuroLMFeatureExtractor(nn.Module):
    """
    NeuroLM EEG feature extractor.

    Uses tokenizer + embedding layers only.
    DOES NOT call GPT2 language model.
    """
    def __init__(
        self,
        model: nn.Module,
        pool: PoolMode = "mean",
        norm: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.model = model
        self.pool = pool
        self.norm = norm if norm is not None else IdentityNorm()

        # NeuroLM's tokenizer was pretrained with a fixed patch_size (timepoints) per channel-token.
        # Many downstream datasets use different window lengths, which will break the tokenizer's
        # TemporalConv -> Linear(in_features=400) projection. Normalize to the expected length.
        target_samples = getattr(self.model.tokenizer, "patch_size", None)
        if target_samples is None:
            target_samples = 200
        self.length_adapter = LengthAdapter(
            target_samples=int(target_samples),
            mode="center_crop",
            pad_mode="zero",
        )

        # Freeze tokenizer & GPT by default (pretraining behavior)
        self.model.tokenizer.eval()
        for p in self.model.tokenizer.parameters():
            p.requires_grad = False

        # Embedding dimension
        self.embed_dim = self.model.GPT2.config.n_embd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, T]
        """
        x = self.norm(x)
        x = self.length_adapter(x)   # [B, C, T_fixed]

        B, C, T = x.shape
        device = x.device

        # ---- build required NeuroLM inputs ----
        input_chans = torch.arange(C, device=device).unsqueeze(0).repeat(B, 1)
        input_time = torch.zeros((B, C), dtype=torch.long, device=device)
        # NeuroLM tokenizer/transformer attends over channel-tokens (sequence length == C after patch_embed),
        # so the attention mask must be built over C (not raw time samples T).
        input_mask = torch.ones((B, C), dtype=torch.long, device=device)

        # ---- tokenizer ----
        tokens = self.model.tokenizer(
            x,
            input_chans=input_chans,
            input_times=input_time,
            mask=input_mask.unsqueeze(1).repeat(1, C, 1).unsqueeze(1),
            return_all_tokens=True,
        )

        tokens = self.model.encode_transform_layer(tokens)
        tokens = tokens + self.model.pos_embed(input_chans)

        # tokens: [B, N, D]
        if self.pool == "cls":
            return tokens[:, 0]
        else:
            return tokens.mean(dim=1)

class BENDRFeatureExtractor(nn.Module):
    """
    Feature extractor for BENDRClassification.

    Uses official BENDR forward semantics:
    - ConvEncoder
    - Contextualizer
    - last timestep representation
    """
    def __init__(
        self,
        model: nn.Module,
        norm: Optional[nn.Module] = None
    ):
        super().__init__()
        self.model = model
        self.norm = norm if norm is not None else IdentityNorm()
        self.input_freq = 200
        self.target_freq = 256
        # Remove projection / classifier heads if present
        for attr in ("projection_mlp", "extended_classifier", "classifier"):
            if hasattr(self.model, attr):
                setattr(self.model, attr, nn.Identity())

        # Embed dim is encoder_h
        self.embed_dim = int(self.model.encoder_h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, T]
        """
        x = self.norm(x)

        if self.input_freq != self.target_freq:
            orig_T = x.shape[-1]
            new_T = int(orig_T * self.target_freq / self.input_freq)
            x = F.interpolate(x, size=new_T, mode='linear', align_corners=False) 

        # Replicate BENDRClassification.features_forward
        encoded = self.model.encoder(x)           # [B, D, T′]
        context = self.model.contextualizer(encoded)

        # last timestep is the representation
        feat = context[:, :, -1]                  # [B, D]
        return feat


class BrantPretrainFeatureExtractor(nn.Module):
    """
    Feature extractor for Brant (TimeEncoder + optional ChannelEncoder).

    Input:  x [B, C, T]
    Output: feat [B, D]
    """
    def __init__(
        self,
        model: nn.Module,
        seg_len: int,          # L
        seq_len: int,          # S
        band_num: int,
        use_power: bool = False,
        norm: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.model = model
        self.seg_len = int(seg_len)
        self.seq_len = int(seq_len)
        self.band_num = int(band_num)
        self.use_power = bool(use_power)
        self.norm = norm if norm is not None else IdentityNorm()
        self.input_freq = 200
        self.target_freq = 250

        self.target_samples = self.seg_len * self.seq_len

        # Brant embed dim is TimeEncoder d_model.
        # The upstream TimeEncoder implementation doesn't store `d_model` as an attribute,
        # but it is the last dimension of the learnable positional encoding.
        try:
            self.embed_dim = int(self.model.time.input_embedding.positional_encoding.shape[-1])
        except Exception as e:
            raise AttributeError("Cannot infer Brant embed_dim from model.time.input_embedding.positional_encoding") from e

        # Length normalization (STRICT)
        self.length_adapter = LengthAdapter(
            target_samples=self.target_samples,
            mode="center_crop",
            pad_mode="zero",
        )

    def _to_4d(self, x: torch.Tensor) -> torch.Tensor:
        """
        [B, C, T] → [B, C, S, L]
        """
        B, C, T = x.shape
        if T != self.target_samples:
            raise ValueError(
                f"Brant requires T={self.target_samples}, got {T}"
            )
        return x.view(B, C, self.seq_len, self.seg_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, T]
        """
        x = self.norm(x)
        

        if self.input_freq != self.target_freq:
            orig_T = x.shape[-1]
            new_T = int(orig_T * self.target_freq / self.input_freq)
            x = F.interpolate(x, size=new_T, mode='linear', align_corners=False) 
            
        x = self.length_adapter(x)
        data = self._to_4d(x)

        # Power features
        if self.use_power:
            # placeholder: real power features should be precomputed
            power = torch.zeros(
                data.shape[0],
                data.shape[1],
                data.shape[2],
                self.band_num,
                device=data.device,
            )
        else:
            power = None

        # Delegate pooling semantics to Brant
        feat = self.model(data, power=power, use_power=self.use_power)
        return feat  # [B, D]

    
# -----------------------------
# Unified downstream model
# -----------------------------
class DownstreamModel(nn.Module):
    """
    feature_extractor: x -> feat [B,D]
    probe_head:        feat -> logits/regression outputs
    """
    def __init__(self, feature_extractor: nn.Module, probe_head: nn.Module):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.probe_head = probe_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.feature_extractor(x)
        return self.probe_head(feat)


# -----------------------------
# Training mode helpers
# -----------------------------
def set_linear_probe(model: DownstreamModel) -> None:
    """
    Freeze feature extractor; train only head.
    """
    model.feature_extractor.requires_grad_(False)
    model.probe_head.requires_grad_(True)
    model.feature_extractor.eval()  # optional but recommended for deterministic behavior


def set_full_finetune(model: DownstreamModel) -> None:
    """
    Train feature extractor + head.
    """
    model.feature_extractor.requires_grad_(True)
    model.probe_head.requires_grad_(True)


def set_partial_finetune_last_n_transformer_blocks(
    model: DownstreamModel,
    n_last_blocks: int,
    blocks_attr_candidates: Sequence[str] = ("blocks", "transformer", "encoder"),
) -> None:
    """
    Optional: fine-tune only last N blocks (if your backbone exposes blocks).
    This is best-effort; use only if you know the backbone structure.
    """
    # default: freeze all
    model.feature_extractor.requires_grad_(False)
    model.probe_head.requires_grad_(True)

    fe = model.feature_extractor
    backbone = getattr(fe, "backbone", None) or getattr(fe, "model", None) or fe

    blocks = None
    for attr in blocks_attr_candidates:
        if hasattr(backbone, attr):
            blocks = getattr(backbone, attr)
            break

    if blocks is None:
        raise AttributeError("Cannot find transformer blocks on backbone; use full/linear modes instead.")

    # blocks could be ModuleList or a module containing blocks
    if isinstance(blocks, nn.ModuleList):
        target = blocks[-n_last_blocks:]
        for m in target:
            m.requires_grad_(True)
    else:
        # if blocks is a module, you must adapt this to your backbone
        blocks.requires_grad_(True)


