

from __future__ import annotations
from typing import Optional, Sequence, Literal, Union, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from omegaconf import OmegaConf

try:
    from einops import rearrange
except ImportError as e:
    raise ImportError("pip install einops") from e

PoolMode = Literal["cls", "mean"]
HeadType = Literal["linear", "mlp"]


def resample_along_time(
    x: torch.Tensor,
    input_freq: int,
    target_freq: int,
    *,
    mode: str = "linear",
    align_corners: bool = False,
) -> torch.Tensor:
    """Unified time-domain resampling for EEG tensors shaped [B, C, T]."""
    if x.ndim != 3:
        raise ValueError(f"Expected x [B,C,T], got {tuple(x.shape)}")
    if int(input_freq) <= 0 or int(target_freq) <= 0:
        raise ValueError(f"input_freq and target_freq must be positive, got {input_freq}, {target_freq}")
    if int(input_freq) == int(target_freq):
        return x

    orig_t = x.shape[-1]
    new_t = max(1, int(round(orig_t * float(target_freq) / float(input_freq))))
    return F.interpolate(x, size=new_t, mode=mode, align_corners=align_corners)


def _build_standard_coords_norm(pt_path: str = "data/standard_coords.pt") -> Dict[str, np.ndarray]:
    data = torch.load(pt_path, map_location="cpu")
    return {
        name.upper(): pos[:3].numpy()
        for name, pos in zip(data["ch_names"], data["pos"])
    }


def _normalize_eeg_coord_name(name: str) -> str:
    key = str(name).strip().upper()
    if key.startswith("EEG "):
        key = key[4:].strip()
    for suffix in ("-REF", "-LE", "-AR", "-AVG", "-A1", "-A2", "-M1", "-M2"):
        if key.endswith(suffix):
            key = key[: -len(suffix)].strip()
    return key


def _coord_or_dummy(
    name: str,
    coords_dict: Dict[str, np.ndarray],
    *,
    strict: bool = False,
) -> tuple[np.ndarray, bool]:
    key = _normalize_eeg_coord_name(name)
    if key in coords_dict:
        return coords_dict[key], True
    if strict:
        raise KeyError(f"Channel `{name}` not found in standard coords.")
    print(f"⚠️ Warning: Channel {name} not found in standard coords, using dummy.")
    return np.array([0.0, 0.0, 1.0]), False


def build_coord_projection_matrix(
    input_names: Sequence[str],
    target_names: Sequence[str],
    coords_dict: Dict[str, np.ndarray],
    *,
    strict: bool = False,
) -> tuple[nn.Parameter, List[str], List[str]]:
    """
    Build a fixed projection matrix from arbitrary input channel names to target channel names
    using exact-match-or-inverse-distance interpolation over standard coordinates.
    """
    input_coords = []
    missing_inputs: List[str] = []
    for name in input_names:
        coord, found = _coord_or_dummy(name, coords_dict, strict=strict)
        input_coords.append(coord)
        if not found:
            missing_inputs.append(name)

    target_coords = []
    missing_targets: List[str] = []
    for name in target_names:
        coord, found = _coord_or_dummy(name, coords_dict, strict=strict)
        target_coords.append(coord)
        if not found:
            missing_targets.append(name)

    in_coords = torch.tensor(np.stack(input_coords), dtype=torch.float32)
    tgt_coords = torch.tensor(np.stack(target_coords), dtype=torch.float32)

    dists = torch.cdist(tgt_coords, in_coords)
    epsilon = 1e-6
    is_match = (dists < 1e-4).float()
    weights = 1.0 / (dists + epsilon)
    has_match = is_match.sum(dim=1, keepdim=True) > 0
    final_weights = torch.where(has_match, is_match, weights)
    proj = final_weights / final_weights.sum(dim=1, keepdim=True)

    return nn.Parameter(proj, requires_grad=False), missing_inputs, missing_targets


# -----------------------------
# Normalization
# -----------------------------
class IdentityNorm(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class PerTrialMinMax(nn.Module):
    """x: [B,C,T], scale each trial/channel to [-1, 1]."""
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_min = x.amin(dim=-1, keepdim=True)
        x_max = x.amax(dim=-1, keepdim=True)
        denom = (x_max - x_min).clamp_min(self.eps)
        return 2.0 * (x - x_min) / denom - 1.0


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
            if int(np.asarray(self.ch_use).sum()) <= 0:
                print(
                    "⚠️ Warning: LaBraM found no channels matching the standard channel vocabulary. "
                    "Falling back to the original channel order without channel-name filtering."
                )
                input_chans = None
            else:
                x4 = x4[:, self.ch_use, :, :]
        tokens = self.backbone(x4, input_chans=input_chans, return_all_tokens=True)
        return self._pool_tokens(tokens)  # [B,D]

BIOT_18_PAIRS = [
    ("FP1", "F7"),
    ("F7", "T7"),
    ("T7", "P7"),
    ("P7", "O1"),
    ("FP2", "F8"),
    ("F8", "T8"),
    ("T8", "P8"),
    ("P8", "O2"),
    ("FP1", "F3"),
    ("F3", "C3"),
    ("C3", "P3"),
    ("P3", "O1"),
    ("FP2", "F4"),
    ("F4", "C4"),
    ("C4", "P4"),
    ("P4", "O2"),
    ("C3", "A2"),
    ("C4", "A1"),
]

BIOT_18_NAMES = [f"{a}-{b}" for a, b in BIOT_18_PAIRS]

# =========================
# 常见别名统一
# =========================
ALIASES = {
    # 老式命名 -> 新式命名
    "T3": "T7",
    "T4": "T8",
    "T5": "P7",
    "T6": "P8",

    # 额极大小写 / 写法
    "FPZ": "FPZ",
    "FP1": "FP1",
    "FP2": "FP2",

    # 耳参考常见写法
    "M1": "A1",
    "M2": "A2",
    "A1": "A1",
    "A2": "A2",

    # 少数数据集会写成 OZ / CZ / PZ 之类，保留原样即可
}



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
        ch_names: Optional[list[str]] = BIOT_18_NAMES,
        require_referential_names: bool = True,
    ):
        super().__init__()
        self.model = model_or_encoder
        self.norm = norm if norm is not None else IdentityNorm()
        self.channel_select = list(channel_select) if channel_select is not None else None
        self.n_channel_offset = int(n_channel_offset)
        self.expected_channels = expected_channels
        self.strict_channels = strict_channels
        self.require_referential_names = bool(require_referential_names)

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
        self.ch_names = ch_names
        self._warned_missing_montage = False
        self._warned_bipolar_names = False
        x, _, missing_pair, missing_node = self.biot_prepare_input(torch.rand(1, len(self.ch_names), 1000), self.ch_names)
        self.missing_pairs = list(missing_pair)
        self.missing_nodes = list(missing_node)
        self.effective_pairs = [p for p in BIOT_18_NAMES if p not in set(self.missing_pairs)]
        self.benchmark_metadata = {
            "implementation": "official_biot18" if not self.missing_pairs else "biot18_zero_fill_adapter",
            "variant": "biot" if not self.missing_pairs else "biot_zero_fill_adapter",
            "target_sampling_rate": 200,
            "target_channels": BIOT_18_NAMES,
            "channel_policy": "construct BIOT-18 bipolar montage; missing pairs are zero-filled only in adapter mode",
            "missing_pairs": self.missing_pairs,
            "missing_nodes": self.missing_nodes,
            "effective_pairs": self.effective_pairs,
            "official_alignment": "strict" if not self.missing_pairs else "adapter",
        }
        if len(missing_pair) > 0:
            msg = (
                "BIOT requires referential single-electrode names that can be mapped to the BIOT-18 bipolar montage. "
                f"Missing pairs/nodes: {missing_pair, missing_node}."
            )
            if self.strict_channels:
                raise ValueError(msg)
            print(f"⚠️ Warning: {msg}")
            self._warned_missing_montage = True

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
        x, _, missing_pairs, missing_nodes = self.biot_prepare_input(x, self.ch_names)
        if missing_pairs:
            msg = (
                "BIOT expected referential single-electrode names required; "
                f"cannot build BIOT-18 montage due to missing pairs/nodes: {missing_pairs, missing_nodes}."
            )
            if self.strict_channels:
                raise ValueError(msg)
            if not getattr(self, "_warned_missing_montage", False):
                print(f"⚠️ Warning: {msg}")
                self._warned_missing_montage = True
        x = self.norm(x)
        x = self._adapt_channels(x)
        return self.encoder(x, n_channel_offset=self.n_channel_offset)  # [B,D]

    def _norm_name(self, name: str) -> str:
        name = _normalize_eeg_coord_name(name)
        return ALIASES.get(name, name)

    def biot_prepare_input(self, 
        data: torch.Tensor,
        channel_names: list[str],
        *,
        pair_mode: str = "subtract",
        missing_policy: str = "zero",
        normalize: str | None = None,
    ):
        """
        将 [B, C, T] + 单电极 channel_names 转成 BIOT 可接受的 [B, 18, T]

        Args:
            data:
                torch.Tensor, shape [B, C, T]
            channel_names:
                长度为 C 的通道名列表，通常是单电极名，如:
                ["FP1", "FP2", "F7", "F3", ...]
            pair_mode:
                构造双导联的方式：
                - "subtract": x(A-B) = x(A) - x(B)   （推荐）
                - "reverse_subtract": x(B-A) = x(B) - x(A)
            missing_policy:
                如果 pair 中某个电极缺失：
                - "zero": 该双导联整条补 0
                - "ignore": 同样补 0，但会在 missing_pairs 中记录
            normalize:
                是否做额外归一化：
                - None: 不做
                - "sample_zscore": 对每个样本整体做 z-score
                - "sample_minmax": 对每个样本整体缩放到 [-1, 1]

        Returns:
            x_biot: torch.Tensor, [B, 18, T]
            mapped_names: list[str], 长度 18，对应 BIOT 双导联名称
            missing_pairs: list[str], 无法构造的双导联
            missing_nodes: list[str], 缺失的单电极名
        """
        assert data.ndim == 3, f"expect [B, C, T], got {tuple(data.shape)}"
        B, C, T = data.shape
        assert len(channel_names) == C, "channel_names 长度必须等于 data.shape[1]"
        normed_names = [self._norm_name(ch) for ch in channel_names]

        if self.require_referential_names:
            bipolar_like = [name for name in normed_names if "-" in name]
            if bipolar_like:
                msg = (
                    "BIOT expects referential single-electrode channel names, "
                    f"but found bipolar-like names: {bipolar_like[:5]}"
                )
                if self.strict_channels:
                    raise ValueError(msg)
                if not getattr(self, "_warned_bipolar_names", False):
                    print(f"⚠️ Warning: {msg}")
                    self._warned_bipolar_names = True

        device = data.device
        dtype = data.dtype

        # 1) 通道名规范化
        src_map = {ch: i for i, ch in enumerate(normed_names)}

        # 2) 构造 BIOT 18 个双导联
        x18 = torch.zeros((B, len(BIOT_18_PAIRS), T), device=device, dtype=dtype)
        missing_pairs = []
        missing_nodes = set()

        for i, (a, b) in enumerate(BIOT_18_PAIRS):
            has_a = a in src_map
            has_b = b in src_map

            if has_a and has_b:
                xa = data[:, src_map[a], :]
                xb = data[:, src_map[b], :]
                if pair_mode == "subtract":
                    x18[:, i, :] = xa - xb
                else:
                    x18[:, i, :] = xb - xa
            else:
                missing_pairs.append(f"{a}-{b}")
                if not has_a:
                    missing_nodes.add(a)
                if not has_b:
                    missing_nodes.add(b)

                if missing_policy in {"zero", "ignore"}:
                    # 保持为 0
                    pass


        return x18, BIOT_18_NAMES, missing_pairs, sorted(missing_nodes)
class CBraModFeatureExtractor(nn.Module):
    """
    Accept x: [B,C,T]
    -> [B,C,P,200]
    -> CBraMod(proj_out stripped) -> feats [B,C,P,D] or compatible output
    -> pool over (C,P) -> feat [B,D]
    """
    def __init__(
        self,
        backbone: nn.Module,
        patch_size: int = 200,
        norm: Optional[nn.Module] = None,
        input_freq: int = 200,
        target_freq: int = 200,
        interpolation_mode: str = "linear",
        interpolation_align_corners: bool = False,
        debug_input_range: bool = True,
    ):
        super().__init__()
        self.backbone = backbone
        self.patch_size = patch_size
        self.norm = norm if norm is not None else IdentityNorm()
        self.input_freq = int(input_freq)
        self.target_freq = int(target_freq)
        self.interpolation_mode = interpolation_mode
        self.interpolation_align_corners = bool(interpolation_align_corners)
        self.debug_input_range = bool(debug_input_range)
        self._input_range_logged = False

        if self.patch_size != 200:
            raise ValueError("CBraModFeatureExtractor requires patch_size=200 in this implementation.")

        if hasattr(self.backbone, "proj_out"):
            self.backbone.proj_out = nn.Identity()

        # infer d_model
        self.embed_dim = None
        for attr in ("d_model", "model_dim", "hidden_dim", "dim", "embed_dim", "num_features", "feature_dim"):
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

    def _range_text(self, name: str, x: torch.Tensor) -> str:
        with torch.no_grad():
            y = x.detach().float()
            finite = torch.isfinite(y)
            finite_ratio = finite.float().mean().item() if y.numel() else 1.0
            if finite.any():
                yf = y[finite]
                return (
                    f"{name}: shape={tuple(x.shape)} min={yf.min().item():.6g} "
                    f"max={yf.max().item():.6g} mean={yf.mean().item():.6g} "
                    f"std={yf.std(unbiased=False).item():.6g} finite={finite_ratio:.6f}"
                )
            return f"{name}: shape={tuple(x.shape)} no finite values finite={finite_ratio:.6f}"

    def _log_input_range_once(self, raw: torch.Tensor, normalized: torch.Tensor, resampled: torch.Tensor) -> None:
        if (not self.debug_input_range) or self._input_range_logged:
            return
        self._input_range_logged = True
        print("[CBraModInputRange] " + self._range_text("raw", raw))
        print("[CBraModInputRange] " + self._range_text("after_FixedScaleTo01mV", normalized))
        if resampled.data_ptr() != normalized.data_ptr() or tuple(resampled.shape) != tuple(normalized.shape):
            print("[CBraModInputRange] " + self._range_text("after_resample", resampled))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw_x = x
        x = self.norm(x)
        norm_x = x
        if self.input_freq != self.target_freq:
            x = resample_along_time(
                x,
                self.input_freq,
                self.target_freq,
                mode=self.interpolation_mode,
                align_corners=self.interpolation_align_corners,
            )
        self._log_input_range_once(raw_x, norm_x, x)
        x4 = self._to_4d(x)
        feats = self.backbone(x4)  # expect [B,C,P,D]
        if feats.ndim != 4:
            raise ValueError(f"Expected CBraMod feats [B,C,P,D], got {tuple(feats.shape)}")

        return feats.mean(dim=(1, 2))  # [B,D]
# class CBraModFeatureExtractor(nn.Module):
#     """
#     Accept x: [B,C,T] -> [B,C,P,200] -> CBraMod(proj_out stripped) -> feats [B,C,P,D]
#     -> pool over (C,P) -> feat [B,D]
#     """
#     def __init__(
#         self,
#         backbone: nn.Module,
#         patch_size: int = 200,
#         norm: Optional[nn.Module] = None,
#     ):
#         super().__init__()
#         self.backbone = backbone
#         self.patch_size = patch_size
#         self.norm = norm if norm is not None else IdentityNorm()

#         if self.patch_size != 200:
#             raise ValueError("CBraModFeatureExtractor requires patch_size=200 in this implementation.")

#         if hasattr(self.backbone, "proj_out"):
#             self.backbone.proj_out = nn.Identity()

#         # infer d_model
#         self.embed_dim = None
#         for attr in ("d_model", "model_dim", "hidden_dim", "dim"):
#             if hasattr(self.backbone, attr):
#                 self.embed_dim = int(getattr(self.backbone, attr))
#                 break
#         if self.embed_dim is None:
#             self.embed_dim = 200

#     def _to_4d(self, x: torch.Tensor) -> torch.Tensor:
#         if x.ndim != 3:
#             raise ValueError(f"Expected x [B,C,T], got {tuple(x.shape)}")
#         _, _, t = x.shape
#         if t % self.patch_size != 0:
#             raise ValueError(f"T={t} must be divisible by patch_size={self.patch_size}")
#         p = t // self.patch_size
#         return rearrange(x, "b c (p s) -> b c p s", p=p, s=self.patch_size)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.norm(x)
#         x4 = self._to_4d(x)
#         feats = self.backbone(x4)  # [B,C,P,D]
#         if feats.ndim != 4:
#             raise ValueError(f"Expected CBraMod feats [B,C,P,D], got {tuple(feats.shape)}") 
#         return feats.mean(dim=(1, 2))  # [B,D]


class REVEFeatureExtractor(nn.Module):
    """Thin wrapper around the official REVE Hugging Face interface.

    Official example:
      pos_bank = AutoModel.from_pretrained("brain-bzh/reve-positions", trust_remote_code=True)
      model = AutoModel.from_pretrained("brain-bzh/reve-base", trust_remote_code=True)
      output = model(eeg_data, positions)

    Input: x [B, C, T]
    Output: feat [B, D]
    """

    def __init__(
        self,
        backbone: nn.Module,
        electrode_names: Sequence[str],
        *,
        pos_bank: Optional[nn.Module] = None,
        norm: Optional[nn.Module] = None,
        input_freq: int = 200,
        target_freq: int = 200,
        interpolation_mode: str = "linear",
        interpolation_align_corners: bool = False,
        use_official_positions: bool = True,
    ):
        super().__init__()
        self.backbone = backbone
        self.pos_bank = pos_bank
        self.norm = norm if norm is not None else IdentityNorm()
        self.input_freq = int(input_freq)
        self.target_freq = int(target_freq)
        self.interpolation_mode = interpolation_mode
        self.interpolation_align_corners = bool(interpolation_align_corners)
        self.electrode_names = [str(name) for name in electrode_names]
        self.use_official_positions = bool(use_official_positions)
        self.register_buffer('_positions_cache', self._build_positions(), persistent=False)

        self.embed_dim = None
        for attr in ("hidden_size", "embed_dim", "d_model", "dim", "feature_dim"):
            if hasattr(self.backbone, attr):
                self.embed_dim = int(getattr(self.backbone, attr))
                break
        cfg = getattr(self.backbone, 'config', None)
        if self.embed_dim is None and cfg is not None:
            for attr in ('hidden_size', 'd_model', 'dim', 'n_embd'):
                if hasattr(cfg, attr):
                    self.embed_dim = int(getattr(cfg, attr))
                    break
        if self.embed_dim is None:
            self.embed_dim = 768

    def _official_position_name_set(self) -> set[str]:
        if self.pos_bank is None:
            return set()
        if hasattr(self.pos_bank, "mapping") and isinstance(self.pos_bank.mapping, dict):
            return set(str(name) for name in self.pos_bank.mapping.keys())
        if hasattr(self.pos_bank, "get_all_positions"):
            return set(str(name) for name in self.pos_bank.get_all_positions())
        names = getattr(getattr(self.pos_bank, "config", None), "position_names", None)
        return set(str(name) for name in names) if names is not None else set()

    def _resolve_official_electrode_names(self) -> list[str]:
        official_names = self._official_position_name_set()
        if not official_names:
            return list(self.electrode_names)
        official_by_upper = {name.upper(): name for name in official_names}

        resolved: list[str] = []
        missing: list[str] = []
        for name in self.electrode_names:
            candidates = [str(name), _normalize_eeg_coord_name(name)]
            match = next((cand for cand in candidates if cand in official_names), None)
            if match is None:
                match = next((official_by_upper[cand.upper()] for cand in candidates if cand.upper() in official_by_upper), None)
            if match is None:
                missing.append(str(name))
            else:
                resolved.append(match)

        if missing:
            preview = ", ".join(missing[:20])
            more = "" if len(missing) <= 20 else f", ... (+{len(missing) - 20} more)"
            raise KeyError(
                "REVE official position bank is missing positions for "
                f"{len(missing)}/{len(self.electrode_names)} channels: {preview}{more}"
            )
        return resolved

    def _build_positions(self) -> torch.Tensor:
        if self.use_official_positions and self.pos_bank is not None:
            resolved_names = self._resolve_official_electrode_names()
            with torch.no_grad():
                pos = self.pos_bank(resolved_names)
            if isinstance(pos, dict):
                for key in ('positions', 'coords', 'x'):
                    if key in pos and torch.is_tensor(pos[key]):
                        pos = pos[key]
                        break
            elif hasattr(pos, 'positions') and torch.is_tensor(pos.positions):
                pos = pos.positions
            elif hasattr(pos, 'last_hidden_state') and torch.is_tensor(pos.last_hidden_state):
                pos = pos.last_hidden_state
            if not torch.is_tensor(pos):
                raise TypeError(f'REVE position bank returned unsupported type: {type(pos)}')
            pos = pos.detach().float().cpu()
            if pos.ndim == 3 and pos.shape[0] == 1:
                pos = pos[0]
            if pos.ndim != 2 or pos.shape[-1] < 3:
                raise ValueError(f'Unexpected REVE positions shape: {tuple(pos.shape)}')
            if pos.shape[0] != len(self.electrode_names):
                raise ValueError(
                    "REVE official position bank returned "
                    f"{pos.shape[0]} positions for {len(self.electrode_names)} channels. "
                    "Check channel names and ordering."
                )
            return pos[:, :3].contiguous()

        coords = _build_standard_coords_norm()
        pos = []
        for name in self.electrode_names:
            coord, _ = _coord_or_dummy(name, coords, strict=False)
            pos.append(coord[:3])
        return torch.tensor(np.stack(pos), dtype=torch.float32)

    def _pool_output(self, output) -> torch.Tensor:
        if torch.is_tensor(output):
            feat = output
        elif isinstance(output, dict):
            for key in ('pooler_output', 'last_hidden_state', 'hidden_states', 'x'):
                if key in output and torch.is_tensor(output[key]):
                    feat = output[key]
                    break
            else:
                raise TypeError(f'Unsupported REVE output dict keys: {list(output.keys())}')
        elif hasattr(output, 'pooler_output') and torch.is_tensor(output.pooler_output):
            feat = output.pooler_output
        elif hasattr(output, 'last_hidden_state') and torch.is_tensor(output.last_hidden_state):
            feat = output.last_hidden_state
        else:
            raise TypeError(f'Unsupported REVE output type: {type(output)}')

        if feat.ndim == 3:
            feat = feat.mean(dim=1)
        elif feat.ndim > 3:
            feat = feat.flatten(1)
        return feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        if self.input_freq != self.target_freq:
            x = resample_along_time(
                x,
                self.input_freq,
                self.target_freq,
                mode=self.interpolation_mode,
                align_corners=self.interpolation_align_corners,
            )
        positions = self._positions_cache.to(device=x.device, dtype=x.dtype).unsqueeze(0).expand(x.size(0), -1, -1)
        output = self.backbone(x, positions)
        return self._pool_output(output)

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
        input_freq: int = 256,
        target_freq: int = 256,
        interpolation_mode: str = "linear",
        interpolation_align_corners: bool = False,
    ):
        super().__init__()
        self.model = model
        self.pool = pool
        self.norm = norm if norm is not None else IdentityNorm()
        self.input_freq = int(input_freq)
        self.target_freq = int(target_freq)
        self.interpolation_mode = interpolation_mode
        self.interpolation_align_corners = bool(interpolation_align_corners)
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
        if self.input_freq != self.target_freq:
            x = resample_along_time(
                x,
                self.input_freq,
                self.target_freq,
                mode=self.interpolation_mode,
                align_corners=self.interpolation_align_corners,
            )
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

    def get_parameter_groups(
        self,
        lr: float,
        weight_decay: float,
        *,
        probe_head: Optional[nn.Module] = None,
        cfg=None,
    ):
        backbone_lr = float(
            OmegaConf.select(cfg, "model.brainomni.backbone_lr")
            if cfg is not None and OmegaConf.select(cfg, "model.brainomni.backbone_lr") is not None
            else min(float(lr), 3e-5)
        )
        head_lr = float(
            OmegaConf.select(cfg, "model.brainomni.head_lr")
            if cfg is not None and OmegaConf.select(cfg, "model.brainomni.head_lr") is not None
            else float(lr)
        )
        backbone_wd = float(
            OmegaConf.select(cfg, "model.brainomni.weight_decay")
            if cfg is not None and OmegaConf.select(cfg, "model.brainomni.weight_decay") is not None
            else float(weight_decay)
        )

        groups = self.model.get_parameters_groups(
            lr=backbone_lr,
            weight_decay=backbone_wd,
        )
        if probe_head is not None:
            groups.append({
                "params": [p for p in probe_head.parameters() if p.requires_grad],
                "lr": head_lr,
                "weight_decay": float(weight_decay),
            })
        groups = [g for g in groups if len(g.get("params", [])) > 0]
        print(
            f"[Optimizer] custom parameter groups: "
            f"backbone_lr={backbone_lr:g}, head_lr={head_lr:g}, backbone_wd={backbone_wd:g}"
        )
        return groups
    
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
        interpolation_mode: str = "linear",
        interpolation_align_corners: bool = False,
        strict_channel_names: bool = False,
    ):
        super().__init__()
        self.model = model
        self.input_freq = input_freq
        self.target_freq = 200 
        self.norm = norm if norm is not None else nn.Identity()
        self.interpolation_mode = interpolation_mode
        self.interpolation_align_corners = bool(interpolation_align_corners)
        self.strict_channel_names = bool(strict_channel_names)

        # add coord info
        self.STANDARD_COORDS_NORM = _build_standard_coords_norm()
   
        #[22, Input_Channels]
        self.transform_matrix = self._build_spatial_matrix(input_chn_names)
        
        grid_h = self.model.patch_embed.grid_size[0]
        self.embed_dim = grid_h * self.model.embed_dim
        self.benchmark_metadata = {
            "implementation": "adapter",
            "variant": "femba_adapter",
            "target_sampling_rate": self.target_freq,
            "target_channels": [f"{a}-{b}" for a, b in TUEG_PAIRS],
            "channel_policy": "coordinate interpolation to TUEG electrode set followed by bipolar pair differencing",
            "official_alignment": "adapter",
        }


    def _build_spatial_matrix(self, input_names):
        num_targets = len(TUEG_PAIRS)
        needed_names = sorted(list(set([p[0] for p in TUEG_PAIRS] + [p[1] for p in TUEG_PAIRS])))
        num_needed = len(needed_names)
        M_interp, missing_inputs, _ = build_coord_projection_matrix(
            input_names,
            needed_names,
            self.STANDARD_COORDS_NORM,
            strict=self.strict_channel_names,
        )
        if missing_inputs and not self.strict_channel_names:
            print(
                f"⚠️ Warning: FEMBA spatial interpolation received unknown channel names: "
                f"{missing_inputs[:5]}{'...' if len(missing_inputs) > 5 else ''}"
            )

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
            x = resample_along_time(
                x,
                self.input_freq,
                self.target_freq,
                mode=self.interpolation_mode,
                align_corners=self.interpolation_align_corners,
            )
        # x: [B, 19, T] -> [B, 22, T]
        x = torch.matmul(self.transform_matrix.to(x.device), x)

        with torch.amp.autocast("cuda", enabled=False):
            x = x.float()
            tokens = self.model.patch_embed(x)
            pos_embed = self.model.pos_embed.float()

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
    def __init__(
        self,
        encoder,        # EEGConformer (22ch, n_times=500)
        decoder,        # NeuroGPTDecoder (GPT-6 with n_embd=1024)
        embedder,       # nn.Sequential: Linear(1080→1024) + LayerNorm
        input_chns,
        norm=None,
        embed_dim=1024,
        chunk_len=500,  # 2 s at 250 Hz
        stride=500,
        input_freq=200,
        interpolation_mode: str = "linear",
        interpolation_align_corners: bool = False,
        strict_channel_names: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.embedder = embedder
        self.chunk_len = chunk_len
        self.stride = stride
        self.input_freq = input_freq
        self.target_freq = 250      # NeuroGPT pretraining rate
        self.embed_dim = embed_dim
        self.interpolation_mode = interpolation_mode
        self.interpolation_align_corners = bool(interpolation_align_corners)
        self.strict_channel_names = bool(strict_channel_names)
        self.STANDARD_COORDS_NORM = _build_standard_coords_norm()
        self.proj_mat = self._build_spatial_matrix(input_chns)
        self.norm = norm if norm is not None else nn.Identity()
        self.benchmark_metadata = {
            "implementation": "encoder_adapter",
            "variant": "neurogpt_encoder_adapter",
            "target_sampling_rate": self.target_freq,
            "target_channels": NEUROGPT_CHANNELS,
            "channel_policy": "coordinate interpolation to NeuroGPT 22-channel set; full EEGConformer+GPT pipeline",
            "official_alignment": "full",
        }

    def _build_spatial_matrix(self, input_names):
        proj, missing_inputs, _ = build_coord_projection_matrix(
            input_names,
            NEUROGPT_CHANNELS,
            self.STANDARD_COORDS_NORM,
            strict=self.strict_channel_names,
        )
        if missing_inputs and not self.strict_channel_names:
            print(
                f"⚠️ Warning: NeuroGPT spatial interpolation received unknown channel names: "
                f"{missing_inputs[:5]}{'...' if len(missing_inputs) > 5 else ''}"
            )
        return proj

    def forward(self, x):
        # 1. Normalise
        x = self.norm(x)                                                    # [B, C, T]
        # 2. Channel projection → 22 NeuroGPT channels
        x = torch.einsum('oc,bct->bot', self.proj_mat.to(x.device), x)    # [B, 22, T]
        # 3. Resample to 250 Hz
        if self.input_freq != self.target_freq:
            x = resample_along_time(
                x, self.input_freq, self.target_freq,
                mode=self.interpolation_mode,
                align_corners=self.interpolation_align_corners,
            )
        B, C, T = x.shape
        # 4. Pad if shorter than one chunk
        if T < self.chunk_len:
            x = torch.nn.functional.pad(x, (0, self.chunk_len - T))
        # 5. Unfold → [B, N, 22, chunk_len]
        chunks = x.unfold(2, self.chunk_len, self.stride)                  # [B, 22, N, L]
        N = chunks.shape[2]
        chunks = chunks.permute(0, 2, 1, 3).contiguous()                   # [B, N, 22, L]
        # 6. EEGConformer encodes each chunk → [B*N, seq_tokens, 40]
        enc = self.encoder(chunks)                                          # [B*N, 27, 40]
        # 7. Flatten per chunk and embed to GPT dimension
        enc_flat = enc.reshape(B * N, -1)                                  # [B*N, 1080]
        tok = self.embedder(enc_flat)                                       # [B*N, 1024]
        tok = tok.view(B, N, -1)                                            # [B, N, 1024]
        # 8. GPT decoder: positional + transformer + pooler → [B, 1024]
        feat = self.decoder(tok)
        return feat

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
from typing import Optional, Literal, Sequence, Dict
import torch
import torch.nn as nn
from einops import rearrange


class EEGMambaFeatureExtractor(nn.Module):
    """
    EEGMamba feature extractor with channel reordering + length normalization.

    输入:
        x: [B, C, T]
    其中 C 维的通道顺序由 ch_name 指定。

    该模块会先根据 ch_name 将 x 重排/补零到 canonical_order，
    再送入 EEGMamba。
    """
    def __init__(
        self,
        model: nn.Module,
        patch_size: int,
        ch_name: Sequence[str],
        canonical_order: Sequence[str],
        target_samples: int = 6000,
        norm: Optional[nn.Module] = None,
        crop_mode: Literal["center_crop", "left_crop", "right_crop"] = "center_crop",
        pad_mode: Literal["zero", "repeat", "reflect"] = "zero",
        strict: bool = False,
        use_legacy_1020_alias: bool = True,
    ):
        super().__init__()
        self.model = model
        self.patch_size = patch_size
        self.norm = norm if norm is not None else IdentityNorm()
        self.strict = strict
        self.use_legacy_1020_alias = use_legacy_1020_alias

        # 输入通道名（与 x 的 channel 维一一对应）
        self.ch_name = [self._normalize_name(n) for n in ch_name]

        # 目标标准顺序
        self.canonical_order = [self._normalize_name(n) for n in canonical_order]

        self.length_adapter = LengthAdapter(
            target_samples=target_samples,
            mode=crop_mode,
            pad_mode=pad_mode,
        )

        if hasattr(self.model, "proj_out"):
            self.embed_dim = self.model.proj_out[0].out_features
        else:
            raise AttributeError("EEGMamba must have proj_out")

        # 预先建立输入通道到索引的映射
        self._name_to_idx = self._build_name_to_idx(self.ch_name)

        # 预先建立 canonical_order 对应的索引表
        self._reorder_index = self._build_reorder_index()
        self.missing_channels = [name for name, idx in zip(self.canonical_order, self._reorder_index) if idx < 0]
        self.benchmark_metadata = {
            "implementation": "adapter",
            "variant": "eegmamba_19ch_adapter",
            "target_sampling_rate": None,
            "target_channels": self.canonical_order,
            "target_samples": int(target_samples),
            "channel_policy": "reorder to canonical 19-channel set; missing channels are zero-filled",
            "missing_channels": self.missing_channels,
            "official_alignment": "adapter",
        }

    def _normalize_name(self, name: str) -> str:
        name = _normalize_eeg_coord_name(name)

        if self.use_legacy_1020_alias:
            alias_map = {
                # 新命名 -> 旧命名（因为你前面更倾向于 T3/T4/T5/T6 体系）
                "T7": "T3",
                "T8": "T4",
                "P7": "T5",
                "P8": "T6",
                # 常见大小写变体其实 upper 后已统一，这里主要留给可扩展
            }
            name = alias_map.get(name, name)

        return name

    def _build_name_to_idx(self, ch_name: Sequence[str]) -> Dict[str, int]:
        """
        若有重复通道名，默认保留第一次出现。
        """
        name_to_idx: Dict[str, int] = {}
        for i, name in enumerate(ch_name):
            if name not in name_to_idx:
                name_to_idx[name] = i
        return name_to_idx

    def _build_reorder_index(self):
        """
        返回长度为 len(canonical_order) 的索引列表:
        - 若某目标通道存在，则给出其在输入 x 中的 channel index
        - 若不存在，则记为 -1，forward 时补零
        """
        reorder_index = []
        missing = []

        for name in self.canonical_order:
            if name in self._name_to_idx:
                reorder_index.append(self._name_to_idx[name])
            else:
                reorder_index.append(-1)
                missing.append(name)

        if self.strict and len(missing) > 0:
            raise ValueError(
                f"Missing channels for canonical_order: {missing}. "
                f"Available channels: {list(self._name_to_idx.keys())}"
            )

        return reorder_index

    def _reorder_to_canonical(self, x: torch.Tensor) -> torch.Tensor:
        """
        将 x: [B, C, T] 重排为 [B, C_target, T]
        对缺失通道补零。
        """
        if x.ndim != 3:
            raise ValueError(f"Expected x [B,C,T], got {tuple(x.shape)}")

        B, C, T = x.shape
        if C != len(self.ch_name):
            raise ValueError(
                f"x has {C} channels, but len(ch_name)={len(self.ch_name)}. "
                f"These must match exactly."
            )

        out = []
        for idx in self._reorder_index:
            if idx >= 0:
                out.append(x[:, idx:idx + 1, :])  # [B,1,T]
            else:
                out.append(torch.zeros(B, 1, T, dtype=x.dtype, device=x.device))

        return torch.cat(out, dim=1)  # [B, C_target, T]

    def _to_4d(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        if T % self.patch_size != 0:
            raise ValueError(
                f"T={T} is not divisible by patch_size={self.patch_size}."
            )
        L = T // self.patch_size
        return rearrange(x, "b c (l s) -> b c l s", l=L, s=self.patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T_any]
        x = self.norm(x)

        # 1) 先按通道名重排到 canonical_order
        x = self._reorder_to_canonical(x)   # [B, C_target, T_any]

        # 2) 再做长度标准化
        x = self.length_adapter(x)          # [B, C_target, target_samples]

        # 3) 切成 EEGMamba 需要的 4D
        x4 = self._to_4d(x)                 # [B, C_target, L, patch_size]

        # 4) Mamba forward
        with torch.amp.autocast("cuda", enabled=False):
            feats = self.model(x4)          # [B, C_target, L, D]

        return feats.mean(dim=(1, 2))       # [B, D]

# class EEGMambaFeatureExtractor(nn.Module):
#     """
#     EEGMamba feature extractor with length normalization.
#     """
#     def __init__(
#         self,
#         model: nn.Module,
#         patch_size: int,
#         target_samples: int = 6000,
#         norm: Optional[nn.Module] = None,
#         crop_mode: Literal["center_crop", "left_crop", "right_crop"] = "center_crop",
#         pad_mode: Literal["zero", "repeat", "reflect"] = "zero",
#     ):
#         super().__init__()
#         self.model = model
#         self.patch_size = patch_size
#         self.norm = norm if norm is not None else IdentityNorm()

#         self.length_adapter = LengthAdapter(
#             target_samples=target_samples,
#             mode=crop_mode,
#             pad_mode=pad_mode,
#         )

#         if hasattr(self.model, "proj_out"):
#             self.embed_dim = self.model.proj_out[0].out_features
#         else:
#             raise AttributeError("EEGMamba must have proj_out")

#     def _to_4d(self, x: torch.Tensor) -> torch.Tensor:
#         B, C, T = x.shape
#         L = T // self.patch_size
#         return rearrange(x, "b c (l s) -> b c l s", l=L, s=self.patch_size)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # x: [B, C, T_any]
#         x = self.norm(x)
#         x = self.length_adapter(x)       # ✅ ALWAYS [B,C,6000]
#         x4 = self._to_4d(x)              # → [B,C,30,200]

#         # 🚨 Mamba kernels are NOT AMP-safe
#         with torch.amp.autocast("cuda", enabled=False):
#             feats = self.model(x4)  # [B,C,L,D]

#         return feats.mean(dim=(1, 2))    # [B,D]

from typing import Optional, Sequence, List, Dict

class NeuroLMFeatureExtractor(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        pool: PoolMode = "mean",
        norm: Optional[nn.Module] = None,
        standard_1020: Optional[Sequence[str]] = None,
        target_chn_names: Optional[Sequence[str]] = None,
        pad_missing: bool = False,
        input_chn_names: Sequence[str] = NEUROGPT_CHANNELS,
        variant: str = "tokenizer_probe",
    ):
        super().__init__()
        self.model = model
        self.pool = pool
        self.norm = norm if norm is not None else IdentityNorm()
        self.pad_missing = pad_missing
        self.input_chn_names = input_chn_names
        self.variant = str(variant or "tokenizer_probe").lower()
        self.use_gpt_body = self.variant in {"full_gpt", "gpt", "neurolm_full_gpt", "full"}
        self._warned_missing_standard_1020 = set()
        
        # NeuroLM standard channel vocabulary
        if standard_1020 is None:
            raise ValueError("standard_1020 must be provided and must match the NeuroLM checkpoint.")
        self.standard_1020 = [n.strip().upper() for n in standard_1020]
        self.standard_1020_map = {n: i for i, n in enumerate(self.standard_1020)}

        self.target_chn_names = None
        if target_chn_names is not None:
            self.target_chn_names = [n.strip().upper() for n in target_chn_names]

        target_samples = getattr(self.model.tokenizer, "patch_size", None)
        if target_samples is None:
            target_samples = 200
        self.length_adapter = LengthAdapter(
            target_samples=int(target_samples),
            mode="center_crop",
            pad_mode="zero",
        )

        # Freeze tokenizer by default
        self.model.tokenizer.eval()
        for p in self.model.tokenizer.parameters():
            p.requires_grad = False

        self.embed_dim = self.model.GPT2.config.n_embd
        self.benchmark_metadata = {
            "implementation": "full_gpt" if self.use_gpt_body else "tokenizer_probe",
            "variant": "neurolm_full_gpt" if self.use_gpt_body else "neurolm_tokenizer_probe",
            "target_sampling_rate": None,
            "target_channels": self.target_chn_names if self.target_chn_names is not None else "dataset-native filtered by NeuroLM vocabulary",
            "target_samples": int(target_samples),
            "channel_policy": "use NeuroLM tokenizer/channel embeddings; missing requested channels may be padded and masked",
            "uses_gpt_body": self.use_gpt_body,
            "time_policy": "patchify full window into tokenizer patch sequence" if self.use_gpt_body else "center crop/pad to one tokenizer patch",
            "official_alignment": "closer_to_official" if self.use_gpt_body else "adapter",
        }

    def _normalize_names(self, names: Sequence[str]) -> List[str]:
        return [_normalize_eeg_coord_name(n) for n in names]

    def _select_channels(
        self,
        x: torch.Tensor,
        input_chn_names: Sequence[str],
    ):
        input_names = self._normalize_names(input_chn_names)

        if x.ndim != 3:
            raise ValueError(f"Expected x [B,C,T], got {tuple(x.shape)}")
        if x.shape[1] != len(input_names):
            raise ValueError(
                f"x has {x.shape[1]} channels but got {len(input_names)} input_chn_names"
            )

        # first occurrence wins
        name_to_idx: Dict[str, int] = {}
        for i, name in enumerate(input_names):
            if name not in name_to_idx:
                name_to_idx[name] = i

        # no target subset specified: keep all input channels in current order
        if self.target_chn_names is None:
            valid_mask = torch.tensor(
                [name in self.standard_1020_map for name in input_names],
                dtype=torch.bool,
                device=x.device,
            )
            return x, input_names, valid_mask

        selected_tensors = []
        selected_names = []
        valid_list = []

        for name in self.target_chn_names:
            if name in name_to_idx:
                idx = name_to_idx[name]
                selected_tensors.append(x[:, idx:idx+1, :])
                selected_names.append(name)
                valid_list.append(True)
            else:
                if self.pad_missing:
                    zeros = torch.zeros(
                        x.shape[0], 1, x.shape[2],
                        dtype=x.dtype, device=x.device
                    )
                    selected_tensors.append(zeros)
                    selected_names.append(name)
                    valid_list.append(False)
                # else: skip missing channel

        if len(selected_tensors) == 0:
            raise ValueError("No requested target channels were found in input_chn_names.")

        x_sel = torch.cat(selected_tensors, dim=1)
        valid_mask = torch.tensor(valid_list, dtype=torch.bool, device=x.device)
        return x_sel, selected_names, valid_mask

    def _build_input_chans(self, selected_names: Sequence[str], batch_size: int, device: torch.device):
        ch_ids = []
        for name in selected_names:
            if name not in self.standard_1020_map:
                if name not in self._warned_missing_standard_1020:
                    print(
                        f"⚠️ Warning: NeuroLM channel `{name}` not found in standard_1020. "
                        "Falling back to channel id 0 and relying on the attention mask to suppress it."
                    )
                    self._warned_missing_standard_1020.add(name)
                ch_ids.append(0)
            else:
                ch_ids.append(self.standard_1020_map[name])

        ch_ids = torch.tensor(ch_ids, dtype=torch.long, device=device)   # [C]
        input_chans = ch_ids.unsqueeze(0).repeat(batch_size, 1)          # [B, C]
        return input_chans

    def _build_token_sequence(
        self,
        x: torch.Tensor,
        selected_names: Sequence[str],
        valid_mask: torch.Tensor,
    ):
        B, C, T = x.shape
        device = x.device
        patch_size = int(self.length_adapter.target_samples)

        if self.use_gpt_body:
            pad = (-T) % patch_size
            if pad:
                x = torch.nn.functional.pad(x, (0, pad))
            P = x.shape[-1] // patch_size
            x = x.view(B, C, P, patch_size).permute(0, 2, 1, 3).contiguous()
            x = x.view(B, P * C, patch_size)

            ch_ids = self._build_input_chans(selected_names, batch_size=1, device=device)[0]
            input_chans = ch_ids.repeat(P).unsqueeze(0).repeat(B, 1)
            input_time = torch.arange(P, dtype=torch.long, device=device).repeat_interleave(C)
            input_time = input_time.unsqueeze(0).repeat(B, 1)
            input_mask = valid_mask.to(dtype=torch.long, device=device).repeat(P).unsqueeze(0).repeat(B, 1)
            return x, input_chans, input_time, input_mask

        x = self.length_adapter(x)
        input_chans = self._build_input_chans(selected_names, batch_size=B, device=device)
        input_time = torch.zeros((B, C), dtype=torch.long, device=device)
        input_mask = valid_mask.to(dtype=torch.long, device=device).unsqueeze(0).repeat(B, 1)
        return x, input_chans, input_time, input_mask

    def _pool_tokens(self, tokens: torch.Tensor, input_mask: torch.Tensor) -> torch.Tensor:
        if self.pool == "cls":
            return tokens[:, 0]
        if self.pool == "mean":
            if self.pad_missing or self.use_gpt_body:
                w = input_mask.unsqueeze(-1).to(tokens.dtype)
                denom = w.sum(dim=1).clamp_min(1.0)
                return (tokens * w).sum(dim=1) / denom
            return tokens.mean(dim=1)
        raise ValueError(f"Unsupported NeuroLM pool mode: {self.pool!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)

        # 1) select / reorder channels by name
        x, selected_names, valid_mask = self._select_channels(x, self.input_chn_names)

        x, input_chans, input_time, input_mask = self._build_token_sequence(
            x,
            selected_names,
            valid_mask,
        )

        seq_len = x.shape[1]
        attn_mask = input_mask.unsqueeze(1).repeat(1, seq_len, 1).unsqueeze(1)  # [B,1,N,N]

        # 4) tokenizer
        tokens = self.model.tokenizer(
            x,
            input_chans=input_chans,
            input_times=input_time,
            mask=attn_mask,
            return_all_tokens=True,
        )

        tokens = self.model.encode_transform_layer(tokens)
        tokens = tokens + self.model.pos_embed(input_chans)

        if self.use_gpt_body:
            tokens = self.model.GPT2(
                x_eeg=tokens,
                eeg_time_idx=input_time,
                eeg_mask=attn_mask,
                lm_head=False,
            )

        return self._pool_tokens(tokens, input_mask)


# class NeuroLMFeatureExtractor(nn.Module):
#     """
#     NeuroLM EEG feature extractor.

#     Uses tokenizer + embedding layers only.
#     DOES NOT call GPT2 language model.
#     """
#     def __init__(
#         self,
#         model: nn.Module,
#         pool: PoolMode = "mean",
#         norm: Optional[nn.Module] = None,
#     ):
#         super().__init__()
#         self.model = model
#         self.pool = pool
#         self.norm = norm if norm is not None else IdentityNorm()

#         # NeuroLM's tokenizer was pretrained with a fixed patch_size (timepoints) per channel-token.
#         # Many downstream datasets use different window lengths, which will break the tokenizer's
#         # TemporalConv -> Linear(in_features=400) projection. Normalize to the expected length.
#         target_samples = getattr(self.model.tokenizer, "patch_size", None)
#         if target_samples is None:
#             target_samples = 200
#         self.length_adapter = LengthAdapter(
#             target_samples=int(target_samples),
#             mode="center_crop",
#             pad_mode="zero",
#         )

#         # Freeze tokenizer & GPT by default (pretraining behavior)
#         self.model.tokenizer.eval()
#         for p in self.model.tokenizer.parameters():
#             p.requires_grad = False

#         # Embedding dimension
#         self.embed_dim = self.model.GPT2.config.n_embd

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         x: [B, C, T]
#         """
#         x = self.norm(x)
#         x = self.length_adapter(x)   # [B, C, T_fixed]

#         B, C, T = x.shape
#         device = x.device

#         # ---- build required NeuroLM inputs ----
#         input_chans = torch.arange(C, device=device).unsqueeze(0).repeat(B, 1)
#         input_time = torch.zeros((B, C), dtype=torch.long, device=device)
#         # NeuroLM tokenizer/transformer attends over channel-tokens (sequence length == C after patch_embed),
#         # so the attention mask must be built over C (not raw time samples T).
#         input_mask = torch.ones((B, C), dtype=torch.long, device=device)

#         # ---- tokenizer ----
#         tokens = self.model.tokenizer(
#             x,
#             input_chans=input_chans,
#             input_times=input_time,
#             mask=input_mask.unsqueeze(1).repeat(1, C, 1).unsqueeze(1),
#             return_all_tokens=True,
#         )

#         tokens = self.model.encode_transform_layer(tokens)
#         tokens = tokens + self.model.pos_embed(input_chans)

#         # tokens: [B, N, D]
#         if self.pool == "cls":
#             return tokens[:, 0]
#         else:
#             return tokens.mean(dim=1)
BENDR_19 = [
"FP1", "FP2",
"F7", "F3", "FZ", "F4", "F8",
"T7", "C3", "CZ", "C4", "T8",
"P7", "P3", "PZ", "P4", "P8",
"O1", "O2",
]

ALIASES = {
    "T3": "T7",
    "T4": "T8",
    "T5": "P7",
    "T6": "P8",
}
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
        norm: Optional[nn.Module] = None,
        ch_names: Optional[list[str]] = BENDR_19,
        strict_channels: bool = True,
        interpolation_mode: str = "linear",
        interpolation_align_corners: bool = False,
        input_freq: int = 200,
    ):
        super().__init__()
        self.model = model
        self.norm = norm if norm is not None else IdentityNorm()
        self.input_freq = int(input_freq)
        self.target_freq = 256
        self.strict_channels = bool(strict_channels)
        self.interpolation_mode = interpolation_mode
        self.interpolation_align_corners = bool(interpolation_align_corners)
        # Remove projection / classifier heads if present
        for attr in ("projection_mlp", "extended_classifier", "classifier"):
            if hasattr(self.model, attr):
                setattr(self.model, attr, nn.Identity())

        # Embed dim is encoder_h
        self.embed_dim = int(self.model.encoder_h)
        self.ch_names = ch_names
        x, _, missing = self.bendr_prepare_input(torch.rand(1, len(self.ch_names), 256), self.ch_names)
        self.missing_channels = list(missing)
        self.benchmark_metadata = {
            "implementation": "adapter",
            "variant": "bendr_19plus1_adapter",
            "target_sampling_rate": self.target_freq,
            "target_channels": BENDR_19 + ["RELATIVE_AMPLITUDE"],
            "channel_policy": "map to 19 canonical referential channels with zero-fill, then append relative-amplitude channel",
            "missing_channels": self.missing_channels,
            "official_alignment": "adapter",
        }
        if len(missing) > 0:
            msg = f"BENDR requires canonical referential channel names; missing channels: {missing}."
            if self.strict_channels:
                raise ValueError(msg)
            print(f"⚠️ Warning: {msg}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, T]
        """
        # print(x.shape, self.ch_names)
        x, _, missing = self.bendr_prepare_input(x, self.ch_names)
        if len(missing) > 0:
            msg = f"BENDR requires canonical referential channel names; missing channels: {missing}."
            if self.strict_channels:
                raise ValueError(msg)
        x = self.norm(x)

        if self.input_freq != self.target_freq:
            x = resample_along_time(
                x,
                self.input_freq,
                self.target_freq,
                mode=self.interpolation_mode,
                align_corners=self.interpolation_align_corners,
            )

        # Replicate BENDRClassification.features_forward
        encoded = self.model.encoder(x)           # [B, D, T′]
        context = self.model.contextualizer(encoded)

        # last timestep is the representation
        feat = context[:, :, -1]                  # [B, D]
        return feat

    def _norm_name(self, name: str) -> str:
        name = _normalize_eeg_coord_name(name)
        return ALIASES.get(name, name)

    def bendr_prepare_input(self,
        data: torch.Tensor,
        channel_names: list[str],
        dataset_global_min: float | None = None,
        dataset_global_max: float | None = None,
        add_relative_amplitude: bool = True,
    ):
        """
        将 [B, C, T] + channel_names 转成 BENDR 可接受的 [B, 20, T]

        Args:
            data: torch.Tensor, shape [B, C, T]
            channel_names: 长度为 C 的通道名列表
            dataset_global_min, dataset_global_max:
                用来构造第20个 relative amplitude channel。
                如果不给，就退化为按当前 batch 估计。
            add_relative_amplitude:
                True -> 输出 [B, 20, T]
                False -> 只输出 [B, 19, T]

        Returns:
            x_bendr: [B, 20, T] 或 [B, 19, T]
            mapped_names: 最终 19 个 EEG 通道名
        """
        assert data.ndim == 3, f"expect [B,C,T], got {tuple(data.shape)}"
        B, C, T = data.shape
        assert len(channel_names) == C, "channel_names 长度必须等于 data.shape[1]"

        device = data.device
        dtype = data.dtype

        # 1) 先做通道名规范化
        src_map = {self._norm_name(ch): i for i, ch in enumerate(channel_names)}

        # 2) Map to fixed 19 channels. Missing channels stay exactly zero.
        # Scale only channels that are actually present; otherwise zero-fill would
        # become a non-zero constant after min-max scaling.
        x19_raw = torch.zeros((B, len(BENDR_19), T), device=device, dtype=dtype)
        present = []
        missing = []

        for i, ch in enumerate(BENDR_19):
            if ch in src_map:
                x19_raw[:, i, :] = data[:, src_map[ch], :]
                present.append(i)
            else:
                missing.append(ch)

        x19 = torch.zeros_like(x19_raw)
        if present:
            present_tensor = x19_raw[:, present, :]
            seq_min = present_tensor.amin(dim=(1, 2), keepdim=True)
            seq_max = present_tensor.amax(dim=(1, 2), keepdim=True)
            denom = (seq_max - seq_min).clamp_min(1e-8)
            x19[:, present, :] = 2.0 * (present_tensor - seq_min) / denom - 1.0

        if not add_relative_amplitude:
            return x19, BENDR_19, missing

        # 4) 构造第 20 个 relative amplitude channel（常数通道）
        #    论文公式本质上是：
        #    当前子序列幅值范围 / 整个数据集幅值范围
        if dataset_global_min is None or dataset_global_max is None:
            # 没有整个数据集的统计量时，用当前 batch 近似
            dataset_global_min = float(data.amin().item())
            dataset_global_max = float(data.amax().item())

        dataset_range = max(dataset_global_max - dataset_global_min, 1e-8)

        if present:
            present_raw = x19_raw[:, present, :]
            seq_range = present_raw.amax(dim=(1, 2)) - present_raw.amin(dim=(1, 2))
        else:
            seq_range = torch.zeros((B,), device=device, dtype=dtype)
        rel_amp = (seq_range / dataset_range).view(B, 1, 1).expand(B, 1, T)

        x20 = torch.cat([x19, rel_amp.to(dtype=dtype, device=device)], dim=1)
        return x20, BENDR_19 + ["RELATIVE_AMPLITUDE"], missing

class BENDRLearnedChannelAdapterFeatureExtractor(nn.Module):
    """BENDR wrapper with trainable [B,N,T] -> Conv1d(N,20,1) channel projection."""
    def __init__(self, model: nn.Module, in_channels: int, input_channel_names: Sequence[str], norm: Optional[nn.Module] = None, interpolation_mode: str = "linear", interpolation_align_corners: bool = False, input_freq: int = 200):
        super().__init__()
        self.model = model
        self.norm = norm if norm is not None else IdentityNorm()
        self.input_freq = int(input_freq)
        self.target_freq = 256
        self.interpolation_mode = interpolation_mode
        self.interpolation_align_corners = bool(interpolation_align_corners)
        self.input_channel_names = [str(name) for name in input_channel_names]
        self.channel_adapter = nn.Conv1d(int(in_channels), 20, kernel_size=1, bias=False)
        nn.init.xavier_uniform_(self.channel_adapter.weight)

        for attr in ("projection_mlp", "extended_classifier", "classifier"):
            if hasattr(self.model, attr):
                setattr(self.model, attr, nn.Identity())

        self.embed_dim = int(self.model.encoder_h)
        self.benchmark_metadata = {
            "implementation": "adapter",
            "variant": "bendr_learned_channel_adapter",
            "target_sampling_rate": self.target_freq,
            "target_channels": [f"LEARNED_{idx:02d}" for idx in range(20)],
            "input_channels": self.input_channel_names,
            "channel_policy": "trainable Conv1d(in_channels=N, out_channels=20, kernel_size=1) before BENDR encoder",
            "official_alignment": "learned_channel_adapter",
        }

    def enable_linear_probe_trainables(self) -> None:
        self.channel_adapter.requires_grad_(True)
        self.channel_adapter.train(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.ndim == 3:
            raise ValueError(f"BENDR learned adapter expects [B, C, T], got shape {tuple(x.shape)}")
        expected_channels = self.channel_adapter.in_channels
        if x.shape[1] != expected_channels:
            raise ValueError(f"BENDR learned adapter expected {expected_channels} channels, got {x.shape[1]}")

        x = self.norm(x)
        x = self.channel_adapter(x)
        if self.input_freq != self.target_freq:
            x = resample_along_time(x, self.input_freq, self.target_freq, mode=self.interpolation_mode, align_corners=self.interpolation_align_corners)

        encoded = self.model.encoder(x)
        context = self.model.contextualizer(encoded)
        return context[:, :, -1]

    def get_parameter_groups(self, lr: float, weight_decay: float, *, probe_head: Optional[nn.Module] = None, cfg=None):
        adapter_lr = float(OmegaConf.select(cfg, "model.bendr.adapter_lr") if cfg is not None and OmegaConf.select(cfg, "model.bendr.adapter_lr") is not None else float(lr))
        head_lr = float(OmegaConf.select(cfg, "model.bendr.head_lr") if cfg is not None and OmegaConf.select(cfg, "model.bendr.head_lr") is not None else float(lr))
        groups = []
        _append_param_group(groups, params=list(self.channel_adapter.parameters()), lr=adapter_lr, weight_decay=float(weight_decay), name="bendr.channel_adapter")
        if probe_head is not None:
            head_decay, head_no_decay = [], []
            for name, param in probe_head.named_parameters():
                (head_no_decay if _is_no_decay_param(name, param) else head_decay).append(param)
            _append_param_group(groups, params=head_decay, lr=head_lr, weight_decay=float(weight_decay), name="bendr.head_decay")
            _append_param_group(groups, params=head_no_decay, lr=head_lr, weight_decay=0.0, name="bendr.head_no_decay")
        return groups


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
        interpolation_mode: str = "linear",
        interpolation_align_corners: bool = False,
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
        self.interpolation_mode = interpolation_mode
        self.interpolation_align_corners = bool(interpolation_align_corners)

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
            x = resample_along_time(
                x,
                self.input_freq,
                self.target_freq,
                mode=self.interpolation_mode,
                align_corners=self.interpolation_align_corners,
            )
            
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
_DEFAULT_BACKBONE_LR = {
    # Small/non-pretrained baselines can usually use the global training LR.
    "eegnet": None,
    "eegconformer": None,

    # Pretrained/large backbones need conservative full-finetune LR by default.
    "biot": 5e-5,
    "cbramod": 5e-5,
    "reve": 1e-5,
    "labram": 3e-5,
    "brainomni": 3e-5,
    "femba": 5e-5,
    "neurogpt": 5e-5,
    "neurolm": 3e-5,
    "eegmamba": 3e-5,
    "bendr": 3e-5,
    "brant": 1e-5,
}


def _is_no_decay_param(name: str, param: nn.Parameter) -> bool:
    lname = name.lower()
    return (
        param.ndim <= 1
        or lname.endswith(".bias")
        or "norm" in lname
        or "bn" in lname
        or "ln" in lname
    )


def _append_param_group(groups, *, params, lr: float, weight_decay: float, name: str):
    params = [p for p in params if p.requires_grad]
    if params:
        groups.append({
            "params": params,
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "name": name,
        })


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

    def get_parameter_groups(self, lr: float, weight_decay: float, cfg=None):
        custom_getter = getattr(self.feature_extractor, "get_parameter_groups", None)
        if callable(custom_getter):
            return custom_getter(
                lr=lr,
                weight_decay=weight_decay,
                probe_head=self.probe_head,
                cfg=cfg,
            )

        cfg_model_name = OmegaConf.select(cfg, "model.name") if cfg is not None else None
        model_name = str(
            getattr(self, "benchmark_metadata", {}).get("base_model")
            or getattr(self, "benchmark_metadata", {}).get("requested_name")
            or cfg_model_name
            or ""
        ).lower()

        default_backbone_lr = _DEFAULT_BACKBONE_LR.get(model_name)
        cfg_backbone_lr = OmegaConf.select(cfg, f"model.{model_name}.backbone_lr") if cfg is not None and model_name else None
        cfg_head_lr = OmegaConf.select(cfg, f"model.{model_name}.head_lr") if cfg is not None and model_name else None
        cfg_wd = OmegaConf.select(cfg, f"model.{model_name}.weight_decay") if cfg is not None and model_name else None
        no_decay_norm_bias = bool(
            OmegaConf.select(cfg, f"model.{model_name}.no_decay_norm_bias")
            if cfg is not None and model_name and OmegaConf.select(cfg, f"model.{model_name}.no_decay_norm_bias") is not None
            else True
        )

        backbone_lr = float(cfg_backbone_lr if cfg_backbone_lr is not None else (default_backbone_lr if default_backbone_lr is not None else lr))
        head_lr = float(cfg_head_lr if cfg_head_lr is not None else lr)
        group_wd = float(cfg_wd if cfg_wd is not None else weight_decay)

        groups = []
        if no_decay_norm_bias:
            backbone_decay, backbone_no_decay = [], []
            for name, param in self.feature_extractor.named_parameters():
                (backbone_no_decay if _is_no_decay_param(name, param) else backbone_decay).append(param)
            head_decay, head_no_decay = [], []
            for name, param in self.probe_head.named_parameters():
                (head_no_decay if _is_no_decay_param(name, param) else head_decay).append(param)

            _append_param_group(groups, params=backbone_decay, lr=backbone_lr, weight_decay=group_wd, name=f"{model_name}.backbone_decay")
            _append_param_group(groups, params=backbone_no_decay, lr=backbone_lr, weight_decay=0.0, name=f"{model_name}.backbone_no_decay")
            _append_param_group(groups, params=head_decay, lr=head_lr, weight_decay=float(weight_decay), name=f"{model_name}.head_decay")
            _append_param_group(groups, params=head_no_decay, lr=head_lr, weight_decay=0.0, name=f"{model_name}.head_no_decay")
        else:
            _append_param_group(
                groups,
                params=list(self.feature_extractor.parameters()),
                lr=backbone_lr,
                weight_decay=group_wd,
                name=f"{model_name}.backbone",
            )
            _append_param_group(
                groups,
                params=list(self.probe_head.parameters()),
                lr=head_lr,
                weight_decay=float(weight_decay),
                name=f"{model_name}.head",
            )

        if not groups:
            groups = [{
                "params": [p for p in self.parameters() if p.requires_grad],
                "lr": float(lr),
                "weight_decay": float(weight_decay),
                "name": f"{model_name}.all",
            }]

        desc = ", ".join(
            f"{g.get('name', 'group')}: lr={g['lr']:g}, wd={g['weight_decay']:g}, n={len(g['params'])}"
            for g in groups
        )
        print(f"[Optimizer] parameter groups for {model_name or 'model'}: {desc}")
        return groups


# -----------------------------
# Training mode helpers
# -----------------------------
def set_linear_probe(model: DownstreamModel) -> None:
    """
    Freeze feature extractor; train the head, plus any explicit probe-time adapters.
    """
    model.feature_extractor.requires_grad_(False)
    model.probe_head.requires_grad_(True)
    model.feature_extractor.eval()  # optional but recommended for deterministic behavior
    enable_probe_adapters = getattr(model.feature_extractor, "enable_linear_probe_trainables", None)
    if callable(enable_probe_adapters):
        enable_probe_adapters()


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
