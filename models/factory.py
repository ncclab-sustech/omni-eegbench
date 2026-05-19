# models/factory.py
import os
import torch
import json
import torch.nn as nn
from typing import Any, Dict, Iterable, Optional, Tuple, List
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import load_file
from timm.models import create_model
import models.modeling_finetune
from models.cbramod import CBraMod
from models.biot import BIOTClassifier
from models.brainomni import BrainOmni
from models.FEMBA import FEMBA
from models.neurogpt import EEGConformer, NeuroGPTDecoder
from models.EEGConformer_baseline import Conformer
import numpy as np
from models.eegmamba import EEGMamba
from models.neurolm import NeuroLM
from models.model_gpt import GPTConfig
from models.bendr import BENDRClassification
from models.brant import Brant
from models.EEGNet import EEGNet

from models.wrappers import (
    LaBraMFeatureExtractor,
    BIOTFeatureExtractor,
    CBraModFeatureExtractor,
    BrainOmniFeatureExtractor,
    REVEFeatureExtractor,
    REVENewFeatureExtractor,
    FembaFeatureExtractor,
    NeuroGPTFeatureExtractor,
    # Add new model extractor here
    BrantPretrainFeatureExtractor,
    NeuroLMFeatureExtractor,
    EEGMambaFeatureExtractor,
    BENDRFeatureExtractor,
    BENDRLearnedChannelAdapterFeatureExtractor,
    ProbeHead,
    DownstreamModel,
    set_linear_probe,
    set_full_finetune,
    IdentityNorm,
    PerTrialZScore,
    INTERPOLATION_VOCAB_PATH,
    _build_standard_coords_norm,
    _build_interpolation_coords_norm,
    _normalize_eeg_coord_name,
)
from data.metadata import load_dataset_metadata

STANDARD_1020 = [
    'FP1', 'FPZ', 'FP2', 'AF9', 'AF7', 'AF5', 'AF3', 'AF1', 'AFZ', 'AF2', 'AF4', 'AF6', 'AF8', 'AF10',
    'F9', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'F10',
    'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10',
    'T9', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'T10',
    'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10',
    'P9', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'P10',
    'PO9', 'PO7', 'PO5', 'PO3', 'PO1', 'POZ', 'PO2', 'PO4', 'PO6', 'PO8', 'PO10',
    'O1', 'OZ', 'O2', 'O9', 'CB1', 'CB2',
    'IZ', 'O10', 'T3', 'T5', 'T4', 'T6', 'M1', 'M2', 'A1', 'A2',
    'CFC1', 'CFC2', 'CFC3', 'CFC4', 'CFC5', 'CFC6', 'CFC7', 'CFC8',
    'CCP1', 'CCP2', 'CCP3', 'CCP4', 'CCP5', 'CCP6', 'CCP7', 'CCP8',
    'T1', 'T2', 'FTT9h', 'TTP7h', 'TPP9h', 'FTT10h', 'TPP8h', 'TPP10h',
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP2-F8", "F8-T8", "T8-P8", "P8-O2", "FP1-F3", "F3-C3", "C3-P3", "P3-O1", "FP2-F4", "F4-C4", "C4-P4", "P4-O2"
]



def _referential_standard_1020() -> List[str]:
    return [name for name in STANDARD_1020 if "-" not in str(name)]


_ELECTRODE_VOCAB_CACHE = None


def _load_electrode_vocab() -> dict:
    global _ELECTRODE_VOCAB_CACHE
    if _ELECTRODE_VOCAB_CACHE is None:
        try:
            with open(INTERPOLATION_VOCAB_PATH, "r", encoding="utf-8") as f:
                vocab = json.load(f)
        except FileNotFoundError:
            vocab = {}
        _ELECTRODE_VOCAB_CACHE = vocab if isinstance(vocab, dict) else {}
    return _ELECTRODE_VOCAB_CACHE


def _channel_coord_lookup_keys(name: str) -> list[str]:
    raw = str(name).strip()
    keys = [_normalize_eeg_coord_name(raw)]
    vocab = _load_electrode_vocab()
    if raw in vocab:
        keys.append(_normalize_eeg_coord_name(vocab[raw]))
    return list(dict.fromkeys(keys))


def _nearest_standard_channels(dataset_ch_names: List[str]) -> tuple[list[Optional[str]], list[bool]]:
    coords = _build_interpolation_coords_norm()
    standard = _referential_standard_1020()
    standard_keys = [_normalize_eeg_coord_name(name) for name in standard]
    standard_with_coords = [
        (name, key, coords[key])
        for name, key in zip(standard, standard_keys)
        if key in coords
    ]
    mapped: list[Optional[str]] = []
    found: list[bool] = []
    for ch in dataset_ch_names:
        lookup_keys = _channel_coord_lookup_keys(ch)
        key = lookup_keys[0]
        if key in standard_keys:
            mapped.append(standard[standard_keys.index(key)])
            found.append(True)
            continue
        coord_key = next((candidate for candidate in lookup_keys if candidate in coords), None)
        if coord_key is None or not standard_with_coords:
            mapped.append(None)
            found.append(False)
            continue
        src = torch.tensor(coords[coord_key], dtype=torch.float32).view(1, -1)
        tgt = torch.tensor(np.stack([item[2] for item in standard_with_coords]), dtype=torch.float32)
        nearest = int(torch.cdist(src, tgt).argmin().item())
        mapped.append(standard_with_coords[nearest][0])
        found.append(True)
    return mapped, found


def map_channels_to_indices(dataset_ch_names: List[str]) -> List[int]:
    upper_standard = [c.upper() for c in STANDARD_1020]
    input_chans = [0]
    missing = []
    ch_use = []
    nearest_names, nearest_found = _nearest_standard_channels(dataset_ch_names)
    for ch, nearest, found in zip(dataset_ch_names, nearest_names, nearest_found):
        if found and nearest is not None and nearest.upper() in upper_standard:
            input_chans.append(upper_standard.index(nearest.upper()) + 1)
            ch_use.append(True)
        else:
            missing.append(ch)
            ch_use.append(False)
    if missing:
        print(f"⚠️ Warning: {len(missing)} channels not found in interpolation montage/STANDARD_1020: {missing[:5]}...")
    
    return input_chans,np.array(ch_use)


def map_channels_to_spatial_order(dataset_ch_names: List[str]) -> tuple[list[int], list[Optional[str]], list[str]]:
    nearest_names, nearest_found = _nearest_standard_channels(dataset_ch_names)
    standard_order = {name.upper(): idx for idx, name in enumerate(_referential_standard_1020())}
    sortable = []
    missing = []
    for idx, (raw_name, nearest, found) in enumerate(zip(dataset_ch_names, nearest_names, nearest_found)):
        if found and nearest is not None:
            sortable.append((standard_order.get(nearest.upper(), len(standard_order)), idx, nearest))
        else:
            missing.append(str(raw_name))
    sortable.sort(key=lambda item: (item[0], item[1]))
    return [idx for _, idx, _ in sortable], nearest_names, missing


def _require(cfg: DictConfig, key: str) -> Any:
    v = OmegaConf.select(cfg, key)
    if v is None:
        raise ValueError(f"[factory] Missing required config key: `{key}`")
    return v


MODEL_ALIAS_TO_BASE = {
    "biot_official_biot18": "biot",
    "biot_zero_fill_adapter": "biot",
    "femba_adapter": "femba",
    "neurogpt_encoder_adapter": "neurogpt",
    "eegmamba_19ch_adapter": "eegmamba",
    "neurolm_tokenizer_probe": "neurolm",
    "neurolm_full_gpt": "neurolm",
    "bendr_19plus1_adapter": "bendr",
    "reve_old": "reve_old",
    "reve_new": "reve_new",
}


def _requested_model_and_variant(cfg: DictConfig) -> tuple[str, str]:
    requested = str(_require(cfg, "model.name")).lower()
    base = MODEL_ALIAS_TO_BASE.get(requested, requested)

    defaults = {
        "biot": "zero_fill_adapter",
        "femba": "adapter",
        "neurogpt": "encoder_adapter",
        "eegmamba": "19ch_adapter",
        "neurolm": "tokenizer_probe",
        "bendr": "19plus1_adapter",
    }
    if requested != base:
        variant = requested.replace(f"{base}_", "", 1)
    else:
        variant = str(OmegaConf.select(cfg, f"model.{base}.variant") or OmegaConf.select(cfg, f"model.{base}.mode") or defaults.get(base, "native")).lower()
    return base, variant


COORD_PATH = "data/standard_coords.pt"
MONTAGE_FALLBACK_PATH = "data/montage(1).json"
COORD_CACHE = None
MONTAGE_FALLBACK_CACHE = None


def _normalize_ch_name_for_lookup(name: str) -> str:
    key = str(name).strip().upper().replace(" ", "").replace(".", "")
    for suffix in ("-REF", "-LE", "-AR", "-AVG", "-A1", "-A2", "-M1", "-M2"):
        if key.endswith(suffix):
            key = key[: -len(suffix)]
    return key


def _load_coords_if_needed():
    global COORD_CACHE
    if COORD_CACHE is None:
        if not os.path.exists(COORD_PATH):
            raise FileNotFoundError(f"Missing coords file: {COORD_PATH}. Please run generate_coords.py")
        print(f"Loading standard coords from {COORD_PATH}")
        data = torch.load(COORD_PATH, map_location="cpu")
        names = [n.upper() for n in data["ch_names"]]
        name_to_idx = {n: i for i, n in enumerate(names)}
        COORD_CACHE = {
            "pos": data["pos"],
            "sensor_type": data["sensor_type"],
            "map": name_to_idx
        }
    return COORD_CACHE


def _load_montage_fallback_if_needed():
    global MONTAGE_FALLBACK_CACHE
    if MONTAGE_FALLBACK_CACHE is None:
        if not os.path.exists(MONTAGE_FALLBACK_PATH):
            MONTAGE_FALLBACK_CACHE = {}
            return MONTAGE_FALLBACK_CACHE
        with open(MONTAGE_FALLBACK_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
        cache = {}
        for name, pos in raw.items():
            if not isinstance(pos, (list, tuple)) or len(pos) < 3:
                continue
            key = _normalize_ch_name_for_lookup(name)
            if key not in cache:
                cache[key] = torch.tensor(
                    [float(pos[0]), float(pos[1]), float(pos[2]), 0.0, 0.0, 0.0],
                    dtype=torch.float32,
                )
        MONTAGE_FALLBACK_CACHE = cache
    return MONTAGE_FALLBACK_CACHE


def map_channels_to_pos_and_type(chn_names):
    cache = _load_coords_if_needed()
    montage = _load_montage_fallback_if_needed()
    pos_list = []
    type_list = []

    for ch in chn_names:
        u = ch.upper()
        if u in cache["map"]:
            idx = cache["map"][u]
            pos_list.append(cache["pos"][idx])
            type_list.append(cache["sensor_type"][idx])
        else:
            key = _normalize_ch_name_for_lookup(ch)
            if key in montage:
                pos_list.append(montage[key])
                type_list.append(torch.tensor(0, dtype=torch.int32))
            else:
                print(f"⚠️ Warning: Channel {ch} not found in standard coords or montage fallback! Using zeros.")
                pos_list.append(torch.zeros(6))
                type_list.append(torch.tensor(0, dtype=torch.int32))

    return torch.stack(pos_list), torch.stack(type_list)


# ============================================================
# checkpoint parsing utilities
# ============================================================
def unwrap_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, nn.Module):
        return ckpt.state_dict()
    if not isinstance(ckpt, dict):
        raise ValueError("Checkpoint is not a dict or nn.Module; cannot unwrap state_dict.")
    for k in ["state_dict", "model", "module", "net", "params", "weights"]:
        v = ckpt.get(k, None)
        if isinstance(v, dict) and any(isinstance(x, torch.Tensor) for x in v.values()):
            return v
    if any(isinstance(x, torch.Tensor) for x in ckpt.values()):
        return ckpt
    raise ValueError(f"Cannot find state_dict in checkpoint keys: {list(ckpt.keys())[:20]} ...")


def strip_prefixes(sd: Dict[str, torch.Tensor], prefixes: Tuple[str, ...]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in sd.items():
        nk = k
        for p in prefixes:
            if nk.startswith(p):
                nk = nk[len(p):]
        out[nk] = v
    return out


def replace_prefix(sd: Dict[str, torch.Tensor], old: str, new: str = "") -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in sd.items():
        nk = k.replace(old, new, 1) if k.startswith(old) else k
        out[nk] = v
    return out


def drop_if_contains(sd: Dict[str, torch.Tensor], substrings: Iterable[str]) -> Dict[str, torch.Tensor]:
    subs = tuple(substrings)
    return {k: v for k, v in sd.items() if not any(s in k for s in subs)}


def filter_by_shape(sd: Dict[str, torch.Tensor], model: nn.Module, verbose: bool = True) -> Dict[str, torch.Tensor]:
    msd = model.state_dict()
    out: Dict[str, torch.Tensor] = {}
    skipped_shape = []
    skipped_missing = 0

    for k, v in sd.items():
        if k not in msd:
            skipped_missing += 1
            continue
        if tuple(v.shape) != tuple(msd[k].shape):
            skipped_shape.append((k, tuple(v.shape), tuple(msd[k].shape)))
            continue
        out[k] = v

    if verbose:
        print(f"  matched keys (name+shape): {len(out)}")
        print(f"  skipped (key not in model): {skipped_missing}")
        if skipped_shape:
            print(f"  skipped (shape mismatch): {len(skipped_shape)}")
            for k, a, b in skipped_shape[:6]:
                print(f"   - {k}: ckpt{a} vs model{b}")
            if len(skipped_shape) > 6:
                print("   ...")

    return out


def load_checkpoint_any(
    model: nn.Module,
    path: Optional[str],
    *,
    model_type: str,
    strict: bool = False,
    verbose: bool = True,
    min_loaded_ratio_warn: float = 0.30,
) -> None:
    if not path or str(path).lower() == "none":
        if verbose:
            print(f"[{model_type}] No checkpoint provided. Using random init.")
        return
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    if verbose:
        print(f"[{model_type}] Loading checkpoint: {path}")
    
    if str(path).endswith(".safetensors"):
        sd = load_file(path)
    else:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)

        # Special-case: DN3/BENDR checkpoints often split encoder/contextualizer.
        if (
            model_type == "bendr"
            and isinstance(ckpt, dict)
            and "encoder_state_dict" in ckpt
            and "contextualizer_state_dict" in ckpt
            and hasattr(model, "encoder")
            and hasattr(model, "contextualizer")
        ):
            enc_sd = ckpt["encoder_state_dict"]
            ctx_sd = ckpt["contextualizer_state_dict"]

            if not isinstance(enc_sd, dict) or not isinstance(ctx_sd, dict):
                raise ValueError("[bendr] encoder_state_dict/contextualizer_state_dict must be dicts")

            enc_sd = strip_prefixes(enc_sd, prefixes=("module.", "model.", "state_dict.", "net."))
            ctx_sd = strip_prefixes(ctx_sd, prefixes=("module.", "model.", "state_dict.", "net."))

            enc_sd = filter_by_shape(enc_sd, model.encoder, verbose=verbose)
            ctx_sd = filter_by_shape(ctx_sd, model.contextualizer, verbose=verbose)

            ratio_enc = len(enc_sd) / max(1, len(model.encoder.state_dict()))
            ratio_ctx = len(ctx_sd) / max(1, len(model.contextualizer.state_dict()))
            if verbose:
                print(f"   encoder loaded ratio: {ratio_enc:.1%}")
                print(f"   contextualizer loaded ratio: {ratio_ctx:.1%}")

            msg_enc = model.encoder.load_state_dict(enc_sd, strict=strict)
            msg_ctx = model.contextualizer.load_state_dict(ctx_sd, strict=strict)
            if verbose:
                print(f"[bendr] encoder load_state_dict(strict={strict}) => {msg_enc}")
                print(f"[bendr] contextualizer load_state_dict(strict={strict}) => {msg_ctx}")
            return

        sd = unwrap_state_dict(ckpt)

    sd = strip_prefixes(sd, prefixes=("module.", "model.", "state_dict.", "net.", "backbone."))
    sd = replace_prefix(sd, old="student.", new="")

    if model_type == "labram":
        sd = drop_if_contains(sd, ("head", "projection_head", "lm_head"))
        if "pos_embed" in sd and hasattr(model, "pos_embed"):
            if tuple(sd["pos_embed"].shape) != tuple(model.pos_embed.shape):
                if verbose:
                    print(f"  ⚠️ skip pos_embed mismatch: ckpt{tuple(sd['pos_embed'].shape)} vs model{tuple(model.pos_embed.shape)}")
                sd.pop("pos_embed", None)

    elif model_type == "biot":
        sd = drop_if_contains(sd, ("classifier", "fc", "head"))
        if not any(k.startswith("biot.") for k in sd.keys()):
            sd = {("biot." + k): v for k, v in sd.items()}

    sd = filter_by_shape(sd, model, verbose=verbose)

    ratio = len(sd) / max(1, len(model.state_dict()))
    if verbose:
        print(f"   loaded ratio: {ratio:.1%}")
    if ratio < min_loaded_ratio_warn and verbose:
        print("   WARNING: loaded ratio is low; checkpoint may not match model config.")

    msg = model.load_state_dict(sd, strict=strict)
    if verbose:
        print(f"[{model_type}] load_state_dict(strict={strict}) => {msg}")

def load_brant_pretrained(
        model: Brant,
        time_ckpt: str,
        channel_ckpt: Optional[str] = None,
        strict: bool = False,
    ):
        # ---- TimeEncoder ----
        time_sd = torch.load(time_ckpt, map_location="cpu")
        time_sd = unwrap_state_dict(time_sd)
        time_sd = strip_prefixes(
            time_sd,
            prefixes=("module.", "model.", "encoder.", "time.", "time_encoder.")
        )
        time_sd = filter_by_shape(time_sd, model.time, verbose=True)
        msg_t = model.time.load_state_dict(time_sd, strict=strict)
        print(f"[brant] time_encoder load => {msg_t}")

        # ---- ChannelEncoder ----
        if model.channel is not None and channel_ckpt is not None:
            ch_sd = torch.load(channel_ckpt, map_location="cpu")
            ch_sd = unwrap_state_dict(ch_sd)
            ch_sd = strip_prefixes(
                ch_sd,
                prefixes=("module.", "model.", "encoder.", "channel.", "channel_encoder.")
            )
            ch_sd = filter_by_shape(ch_sd, model.channel, verbose=True)
            msg_c = model.channel.load_state_dict(ch_sd, strict=strict)
            print(f"[brant] channel_encoder load => {msg_c}")

# norm method
class PerTrialMinMax(nn.Module):
    """x: [B, C, T], min-max over T for each (B, C), scaled to [-1, 1]."""
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_min = x.amin(dim=-1, keepdim=True)
        x_max = x.amax(dim=-1, keepdim=True)
        x01 = (x - x_min) / (x_max - x_min + self.eps)
        return 2.0 * x01 - 1.0
class PerTrialP95AbsScale(nn.Module):
    """x: [B, C, T], divide each (B, C) by P95(|x|) over T."""
    def __init__(self, q: float = 0.95, eps: float = 1e-6):
        super().__init__()
        self.q = q
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.quantile(x.abs(), q=self.q, dim=-1, keepdim=True)
        return x / (scale + self.eps)
class FixedScaleTo01mV(nn.Module):
    """x: [B, C, T], input unit is µV, scaling by 0.1 mV means divide by 100."""
    def __init__(self, scale: float = 100.0):
        super().__init__()
        self.scale = float(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / self.scale
class PerTrialIQRNorm(nn.Module):
    """x: [B, C, T], robust scaling over T for each (B, C)."""
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q25 = torch.quantile(x, 0.25, dim=-1, keepdim=True)
        q50 = torch.quantile(x, 0.50, dim=-1, keepdim=True)
        q75 = torch.quantile(x, 0.75, dim=-1, keepdim=True)
        return (x - q50) / (q75 - q25 + self.eps)

# ============================================================
# factory main (ALL lowercase)
# ============================================================
def get_model(cfg: DictConfig) -> nn.Module:
    """
    Expects cfg fields (lowercase):
      cfg.model.name         (single model, set by main loop)
      cfg.dataset.num_classes
      cfg.dataset.chn_names
      cfg.model.patch_size
      cfg.paths.labram / cfg.paths.biot / cfg.paths.cbramod (as needed)
      cfg.train.tuning_mode
      cfg.train.use_zscore
      cfg.model.head.*
    """
    model_name, model_variant = _requested_model_and_variant(cfg)
    
    dataset_json = load_dataset_metadata(cfg)
    
    num_classes = dataset_json['dataset']['num_labels']
    chn_names = dataset_json['dataset']['channels']
    
    
    patch_size = int(_require(cfg, "model.patch_size"))
    print(f"patch_size: {patch_size}")
    input_sampling_rate = int(round(float(dataset_json['processing']['target_sampling_rate'])))
    time_points = dataset_json['processing']['window_sec'] * dataset_json['processing']['target_sampling_rate']
    use_zscore = bool(_require(cfg, "train.use_zscore"))
    interpolation_mode = str(OmegaConf.select(cfg, "model.interpolation.mode") or "linear")
    interpolation_align_corners = bool(OmegaConf.select(cfg, "model.interpolation.align_corners") or False)
    #norm = PerTrialZScore() if use_zscore else IdentityNorm()
    
    
    # ----- EEGNet -----
    if model_name == "eegnet":
        num_ch = len(chn_names)
        norm = PerTrialZScore()
        extractor = EEGNet(chans=num_ch,time_point=time_points,norm=norm)
    
    # ----- EEGConformer -----   
    elif model_name == "eegconformer":
        num_ch = len(chn_names)
        norm = PerTrialZScore()
        extractor = Conformer(C=num_ch, time_points=time_points, n_classes=None, norm=norm)
    
    # ----- LaBraM -----
    elif model_name == "labram":
        ckpt_path = str(_require(cfg, "paths.labram"))
        input_chans,ch_use = map_channels_to_indices(chn_names)
        num_ch = int(np.asarray(ch_use).sum())
        if num_ch <= 0:
            raise ValueError("LaBraM nearest-channel adapter found no usable coordinate-mapped input channels.")
        norm = FixedScaleTo01mV()
        raw = create_model(
            "labram_base_patch200_200",
            pretrained=False,
            num_classes=num_classes,
            init_values=0.1,
            num_ch=num_ch,
        )
        load_checkpoint_any(raw, ckpt_path, model_type="labram", strict=False)

        extractor = LaBraMFeatureExtractor(
            raw,
            pool="mean",
            patch_size=patch_size,
            norm=norm,
            input_chans=input_chans,
            ch_use=ch_use
        )

    # ----- BIOT -----
    elif model_name == "biot":
        ckpt_path = str(_require(cfg, "paths.biot"))
        in_ch = len(chn_names)
        hop = int(_require(cfg, "model.biot_hop"))
        strict_biot18 = model_variant in {"official", "official_biot18", "strict"}
        if OmegaConf.select(cfg, "model.biot.strict_biot18") is not None:
            strict_biot18 = bool(OmegaConf.select(cfg, "model.biot.strict_biot18"))

        raw = BIOTClassifier(
            n_classes=num_classes,
            n_channels=18,
            n_fft=patch_size,
            hop_length=hop,
        )
        load_checkpoint_any(raw, ckpt_path, model_type="biot", strict=False)

        norm = PerTrialP95AbsScale()
        
        extractor = BIOTFeatureExtractor(
            raw,
            norm=norm,
            expected_channels=18,
            strict_channels=strict_biot18,
            ch_names=chn_names,
            require_referential_names=True,
        )

    # ----- CBraMod -----
    elif model_name == "cbramod":
        ckpt_path = str(_require(cfg, "paths.cbramod"))
        raw = CBraMod()
        load_checkpoint_any(raw, ckpt_path, model_type="cbramod", strict=False)

        norm = FixedScaleTo01mV()
        debug_input_range = OmegaConf.select(cfg, "model.cbramod.debug_input_range")
        cbramod_order, cbramod_mapped_names, cbramod_missing = map_channels_to_spatial_order(chn_names)
        if cbramod_missing:
            print(
                f"⚠️ Warning: CBraMod nearest-channel adapter dropped {len(cbramod_missing)} channels "
                f"without montage coordinates: {cbramod_missing[:5]}..."
            )
        cbramod_target_time = int(round(float(time_points) * 200.0 / float(input_sampling_rate)))
        cbramod_num_channels = len(cbramod_order)
        cbramod_num_patches = cbramod_target_time // 200  # patch_size=200
        extractor = CBraModFeatureExtractor(
            raw,
            patch_size=patch_size,
            norm=norm,
            input_freq=input_sampling_rate,
            target_freq=200,
            interpolation_mode=interpolation_mode,
            interpolation_align_corners=interpolation_align_corners,
            debug_input_range=True if debug_input_range is None else bool(debug_input_range),
            ch_names=chn_names,
            channel_order_indices=cbramod_order,
            mapped_channel_names=cbramod_mapped_names,
            num_channels=cbramod_num_channels,
            num_patches=cbramod_num_patches,
        )

    # ----- REVE -----
    elif model_name in {"reve_old", "reve", "reve_new"}:
        try:
            from transformers import AutoModel
        except ImportError as e:
            raise ImportError("REVE requires `transformers`. Please install transformers in the 5.0 environment.") from e

        reve_model_id = str(OmegaConf.select(cfg, "paths.reve") )
        reve_pos_id = str(OmegaConf.select(cfg, "paths.reve_positions"))
        reve_cfg_key = "reve_old" if model_name == "reve_old" else model_name
        use_official_positions = bool(OmegaConf.select(cfg, f"model.{reve_cfg_key}.use_official_positions") if OmegaConf.select(cfg, f"model.{reve_cfg_key}.use_official_positions") is not None else True)
        reve_norm = str(OmegaConf.select(cfg, f"model.{reve_cfg_key}.norm") or OmegaConf.select(cfg, "model.reve.norm") or ("zscore" if use_zscore else "identity")).lower()
        if reve_norm == "zscore":
            norm = PerTrialZScore()
        else:
            norm = IdentityNorm()

        pos_bank = AutoModel.from_pretrained(reve_pos_id, trust_remote_code=True) if use_official_positions else None
        raw, reve_load_info = AutoModel.from_pretrained(
            reve_model_id,
            trust_remote_code=True,
            output_loading_info=True,
        )
        missing_keys = list(reve_load_info.get("missing_keys", []) or [])
        unexpected_keys = list(reve_load_info.get("unexpected_keys", []) or [])
        mismatched_keys = list(reve_load_info.get("mismatched_keys", []) or [])
        print(
            f"[reve] checkpoint loaded from {reve_model_id}: "
            f"missing={len(missing_keys)} unexpected={len(unexpected_keys)} mismatched={len(mismatched_keys)}"
        )
        local_position_missing = []
        if not use_official_positions and model_name == "reve_old":
            coords = _build_standard_coords_norm()
            local_position_missing = [
                name for name in chn_names
                if _normalize_eeg_coord_name(name) not in coords
            ]
            print(
                "[reve] use_official_positions=false; using local standard_coords.pt positions "
                f"({len(local_position_missing)}/{len(chn_names)} channels missing)."
            )

        reve_positions_metadata = reve_pos_id if use_official_positions else "data/standard_coords.pt"
        if model_name in {"reve", "reve_new"}:
            montage_path = str(OmegaConf.select(cfg, f"model.{reve_cfg_key}.montage_path") or "/benchmark-eeg/cx/montage.json")
            vocab_path = str(OmegaConf.select(cfg, f"model.{reve_cfg_key}.electrode_vocab_path") or "/benchmark-eeg/cx/electrode_vocab.json")
            reve_positions_metadata = {
                "official": reve_pos_id if use_official_positions else None,
                "fallback_montage": montage_path,
                "fallback_vocab": vocab_path,
            }
            reve_patch_size = int(getattr(raw, "patch_size", getattr(getattr(raw, "config", None), "patch_size", 200)))
            reve_patch_overlap = int(getattr(raw, "patch_overlap", getattr(getattr(raw, "config", None), "patch_overlap", 20)))
            reve_stride = max(1, reve_patch_size - reve_patch_overlap)
            target_time_points = int(round(float(time_points) * 200.0 / float(input_sampling_rate)))
            padded_time_points = max(target_time_points, reve_patch_size)
            remainder = (padded_time_points - reve_patch_size) % reve_stride
            if remainder:
                padded_time_points += reve_stride - remainder
            inferred_max_patches = 1 + ((padded_time_points - reve_patch_size) // reve_stride)
            extractor = REVENewFeatureExtractor(
                raw,
                electrode_names=chn_names,
                pos_bank=pos_bank,
                norm=norm,
                input_freq=input_sampling_rate,
                target_freq=200,
                interpolation_mode=interpolation_mode,
                interpolation_align_corners=interpolation_align_corners,
                use_official_positions=use_official_positions,
                montage_path=montage_path,
                vocab_path=vocab_path,
                max_patches=OmegaConf.select(cfg, f"model.{reve_cfg_key}.max_patches") or inferred_max_patches,
            )
        else:
            extractor = REVEFeatureExtractor(
                raw,
                electrode_names=chn_names,
                pos_bank=pos_bank,
                norm=norm,
                input_freq=input_sampling_rate,
                target_freq=200,
                interpolation_mode=interpolation_mode,
                interpolation_align_corners=interpolation_align_corners,
                use_official_positions=use_official_positions,
            )
        extractor.benchmark_metadata = {
            **dict(getattr(extractor, "benchmark_metadata", {}) or {}),
            "implementation": "hf_remote_code_fm_lp" if model_name in {"reve", "reve_new"} else "hf_remote_code",
            "checkpoint": reve_model_id,
            "positions": reve_positions_metadata,
            "use_official_positions": use_official_positions,
            "norm": reve_norm,
            "input_sampling_rate": input_sampling_rate,
            "target_sampling_rate": 200,
            "num_channels": len(chn_names),
            "missing_position_channels": (
                list(getattr(extractor, "missing_position_channels", []))
                if model_name in {"reve", "reve_new"}
                else local_position_missing
            ),
            "load_missing_keys": len(missing_keys),
            "load_unexpected_keys": len(unexpected_keys),
            "load_mismatched_keys": len(mismatched_keys),
        }

    elif model_name == "brainomni":
        ckpt_path = str(_require(cfg, "paths.brainomni"))
        cfg_path = str(_require(cfg, "paths.brainomni_config"))
        static_pos, static_sensor = map_channels_to_pos_and_type(chn_names)
        known_mask = static_pos.abs().sum(dim=1) > 0
        known_ratio = float(known_mask.float().mean().item())
        if known_ratio <= 0.0:
            print(
                "⚠️ Warning: BrainOmni could not map any dataset channels to known coordinates "
                "(checked standard_coords.pt and montage fallback). "
                "Proceeding with zero-filled positions / default sensor types."
            )
        elif known_ratio < 1.0:
            print(
                f"⚠️ Warning: BrainOmni only matched {known_ratio:.1%} of channels to known coordinates "
                "(standard_coords.pt + montage fallback). Some positions may differ from training distribution."
            )

        def get_brainomni_model(ckpt_path,cfg_path):
            with open(cfg_path, 'r') as f:
                    cfg = json.load(f)
            model = BrainOmni(**cfg)
            pt_path = ckpt_path
            if os.path.exists(pt_path):
                model.load_state_dict(torch.load(pt_path, map_location="cpu", weights_only=True), strict=False)
            return model, cfg
        
        raw, brain_cfg = get_brainomni_model(ckpt_path, cfg_path)
        feature_dim = brain_cfg.get("lm_dim", 512)
        n_neuro = brain_cfg.get("n_neuro", None)
        
        norm = PerTrialZScore()
        
        extractor = BrainOmniFeatureExtractor(
            model=raw,
            static_pos=static_pos,     
            static_sensor_type=static_sensor, 
            feature_dim=feature_dim,
            output_num_tokens=n_neuro,
            norm=norm,
            input_freq=input_sampling_rate,
            target_freq=256,
            interpolation_mode=interpolation_mode,
            interpolation_align_corners=interpolation_align_corners,
        )

    # ----- FEMBA -----
    elif model_name == "femba":
        if model_variant not in {"adapter", "femba_adapter"}:
            print(f"[FEMBA] variant={model_variant!r} is not implemented in this checkpoint path; falling back to femba_adapter.")
        ckpt_path = str(_require(cfg, "paths.femba"))
        raw = FEMBA(seq_length=1280, num_channels=22, num_classes=num_classes, embed_dim=35, num_blocks=4)
        load_checkpoint_any(raw, ckpt_path, model_type="femba", strict=False)

        norm = PerTrialIQRNorm()

        extractor = FembaFeatureExtractor(
            model=raw,
            input_freq=200, 
            input_chn_names = chn_names,
            norm=norm,
            interpolation_mode=interpolation_mode,
            interpolation_align_corners=interpolation_align_corners,
            strict_channel_names=bool(OmegaConf.select(cfg, "model.femba.strict_channel_names") or False),
        )
    
    # -----   NeuroGPT -----
    elif model_name == "neurogpt":
        if model_variant not in {"encoder_adapter", "adapter", "neurogpt_encoder_adapter"}:
            print(f"[NeuroGPT] variant={model_variant!r} is not implemented; falling back to neurogpt_encoder_adapter.")
        ckpt_path = str(_require(cfg, "paths.neurogpt"))

        # EEGConformer encoder: 22ch, 2s chunks at 250 Hz (500 samples)
        # Verified from checkpoint: shallownet.1.weight=(40,40,22,1), embedder Linear(1080→1024)
        # where 1080 = 27 patches × 40 dims (27 = AvgPool output for n_times=500)
        encoder = EEGConformer(
            n_outputs=num_classes,
            n_chans=22,
            n_times=500,
            n_filters_time=40,
            filter_time_length=25,
            pool_time_length=75,
            pool_time_stride=15,
            att_depth=6,
            att_heads=10,
            is_decoding_mode=False,
        )

        # GPT decoder: n_layer=6, n_embd=1024, block_size=512 (from checkpoint wpe shape)
        decoder = NeuroGPTDecoder(
            n_embd=1024,
            n_layer=6,
            n_head=16,
            block_size=512,
            bias=True,
            dropout=0.0,
        )

        # Embedder: Linear(1080→1024) + LayerNorm, maps encoder output to GPT token dim
        embedder = nn.Sequential(
            nn.Linear(1080, 1024, bias=True),
            nn.LayerNorm(1024),
        )

        full_sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # Load encoder weights
        encoder_sd = {k[len("encoder."):]: v for k, v in full_sd.items() if k.startswith("encoder.")}
        encoder.load_state_dict(encoder_sd, strict=False)
        print(f"[NeuroGPT] encoder loaded ({len(encoder_sd)} keys)")

        # Load embedder weights (embed_model.model.0=Linear, .1=LayerNorm)
        embedder_sd = {
            "0.weight": full_sd["embedder.embed_model.model.0.weight"],
            "0.bias":   full_sd["embedder.embed_model.model.0.bias"],
            "1.weight": full_sd["embedder.embed_model.model.1.weight"],
            "1.bias":   full_sd["embedder.embed_model.model.1.bias"],
        }
        embedder.load_state_dict(embedder_sd, strict=True)
        print("[NeuroGPT] embedder loaded (4 keys)")

        # Load GPT decoder weights: transformer.{wpe,h.*,ln_f} + pooler_layer
        # The checkpoint was saved with GPT-2 Conv1D convention: weights are (in, out)
        # instead of PyTorch nn.Linear convention (out, in), so non-square projection
        # weights for c_attn / mlp.c_fc / mlp.c_proj must be transposed.
        _CONV1D_SUFFIXES = ("attn.c_attn.weight", "mlp.c_fc.weight", "mlp.c_proj.weight")
        decoder_sd = {}
        for k, v in full_sd.items():
            if k.startswith("decoder.transformer.") and not k.startswith("decoder.transformer.wte"):
                new_k = k[len("decoder.transformer."):]
                if any(new_k.endswith(s) for s in _CONV1D_SUFFIXES):
                    v = v.t().contiguous()
                decoder_sd[new_k] = v
            elif k.startswith("decoder.pooler_layer."):
                decoder_sd[k[len("decoder."):]] = v
        msg = decoder.load_state_dict(decoder_sd, strict=False)
        print(f"[NeuroGPT] decoder loaded ({len(decoder_sd)} keys); missing={msg.missing_keys[:4]}")

        norm = PerTrialZScore()

        extractor = NeuroGPTFeatureExtractor(
            encoder,
            decoder=decoder,
            embedder=embedder,
            input_chns=chn_names,
            embed_dim=1024,
            chunk_len=500,
            stride=500,
            input_freq=input_sampling_rate,
            norm=norm,
            interpolation_mode=interpolation_mode,
            interpolation_align_corners=interpolation_align_corners,
            strict_channel_names=bool(OmegaConf.select(cfg, "model.neurogpt.strict_channel_names") or False),
        )
    

    # ----- NeuroLM -----
    elif model_name == "neurolm":
        if model_variant not in {"tokenizer_probe", "adapter", "neurolm_tokenizer_probe", "full_gpt", "gpt", "neurolm_full_gpt", "full"}:
            print(f"[NeuroLM] variant={model_variant!r} is not implemented; falling back to neurolm_tokenizer_probe.")
            model_variant = "tokenizer_probe"
        ckpt_path = str(_require(cfg, "paths.neurolm"))
        ckpt_path_vq  = str(_require(cfg, "paths.neurolm_vq")) 

        # GPT config must match pretraining
        gpt_cfg = GPTConfig(
            block_size=1024,
            vocab_size=50304,
            n_layer=12,
            n_head=12,
            n_embd=768,
            dropout=0.0,
            bias=True,
        )

        raw = NeuroLM(
            GPT_config=gpt_cfg,
            init_from='scratch', # 'gpt2' / 'gpt2-medium' / 'scratch'
            tokenizer_ckpt_path=ckpt_path_vq,
        )

        load_checkpoint_any(raw, ckpt_path, model_type="neurolm", strict=False)

        # target_samples = int(
        #     OmegaConf.select(cfg, "model.neurolm.target_samples")
        #     or OmegaConf.select(cfg, "dataset.samples")
        # )
        norm = FixedScaleTo01mV()
        
        extractor = NeuroLMFeatureExtractor(
            model=raw,
            # target_samples=target_samples,
            pool=str(OmegaConf.select(cfg, "model.neurolm.pool") or "mean"),
            norm=norm,
            standard_1020 = STANDARD_1020,
            input_chn_names = chn_names,
            pad_missing = bool(OmegaConf.select(cfg, "model.neurolm.pad_missing") if OmegaConf.select(cfg, "model.neurolm.pad_missing") is not None else True),
            target_chn_names=OmegaConf.select(cfg, "model.neurolm.target_chn_names"),
            variant=model_variant,
        )


    # ----- EEGMamba -----
    elif model_name == "eegmamba":
        if model_variant not in {"19ch_adapter", "adapter", "eegmamba_19ch_adapter"}:
            print(f"[EEGMamba] variant={model_variant!r} is not implemented in this checkpoint path; falling back to eegmamba_19ch_adapter.")
        ckpt_path = str(_require(cfg, "paths.eegmamba"))

        patch_size = int(_require(cfg, "model.patch_size"))
        target_samples = int(
            OmegaConf.select(cfg, "model.eegmamba.target_samples") or 6000
        )

        seq_len = target_samples // patch_size
        # if seq_len != 30:
        #     raise ValueError(
        #         "[factory] EEGMamba pretrained with seq_len=30. "
        #         f"Got target_samples={target_samples}."
        #     )
        sd=torch.load(ckpt_path, map_location='cpu', weights_only=True)
        raw = EEGMamba()
        raw.load_state_dict(sd,strict=True)
        # load_checkpoint_any(raw, ckpt_path, model_type="eegmamba", strict=False)

        norm = IdentityNorm()
        CANONICAL_ORDER = [
            "FP1", "FP2",
            "F7", "F3", "FZ", "F4", "F8",
            "T3", "C3", "CZ", "C4", "T4",
            "T5", "P3", "PZ", "P4", "T6",
            "O1", "O2",
        ]
        extractor = EEGMambaFeatureExtractor(
            model=raw,
            patch_size=patch_size,
            target_samples=target_samples,
            norm=norm,
            ch_name = chn_names,
            canonical_order=CANONICAL_ORDER,
            crop_mode=str(OmegaConf.select(cfg, "model.eegmamba.crop_mode") or "center_crop"),
            pad_mode=str(OmegaConf.select(cfg, "model.eegmamba.pad_mode") or "zero"),
            strict=bool(OmegaConf.select(cfg, "model.eegmamba.strict_channels") if OmegaConf.select(cfg, "model.eegmamba.strict_channels") is not None else False),
        )


    # ----- BENDR -----
    elif model_name == "bendr":
        learned_variants = {"learned_channel_adapter", "learned_adapter", "conv_adapter", "bendr_learned_channel_adapter"}
        hard_variants = {"19plus1_adapter", "adapter", "bendr_19plus1_adapter"}
        if model_variant not in learned_variants | hard_variants:
            print(f"[BENDR] variant={model_variant} is not implemented; falling back to learned_channel_adapter.")
            model_variant = "learned_channel_adapter"
        ckpt_path = str(_require(cfg, "paths.bendr"))

        expected_samples = 6000 # BENDR was trained at samples 200 * 30
        samples = int(expected_samples)
        channels = len(chn_names)

        raw = BENDRClassification(
            targets=num_classes,
            samples=samples,
            channels=20,
            encoder_h=512,
            contextualizer_hidden=3076,
            projection_head=False,
            dropout=0.0,
            layer_drop=0.0,
        )

        load_checkpoint_any(raw, ckpt_path, model_type="bendr", strict=False)

        if model_variant in learned_variants:
            extractor = BENDRLearnedChannelAdapterFeatureExtractor(
                model=raw,
                in_channels=channels,
                input_channel_names=chn_names,
                norm=PerTrialMinMax(),
                input_freq=input_sampling_rate,
                interpolation_mode=interpolation_mode,
                interpolation_align_corners=interpolation_align_corners,
            )
        else:
            # BENDRFeatureExtractor already applies BENDR-style min-max scaling before
            # adding the relative-amplitude channel. A second norm pass corrupts zero-filled
            # missing channels and the constant relative-amplitude channel.
            extractor = BENDRFeatureExtractor(
                model=raw,
                norm=IdentityNorm(),
                ch_names=chn_names,
                strict_channels=bool(OmegaConf.select(cfg, "model.bendr.strict_channels") if OmegaConf.select(cfg, "model.bendr.strict_channels") is not None else False),
                input_freq=input_sampling_rate,
                interpolation_mode=interpolation_mode,
                interpolation_align_corners=interpolation_align_corners,
            )

    # ----- Brant -----
    elif model_name == "brant":
        print("[brant] WARNING: Brant is an iEEG model and is kept only for reference; it is excluded from the default scalp-EEG model list.")
        # Brant pretrained settings are encoded in the checkpoint tensor shapes.
        # To ensure alignment, infer them from the checkpoint instead of hardcoding.
        time_ckpt = str(_require(cfg, "paths.brant_time"))
        ch_ckpt = OmegaConf.select(cfg, "paths.brant_channel")

        def _infer_brant_time_spec(path: str):
            sd = torch.load(path, map_location="cpu", weights_only=False)
            # checkpoints are stored as an OrderedDict state_dict
            pos = sd["module.input_embedding.positional_encoding"]
            proj_w = sd["module.input_embedding.proj.0.weight"]
            band = sd["module.input_embedding.band_encoding"]
            ff_w = sd["module.trans_enc.layers.0.linear1.weight"]
            # count encoder layers
            layer_ids = set()
            for k in sd.keys():
                if "module.trans_enc.layers." in k:
                    try:
                        layer_ids.add(int(k.split("module.trans_enc.layers.", 1)[1].split(".", 1)[0]))
                    except Exception:
                        pass
            n_layer = (max(layer_ids) + 1) if layer_ids else 0

            return {
                "seq_len": int(pos.shape[0]),
                "d_model": int(pos.shape[1]),
                "in_dim": int(proj_w.shape[1]),
                "band_num": int(band.shape[0]),
                "dim_feedforward": int(ff_w.shape[0]),
                "n_layer_time": int(n_layer),
            }

        def _infer_brant_channel_spec(path: str):
            sd = torch.load(path, map_location="cpu", weights_only=False)
            # count encoder layers
            layer_ids = set()
            for k in sd.keys():
                if "module.trans_enc.layers." in k:
                    try:
                        layer_ids.add(int(k.split("module.trans_enc.layers.", 1)[1].split(".", 1)[0]))
                    except Exception:
                        pass
            n_layer = (max(layer_ids) + 1) if layer_ids else 0
            out_w = sd.get("module.proj_out.0.weight", None)
            out_dim = int(out_w.shape[0]) if isinstance(out_w, torch.Tensor) and out_w.ndim == 2 else None
            return {"n_layer_ch": int(n_layer), "ch_out_dim": out_dim}

        spec_t = _infer_brant_time_spec(time_ckpt)
        spec_c = _infer_brant_channel_spec(str(ch_ckpt)) if ch_ckpt is not None else {"n_layer_ch": 0, "ch_out_dim": None}

        # Heads aren't inferable from state_dict shapes; keep config default (must divide d_model).
        nhead_time = int(OmegaConf.select(cfg, "model.brant_nhead_time") or 8)
        nhead_ch = int(OmegaConf.select(cfg, "model.brant_nhead_ch") or 8)

        # Informative warning if the user-provided patch_size/seq_len differ from the pretrained spec.
        cfg_patch = int(_require(cfg, "model.patch_size"))
        if cfg_patch != spec_t["in_dim"]:
            print(
                f"[brant] WARNING: cfg.model.patch_size={cfg_patch} but Brant checkpoint expects in_dim={spec_t['in_dim']}. "
                "Using checkpoint in_dim for Brant."
            )
        cfg_seq = OmegaConf.select(cfg, "model.brant_seq_len")
        if cfg_seq is not None and int(cfg_seq) != spec_t["seq_len"]:
            print(
                f"[brant] WARNING: cfg.model.brant_seq_len={int(cfg_seq)} but checkpoint expects seq_len={spec_t['seq_len']}. "
                "Using checkpoint seq_len for Brant."
            )

        # Brant expects input windows of length T = seq_len * in_dim.
        # The feature extractor will crop/pad to this length; if your dataset windows are much shorter,
        # this may heavily pad with zeros and degrade performance.
        target_samples = int(spec_t["seq_len"] * spec_t["in_dim"])
        print(f"[brant] pretrained spec: seq_len={spec_t['seq_len']} in_dim={spec_t['in_dim']} => target_samples={target_samples}")

        raw = Brant(
            in_dim=spec_t["in_dim"],
            seq_len=spec_t["seq_len"],
            d_model=spec_t["d_model"],
            dim_feedforward=spec_t["dim_feedforward"],
            n_layer_time=spec_t["n_layer_time"],
            nhead_time=nhead_time,
            band_num=spec_t["band_num"],
            project_mode="linear",
            learnable_mask=False,
            use_channel_encoder=True,
            n_layer_ch=int(spec_c["n_layer_ch"] or 0),
            nhead_ch=nhead_ch,
            ch_out_dim=int(spec_c["ch_out_dim"] or spec_t["in_dim"]),
        )

        load_brant_pretrained(
            model=raw,
            time_ckpt=time_ckpt,
            channel_ckpt=ch_ckpt,
            strict=False,
        )

        norm = IdentityNorm()  # Brant was pretrained with no input normalization; the model expects raw-like inputs in µV scale.
        
        extractor = BrantPretrainFeatureExtractor(
            model=raw,
            seg_len=spec_t["in_dim"],
            seq_len=spec_t["seq_len"],
            band_num=spec_t["band_num"],
            use_power=bool(OmegaConf.select(cfg, "model.brant_use_power") or False),
            norm=norm,
            interpolation_mode=interpolation_mode,
            interpolation_align_corners=interpolation_align_corners,
        )

    
    else:
        raise ValueError(f"Unknown model.name: {model_name}")
    
    

    # ----- head -----
    head_type = str(_require(cfg, "model.head.type"))
    hidden = OmegaConf.select(cfg, "model.head.hidden")
    dropout = float(_require(cfg, "model.head.dropout"))
    act = str(_require(cfg, "model.head.act"))

    head = ProbeHead(
        in_dim=int(extractor.embed_dim),
        out_dim=num_classes,
        head_type=head_type,
        hidden_dim=(int(hidden) if hidden is not None else None),
        dropout=dropout,
        act=act,
    )

    model = DownstreamModel(extractor, head)
    model.benchmark_metadata = dict(getattr(extractor, "benchmark_metadata", {}) or {})
    model.benchmark_metadata.setdefault("requested_name", str(_require(cfg, "model.name")).lower())
    model.benchmark_metadata.setdefault("base_model", model_name)
    model.benchmark_metadata.setdefault("variant", model_variant)
    return model
