import os
import sys
import copy
import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader
from omegaconf import OmegaConf, DictConfig, ListConfig
import json
import random
from datetime import datetime

from configs.config import load_config
from data.dataset import EEGIndexDataset
from data.zarr_dataset import ZarrSplitDataset
from data.zarr_split import build_zarr_split_manifest_from_cfg
from leaderboard_utils import rebuild_leaderboards_from_summaries
from models.factory import get_model
from engine import train_one_epoch, evaluate, evaluate_embed_nn
from utils import NativeScalerWithGradNormCount as NativeScaler
from tuning import setup_training_mode
from data.make_indices_fixed import generate_indices
from data.make_indices_downsample_fixed import generate_indices_downsample
from data.metadata import load_dataset_metadata

CHECKPOINT_URLS = {
    "brainomni": "https://huggingface.co/OpenTSLab/BrainOmni",
    "brant": "https://huggingface.co/Daoze/Brant/tree/main",
    "cbramod": "https://huggingface.co/weighting666/CBraMod",
    "eegmamba": "https://huggingface.co/weighting666/EEGMamba",
    "femba": "https://huggingface.co/thorir/FEMBA",
    "labram": "https://huggingface.co/braindecode/Labram-Braindecode",
    "neurogpt": "https://huggingface.co/wenhuic/Neuro-GPT/tree/main",
    "neurolm": "https://huggingface.co/Weibang/NeuroLM",
    "bendr": "https://github.com/SPOClab-ca/BENDR/releases/tag/v0.1-alpha",
    "biot": "https://github.com/ycq091044/BIOT",
}

def _resolve_device(cfg: DictConfig) -> torch.device:
    # prioritize cfg.cuda
    if OmegaConf.select(cfg, "cuda") is not None:
        cuda_id = int(cfg.cuda)
        if cuda_id >= 0 and torch.cuda.is_available():
            torch.cuda.set_device(cuda_id)
            return torch.device(f"cuda:{cuda_id}")
        return torch.device("cpu")

    # fallback: cfg.device if exists
    dev = OmegaConf.select(cfg, "device")
    if dev is not None:
        dev = str(dev).lower()
        if dev.startswith("cuda") and torch.cuda.is_available():
            if ":" in dev:
                torch.cuda.set_device(int(dev.split(":")[-1]))
            return torch.device(dev)
        return torch.device("cpu")

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _set_seed(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(seed)


def _ratio_to_tag(r: float) -> str:
    """
    0.02 -> '0p02'
    0.1  -> '0p1'
    """
    s = str(float(r))
    return s.replace(".", "p")


def _select_train_index_file(cfg: DictConfig) -> str:
    """
    Use cfg.train.ratio_shot:
      - null -> train_idx.txt
      - 0.02/0.05/0.1/0.3/0.5 -> train_idx_ratio_0p02.txt ...
    """
    base_dir = cfg.paths.data_index_dir
    ratio = OmegaConf.select(cfg, "train.ratio_shot")

    if ratio is None:
        fp = os.path.join(base_dir, "train_idx.txt")
        if not os.path.exists(fp):
            raise FileNotFoundError(f"train_idx not found: {fp}")
        return fp

    tag = _ratio_to_tag(float(ratio))
    fp = os.path.join(base_dir, f"train_idx_ratio_{tag}.txt")
    if not os.path.exists(fp):
        raise FileNotFoundError(f"train_idx_ratio not found: {fp} (ratio_shot={ratio})")
    return fp


def _read_index_file_as_ints(path: str) -> list[int]:
    with open(path, "r", encoding="utf-8") as f:
        return [int(line.strip().split(",")[0]) for line in f if line.strip()]


def _normalize_channel_name_for_mask(name: str) -> str:
    key = str(name).strip().upper().replace(".", "").replace("_", "_")
    while "  " in key:
        key = key.replace("  ", " ")
    if key.startswith("EEG "):
        key = key[4:].strip()
    for suffix in ("-REF", "-LE", "-AR", "-AVG", "-A1", "-A2", "-M1", "-M2"):
        if key.endswith(suffix):
            key = key[: -len(suffix)].strip()
    return key


def _is_padding_channel_for_mask(name: str) -> bool:
    return _normalize_channel_name_for_mask(name).startswith("PAD_CH_")


def _is_auxiliary_channel_for_mask(name: str) -> bool:
    key = _normalize_channel_name_for_mask(name)
    compact = "".join(ch for ch in key if ch.isalnum())
    aux_tokens = (
        "EOG", "HEO", "HEOG", "VEO", "VEOG",
        "EMG", "ECG", "EKG", "GSR", "EDA",
        "RESP", "BREATH", "PULSE", "PPG", "ACC", "GYRO",
        "TRIG", "TRIGGER", "STI", "STATUS", "MARKER", "EVENT",
    )
    return any(token in compact for token in aux_tokens)


def _channel_names_from_metadata(cfg: DictConfig):
    try:
        meta = load_dataset_metadata(cfg)
    except Exception as exc:
        print(f"[ChannelMask] channel metadata unavailable; mask candidates fall back to all channels: {exc}")
        return None
    dataset = meta.get("dataset", {}) if isinstance(meta, dict) else {}
    names = dataset.get("channels") or dataset.get("ch_names")
    if not names:
        return None
    return [str(x) for x in names]


def _eligible_mask_channels(num_channels: int, channel_names=None) -> list[int]:
    num_channels = int(num_channels)
    if not channel_names:
        return list(range(num_channels))

    names = [str(x) for x in channel_names]
    eligible = []
    for idx in range(num_channels):
        if idx >= len(names):
            continue
        name = names[idx]
        if _is_padding_channel_for_mask(name) or _is_auxiliary_channel_for_mask(name):
            continue
        eligible.append(idx)
    return eligible


def _build_loaders(cfg: DictConfig, device: torch.device):
    # ---- zero-mask config (optional) ----
    noise = OmegaConf.select(cfg, "dataset.noise") or {}
    enable_noise = bool(noise.get("enable", False))
    noise_ratio  = float(noise.get("ratio", 0.0)) if enable_noise else 0.0
    apply_to     = str(noise.get("apply_to", "train")).lower() if enable_noise else "train"
    fixed_mask_channels = bool(noise.get("fixed_channels", False)) if enable_noise else False

    def _get_mask_ratio(split_name):
        if not enable_noise: return 0.0
        if apply_to == "all": return noise_ratio
        if apply_to == "train" and split_name == "train": return noise_ratio
        if apply_to in ["train_val", "train+val"] and split_name in ["train", "val"]: return noise_ratio
        return 0.0

    def _build_fixed_mask_channels(num_channels: int, channel_names=None):
        if not enable_noise or not fixed_mask_channels or noise_ratio <= 0:
            return None
        eligible = _eligible_mask_channels(num_channels, channel_names)
        if not eligible:
            print("[ChannelMask] no eligible EEG channels found; mask candidates fall back to all channels.")
            eligible = list(range(int(num_channels)))
        k = max(1, int(len(eligible) * noise_ratio))
        k = min(k, len(eligible))
        rng = np.random.RandomState(int(cfg.seed))
        selected = rng.choice(np.asarray(eligible, dtype=np.int64), size=k, replace=False).tolist()
        if channel_names:
            excluded = int(num_channels) - len(eligible)
            print(
                f"[ChannelMask] eligible EEG channels: {len(eligible)}/{int(num_channels)} "
                f"(excluded padding/auxiliary: {excluded})"
            )
        return sorted(int(x) for x in selected)

    raw_cache_enable = OmegaConf.select(cfg, "dataset.raw_cache.enable")
    raw_cache_max_gb = OmegaConf.select(cfg, "dataset.raw_cache.max_gb")
    raw_cache_splits = OmegaConf.select(cfg, "dataset.raw_cache.splits")

    def _raw_cache_for_split(split_name: str):
        if raw_cache_enable is None:
            return None
        if raw_cache_splits is None:
            return raw_cache_enable
        if isinstance(raw_cache_splits, str):
            allowed = {x.strip().lower() for x in raw_cache_splits.split(",") if x.strip()}
        else:
            allowed = {str(x).strip().lower() for x in raw_cache_splits}
        return bool(raw_cache_enable) and str(split_name).lower() in allowed

    backend = str(OmegaConf.select(cfg, "dataset.backend") or "h5").lower().strip()
    if backend == "zarr":
        zarr_root = str(
            OmegaConf.select(cfg, "paths.zarr_dataset_name")
            or OmegaConf.select(cfg, "paths.zarr_dataset_dir")
            or cfg.paths.dataset_name
        )
        split_seed = int(OmegaConf.select(cfg, "dataset.zarr_split_seed") or cfg.seed)
        split_manifest_path = os.path.join(str(cfg.paths.data_index_dir), "split_manifest.json")
        train_ratio = OmegaConf.select(cfg, "train.ratio_shot")
        train_split_indices = None
        train_manifest_path = split_manifest_path
        if train_ratio is not None:
            train_idx_path = _select_train_index_file(cfg)
            train_split_indices = _read_index_file_as_ints(train_idx_path)
            train_manifest_path = None
            print(
                f"[ZarrFewshot] using explicit train indices: "
                f"{train_idx_path} ({len(train_split_indices)} samples)"
            )
        fixed_zero_mask_channels = None
        if enable_noise and fixed_mask_channels:
            probe_dataset = ZarrSplitDataset(
                zarr_root,
                split="train",
                zero_mask_ratio=0.0,
                seed=int(cfg.seed),
                ratio_shot=None,
                split_seed=split_seed,
                split_indices=train_split_indices,
                split_manifest_path=train_manifest_path,
                raw_cache=False,
                raw_cache_max_gb=raw_cache_max_gb,
            )
            probe_dataset._ensure_open()
            fixed_zero_mask_channels = _build_fixed_mask_channels(int(probe_dataset._signals.shape[1]), probe_dataset.channel_names)
            if fixed_zero_mask_channels is not None:
                print(
                    f"[ChannelMask] fixed channels seed={int(cfg.seed)} ratio={noise_ratio:g}: "
                    f"{len(fixed_zero_mask_channels)} channels -> {fixed_zero_mask_channels}"
                )
        dataset_train = ZarrSplitDataset(
            zarr_root,
            split="train",
            zero_mask_ratio=_get_mask_ratio("train"),
            seed=int(cfg.seed),
            ratio_shot=None,
            split_seed=split_seed,
            split_indices=train_split_indices,
            split_manifest_path=train_manifest_path,
            raw_cache=_raw_cache_for_split("train"),
            raw_cache_max_gb=raw_cache_max_gb,
            zero_mask_channels=fixed_zero_mask_channels if _get_mask_ratio("train") > 0 else None,
        )
        dataset_val = ZarrSplitDataset(
            zarr_root,
            split="val",
            zero_mask_ratio=_get_mask_ratio("val"),
            seed=int(cfg.seed),
            split_seed=split_seed,
            split_manifest_path=split_manifest_path,
            raw_cache=_raw_cache_for_split("val"),
            raw_cache_max_gb=raw_cache_max_gb,
            zero_mask_channels=fixed_zero_mask_channels if _get_mask_ratio("val") > 0 else None,
        )
        dataset_test = ZarrSplitDataset(
            zarr_root,
            split="test",
            zero_mask_ratio=_get_mask_ratio("test"),
            seed=int(cfg.seed),
            split_seed=split_seed,
            split_manifest_path=split_manifest_path,
            raw_cache=_raw_cache_for_split("test"),
            raw_cache_max_gb=raw_cache_max_gb,
            zero_mask_channels=fixed_zero_mask_channels if _get_mask_ratio("test") > 0 else None,
        )
    else:
        train_idx_path = _select_train_index_file(cfg)
        val_idx_path   = os.path.join(cfg.paths.data_index_dir, "val_idx.txt")
        test_idx_path  = os.path.join(cfg.paths.data_index_dir, "test_idx.txt")

        if not os.path.exists(train_idx_path):
            raise FileNotFoundError(f"Train index not found: {train_idx_path}")
        if not os.path.exists(val_idx_path):
            raise FileNotFoundError(f"Val index not found: {val_idx_path}")
        if not os.path.exists(test_idx_path):
            raise FileNotFoundError(f"Test index not found: {test_idx_path}")

        dataset_train = EEGIndexDataset(
            train_idx_path, split="train", 
            zero_mask_ratio=_get_mask_ratio("train"), seed=int(cfg.seed)
        )
        fixed_zero_mask_channels = None
        if enable_noise and fixed_mask_channels:
            fixed_zero_mask_channels = _build_fixed_mask_channels(dataset_train[0][0].shape[0], _channel_names_from_metadata(cfg))
            if fixed_zero_mask_channels is not None:
                print(
                    f"[ChannelMask] fixed channels seed={int(cfg.seed)} ratio={noise_ratio:g}: "
                    f"{len(fixed_zero_mask_channels)} channels -> {fixed_zero_mask_channels}"
                )
            dataset_train.zero_mask_ratio = 0.0
            dataset_train.zero_mask_channels = fixed_zero_mask_channels if _get_mask_ratio("train") > 0 else None
        dataset_val = EEGIndexDataset(
            val_idx_path, split="val", 
            zero_mask_ratio=_get_mask_ratio("val"), seed=int(cfg.seed),
            zero_mask_channels=fixed_zero_mask_channels if _get_mask_ratio("val") > 0 else None,
        )
        dataset_test = EEGIndexDataset(
            test_idx_path, split="test", 
            zero_mask_ratio=_get_mask_ratio("test"), seed=int(cfg.seed),
            zero_mask_channels=fixed_zero_mask_channels if _get_mask_ratio("test") > 0 else None,
        )

    pin_memory = (device.type == "cuda")
    bs = int(cfg.train.batch_size)
    nw = int(cfg.train.num_workers)
    loader_kwargs = {
        "num_workers": nw,
        "pin_memory": pin_memory,
    }
    if nw > 0:
        loader_kwargs["persistent_workers"] = bool(OmegaConf.select(cfg, "train.persistent_workers") if OmegaConf.select(cfg, "train.persistent_workers") is not None else True)
        loader_kwargs["prefetch_factor"] = int(OmegaConf.select(cfg, "train.prefetch_factor") or 4)

    train_loader = DataLoader(
        dataset_train,
        batch_size=bs,
        shuffle=True,
        drop_last=False,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        dataset_val,
        batch_size=bs,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        dataset_test,
        batch_size=bs,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )
    return train_loader, val_loader, test_loader


def _append_jsonl(path, data):
    dir_name = os.path.dirname(path)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data) + "\n")


def _safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return float(default)



def _dataset_tag_from_index_dir(cfg: DictConfig) -> str:
    p = str(OmegaConf.select(cfg, "paths.data_index_dir") or "")
    p = os.path.normpath(p)
    return os.path.basename(p) if p else "unknown_indices_dir"


def _dataset_name_from_tag(tag: str) -> str:
    if "_indices_" in tag:
        return tag.split("_indices_", 1)[0]
    if tag.endswith("_indices"):
        return tag[:-len("_indices")]
    return tag


def _resolve_downsample_config(cfg: DictConfig, info_path: str) -> tuple[bool, int]:
    mode = str(OmegaConf.select(cfg, "dataset.downsample.mode") or "auto").strip().lower()
    threshold = float(OmegaConf.select(cfg, "dataset.downsample.score_threshold") or 40.0)
    target_num = int(OmegaConf.select(cfg, "dataset.downsample.target_num") or 40)

    if mode in {"on", "true", "always"}:
        print(f"[Strategy] Mode forced by config: Downsampling (Fixed {target_num}).")
        return True, target_num

    if mode in {"off", "false", "never"}:
        print("[Strategy] Mode forced by config: Standard Ratio (Full).")
        return False, target_num

    if mode != "auto":
        raise ValueError(
            f"[Config] Unknown dataset.downsample.mode={mode!r}. "
            "Use one of: auto, on, off."
        )

    use_downsample = True
    with open(info_path, "r", encoding="utf-8") as f:
        d_json = json.load(f)
        stats = d_json["statistics"]

    val_trials = (
        stats.get("valid_segments")
        or stats.get("valid_trials_or_segments")
        or stats.get("valid_trials")
    )
    successful = (
        stats.get("successful")
        or stats.get("successful_subjects")
        or stats.get("successful_experiments")
    )
    num_labels = d_json["dataset"]["num_labels"]

    if not val_trials or not successful or not num_labels:
        raise ValueError(
            "[Strategy] Missing valid_segments/successful/num_labels in dataset_info.json "
            "for auto downsample decision."
        )

    score = (val_trials / successful) / num_labels
    use_downsample = score >= threshold

    if use_downsample:
        print(
            f"[Strategy] Auto mode: score={score:.2f} (threshold={threshold:.2f}). "
            f"Mode: Downsampling (Fixed {target_num})."
        )
    else:
        print(
            f"[Strategy] Auto mode: score={score:.2f} (threshold={threshold:.2f}). "
            "Mode: Standard Ratio (Full)."
        )
    return use_downsample, target_num

# def _require(cfg: DictConfig, key: str):
#     v = OmegaConf.select(cfg, key)
#     if v is None:
#         raise ValueError(f"[factory] Missing required config key: `{key}`")
#     return v

def run_one_model(cfg: DictConfig, model_name: str, device: torch.device, train_loader, val_loader, test_loader):
    # clone cfg for this model run
    cfg_m = copy.deepcopy(cfg)
    cfg_m.model.name = model_name  # <-- single model selector used by factory

    indices_tag = _dataset_tag_from_index_dir(cfg_m)
    dataset_name = _dataset_name_from_tag(indices_tag)

    print("\n" + "=" * 24 + f" Running: {model_name} " + "=" * 24)
    print(f"[Device] {device} | tuning={cfg_m.train.tuning_mode} | zscore={cfg_m.train.use_zscore}")

    # output dir per model
    out_dir = str(OmegaConf.select(cfg_m, "paths.model_output_dir") or os.path.join(cfg_m.paths.output_dir, model_name))
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "train_log.jsonl")
    summary_path = os.path.join(out_dir, "summary.json")

    # build model
    model = get_model(cfg_m).to(device)

    # apply tuning mode
    model = setup_training_mode(model, cfg_m)
    benchmark_metadata = dict(getattr(model, "benchmark_metadata", {}) or {})
    if benchmark_metadata:
        print(f"[Benchmark metadata] {benchmark_metadata}")

    # 1) 参数在哪个 device
    devs = {}
    for n,p in model.named_parameters():
        devs[str(p.device)] = devs.get(str(p.device), 0) + p.numel()
    print("[Param device distribution]", devs)
    # 2) 参数大小
    bytes_params = sum(p.numel()*p.element_size() for p in model.parameters())
    print("[Param size MB]", bytes_params/1024**2)

    # 3) cuda 显存（此时应该至少接近参数大小）
    if device.type == "cuda":
        torch.cuda.synchronize()
        print("[CUDA] allocated(MB)=", torch.cuda.memory_allocated()/1024**2,
            "reserved(MB)=", torch.cuda.memory_reserved()/1024**2)
    # optimizer / scaler
    is_zero_shot = False
    if cfg_m.train.tuning_mode=='zero_shot':
        is_zero_shot = True

    optimizer = None
    if not is_zero_shot:
        optimizer_groups = model.get_parameter_groups(
            lr=float(cfg_m.train.lr),
            weight_decay=float(cfg_m.train.weight_decay),
            cfg=cfg_m,
        )
        optimizer = torch.optim.AdamW(optimizer_groups)

    loss_scaler = NativeScaler() if device.type == "cuda" else None

    # metric for saving (based on VAL)
    dataset_json = load_dataset_metadata(cfg_m)
    
    task_mod = dataset_json['dataset']['downstream_task']
        
    # task_mod = OmegaConf.select(cfg_m, "task_mod") or "Classification"
    save_key = "accuracy" if task_mod.lower() == "classification" else "r2"
    best = -float("inf")
    best_path = ""

    # --- Zero-shot: eval only but still log ---
    if is_zero_shot:
        print("[Zero-shot] no trainable params -> eval only (val + test)")

        # val_stats = evaluate_embed_nn(val_loader, val_loader, model, device, split="val", gallery_split="val", exclude_self=True)
        # test_stats = evaluate_embed_nn(test_loader, test_loader, model, device, split="test", gallery_split="test", exclude_self=False)

        val_stats = evaluate_embed_nn(val_loader, model, device, epoch=0, args=cfg_m, split="val",
                              gallery_loader=val_loader, exclude_self=True) or {}

        test_stats = evaluate_embed_nn(test_loader, model, device, epoch=0, args=cfg_m, split="test",
                               gallery_loader=val_loader, exclude_self=False) or {}

        # val
        # val_stats = evaluate(val_loader, model, device, epoch=0, args=cfg_m, split="val") or {}
        val_metrics = val_stats.get("metrics", {}) or {}
        _append_jsonl(log_path, {
            "model": model_name,
            "phase": "val",
            "epoch": 0,
            "val_loss": _safe_float(val_stats.get("loss", 0.0)),
            **{f"val_{k}": _safe_float(v) for k, v in val_metrics.items()},
        })

        # test
        # test_stats = evaluate(test_loader, model, device, epoch=0, args=cfg_m, split="test") or {}
        test_metrics = test_stats.get("metrics", {}) or {}
        _append_jsonl(log_path, {
            "model": model_name,
            "phase": "test",
            "epoch": 0,
            "test_loss": _safe_float(test_stats.get("loss", 0.0)),
            **{f"test_{k}": _safe_float(v) for k, v in test_metrics.items()},
        })

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({
                "dataset": dataset_name,
                "indices_dir": indices_tag,
                "model": model_name,
                "model_benchmark_metadata": benchmark_metadata,
                "tuning_mode": str(cfg_m.train.tuning_mode),
                "zero_shot": True,
                "save_key": save_key,
                "best_val": None,
                "best_path": None,
                "val_metrics": {k: _safe_float(v) for k, v in val_metrics.items()},
                "test_metrics": {k: _safe_float(v) for k, v in test_metrics.items()},
                "log_path": log_path,
            }, f, ensure_ascii=False, indent=2)

        print(f"[Zero-shot] val_metrics={val_metrics}")
        print(f"[Zero-shot] test_metrics={test_metrics}")
        print(f"[Done] {model_name} (zero-shot). Logs -> {log_path}")

        return {
            "model": model_name,
            "save_key": save_key,
            "best_val": None,
            "best_path": None,
            "val_metrics": {k: _safe_float(v) for k, v in val_metrics.items()},
            "test_metrics": {k: _safe_float(v) for k, v in test_metrics.items()},
            "log_path": log_path,
            "summary_path": summary_path,
            "out_dir": out_dir,
            "zero_shot": True,
        }

    epochs = int(cfg_m.train.epochs)
    eval_freq = int(OmegaConf.select(cfg_m, "train.eval_freq") or 1)
    clip_grad = float(OmegaConf.select(cfg_m, "train.clip_grad") or 0.0)

    # ---------- train loop ----------
    for epoch in range(epochs):
        train_stats = train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            loss_scaler=loss_scaler,
            max_norm=clip_grad,
            args=cfg_m,
        ) or {}

        _append_jsonl(log_path, {
            "model": model_name,
            "phase": "train",
            "epoch": epoch,
            "train_loss": _safe_float(train_stats.get("loss", 0.0)),
            "lr": _safe_float(train_stats.get("lr", 0.0)),
        })

        if (epoch + 1) % eval_freq != 0:
            continue

        # ----- VAL -----
        val_stats = evaluate(
            data_loader=val_loader,
            model=model,
            device=device,
            epoch=epoch,
            args=cfg_m,
            split="val",
        ) or {}

        val_metrics = val_stats.get("metrics", {}) or {}
        curr = _safe_float(val_metrics.get(save_key, 0.0))

        if curr > best:
            best = curr
            best_path = os.path.join(out_dir, f"best_{model_name}_{cfg_m.train.tuning_mode}.pth")
            torch.save(model.state_dict(), best_path)
            print(f"[Save] best(val) {save_key}={best:.4f} -> {best_path}")

        _append_jsonl(log_path, {
            "model": model_name,
            "phase": "val",
            "epoch": epoch,
            "val_loss": _safe_float(val_stats.get("loss", 0.0)),
            **{f"val_{k}": _safe_float(v) for k, v in val_metrics.items()},
            "save_key": save_key,
            "val_save_value": curr,
            "best_val_so_far": _safe_float(best, -float("inf")),
        })

    # ---------- load best & TEST ----------
    test_metrics_final = {}
    test_loss_final = None

    if best_path and os.path.exists(best_path):
        print(f"[Test] loading best ckpt for test: {best_path}")
        sd = torch.load(best_path, map_location="cpu")
        model.load_state_dict(sd, strict=False)
        model.to(device)

    test_stats = evaluate(test_loader, model, device, epoch=epochs, args=cfg_m, split="test") or {}
    test_loss_final = _safe_float(test_stats.get("loss", 0.0))
    test_metrics_final = test_stats.get("metrics", {}) or {}

    _append_jsonl(log_path, {
        "model": model_name,
        "phase": "test",
        "epoch": epochs,
        "test_loss": test_loss_final,
        **{f"test_{k}": _safe_float(v) for k, v in test_metrics_final.items()},
        "based_on": "best_val_ckpt",
        "best_path": best_path or None,
        "best_val": None if best == -float("inf") else float(best),
    })

    # write summary at end
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "dataset": dataset_name,
            "indices_dir": indices_tag,
            "model": model_name,
            "model_benchmark_metadata": benchmark_metadata,
            "tuning_mode": str(cfg_m.train.tuning_mode),
            "epochs": epochs,
            "eval_freq": eval_freq,
            "save_key": save_key,
            "best_val": None if best == -float("inf") else float(best),
            "best_path": best_path or None,
            "test_metrics": {k: _safe_float(v) for k, v in test_metrics_final.items()},
            "test_loss": test_loss_final,
            "log_path": log_path,
            "output_dir": out_dir,
        }, f, ensure_ascii=False, indent=2)

    print(f"[Done] {model_name} best_val {save_key}={best:.4f}")
    if "accuracy" in test_metrics_final:
        print(f"[Test] accuracy={_safe_float(test_metrics_final.get('accuracy', 0.0)):.4f} loss={test_loss_final:.4f}")
    print(f"[Logs] {log_path}")
    print(f"[Summary] {summary_path}")

    return {
        "model": model_name,
        "save_key": save_key,
        "best_val": None if best == -float("inf") else float(best),
        "best_path": best_path or None,
        "test_metrics": {k: _safe_float(v) for k, v in test_metrics_final.items()},
        "test_loss": test_loss_final,
        "log_path": log_path,
        "summary_path": summary_path,
        "out_dir": out_dir,
        "zero_shot": False,
    }



def _main_single_seed(cfg: DictConfig):
    device = _resolve_device(cfg)
    _set_seed(int(cfg.seed))
    
    tuning_mode = str(OmegaConf.select(cfg, "train.tuning_mode") or "").strip().lower()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = tuning_mode != "full_finetune"
        if tuning_mode == "full_finetune":
            print("[cuDNN] benchmark=False for full_finetune.")

    backend = str(OmegaConf.select(cfg, "dataset.backend") or "h5").lower().strip()
    dataset_path = cfg.paths.dataset_name

    if backend == "zarr":
        zarr_downsample_mode = str(OmegaConf.select(cfg, "dataset.downsample.mode") or "off").strip().lower()
        use_downsample = zarr_downsample_mode in {"on", "true", "always", "yes"}
        downsample_target_num = int(OmegaConf.select(cfg, "dataset.downsample.target_num") or 40)
        print("[Dataset] backend=zarr: using ZarrSplitDataset and skipping H5 index generation.")
    else:
        info_path = os.path.join(dataset_path, 'dataset_info.json')
        try:
            use_downsample, downsample_target_num = _resolve_downsample_config(cfg, info_path)
        except Exception as e:
            downsample_target_num = int(OmegaConf.select(cfg, "dataset.downsample.target_num") or 40)
            fallback_mode = str(OmegaConf.select(cfg, "dataset.downsample.mode") or "auto").strip().lower()
            use_downsample = fallback_mode in {"on", "true", "always"}
            print(
                f"[Warning] Could not resolve downsample strategy: {e}. "
                f"Fallback mode={fallback_mode}, use_downsample={use_downsample}."
            )

    if use_downsample:
        suffix = "_downsample"
        if backend == "zarr":
            suffix = f"_downsample_t{downsample_target_num}"
        cfg.paths.data_index_dir = str(cfg.paths.data_index_dir).rstrip("/") + suffix
        cfg.paths.output_dir = str(cfg.paths.output_dir).rstrip("/") + suffix

    dir_path = cfg.paths.data_index_dir

    if backend == "zarr":
        print(f"[Zarr] Dataset root: {OmegaConf.select(cfg, 'paths.zarr_dataset_name')}")
        manifest_path = os.path.join(str(cfg.paths.data_index_dir), "split_manifest.json")
        reuse_manifest = bool(OmegaConf.select(cfg, "dataset.split.reuse_manifest") or False)
        if reuse_manifest and os.path.exists(manifest_path):
            print(f"[ZarrSplit] Reusing manifest: {manifest_path}")
        else:
            print(f"[ZarrSplit] Building manifest at {manifest_path}")
            build_zarr_split_manifest_from_cfg(
                cfg,
                ratio_list=(0.02, 0.05, 0.1, 0.3, 0.5),
            )
    else:
        need_generate = True

        if not os.path.exists(dir_path):
            need_generate = True
        elif not os.listdir(dir_path):
            need_generate = True
        elif not os.path.exists(os.path.join(dir_path, "train_idx.txt")):
            need_generate = True
        if need_generate:
            print(f"[Indices] Generating indices at {dir_path} (Missing or Empty)...")
            os.makedirs(dir_path, exist_ok=True)
            gen_ratio_list = (0.02, 0.05, 0.1, 0.3, 0.5)

            if use_downsample:
                generate_indices_downsample(
                    data_dir=dataset_path,
                    base_save_dir="indices",
                    trial_split_ratio=(0.8, 0.1, 0.1),
                    target_num=downsample_target_num,
                    ratio_list=gen_ratio_list,
                    seed=cfg.seed,
                    custom_label_map=None
                )
            else:
                generate_indices(
                    data_dir=dataset_path,
                    base_save_dir="indices",
                    trial_split_ratio=(0.8, 0.1, 0.1),
                    ratio_list=gen_ratio_list,
                    seed=cfg.seed,
                    custom_label_map=None
                )
        else:
            print(f"[Indices] Found valid existing indices at {dir_path}")

    # --- noise mask ratio loop (channel_mask experiment) ---
    raw_noise_ratio = OmegaConf.select(cfg, "dataset.noise.ratio")
    enable_noise = bool(OmegaConf.select(cfg, "dataset.noise.enable") or False)
    if enable_noise and (isinstance(raw_noise_ratio, (list, tuple, ListConfig)) or OmegaConf.is_list(raw_noise_ratio)):
        noise_ratio_loop = [float(r) for r in raw_noise_ratio]
        noise_is_list = True
    else:
        noise_ratio_loop = [raw_noise_ratio]
        noise_is_list = False

    base_output_dir = str(cfg.paths.output_dir)

    for noise_ratio in noise_ratio_loop:
        cfg_n = copy.deepcopy(cfg)
        if noise_is_list:
            cfg_n.dataset.noise.ratio = float(noise_ratio)
            noise_tag = _ratio_to_tag(float(noise_ratio))
            cfg_n.paths.output_dir = os.path.join(base_output_dir, f"mask_{noise_tag}")
            print(f"\n[NoiseMask] ratio={noise_ratio} -> {cfg_n.paths.output_dir}")

        # --- ratio_shot loop (fewshot experiment) ---
        raw_ratio = cfg_n.train.ratio_shot
        if raw_ratio is None:
            loop_list = [None]
        elif isinstance(raw_ratio, (list, tuple, ListConfig)) or OmegaConf.is_list(raw_ratio):
            loop_list = [float(r) for r in raw_ratio]
        else:
            loop_list = [float(raw_ratio)]

        print(f"[Config] project={cfg_n.project_name} exp={cfg_n.exp_name} seed={cfg_n.seed}")
        print(f"[Paths] Index Dir: {cfg_n.paths.data_index_dir}")

        for ratio in loop_list:
            cfg_r = copy.deepcopy(cfg_n)

            if ratio is None:
                print("[Run] Shot: Full Data (using train_idx.txt)")
                ratio_tag = "full"
            else:
                cfg_r.train.ratio_shot = float(ratio)
                tag = _ratio_to_tag(float(ratio))
                print(f"[Run] Shot: Ratio {float(ratio)} (using train_idx_ratio_{tag}.txt)")
                ratio_tag = tag

            cfg_r.paths.output_dir = os.path.join(str(cfg_n.paths.output_dir), f"ratio_{ratio_tag}")

            train_loader, val_loader, test_loader = _build_loaders(cfg_r, device)

            results = []
            for name in list(cfg_r.model.names):
                print(f"--- Running Model: {name} ---")
                r = run_one_model(cfg_r, str(name).lower(), device, train_loader, val_loader, test_loader)
                if r is not None:
                    results.append(r)

            save_leaderboard(results, cfg_r.paths.output_dir)


def _get_seed_list(cfg: DictConfig) -> list[int]:
    seed_list = OmegaConf.select(cfg, "train.seed_list")
    if seed_list is None:
        seed_list = OmegaConf.select(cfg, "seed_list")
    if seed_list is None:
        return [int(cfg.seed)]
    if isinstance(seed_list, (list, tuple, ListConfig)) or OmegaConf.is_list(seed_list):
        return [int(s) for s in seed_list]
    return [int(seed_list)]


def _set_cfg_seed(cfg: DictConfig, seed: int) -> None:
    cfg.seed = int(seed)
    vary_split_seed = bool(OmegaConf.select(cfg, "train.vary_split_seed") or False)
    if vary_split_seed:
        if OmegaConf.select(cfg, "dataset.zarr_split_seed") is not None:
            cfg.dataset.zarr_split_seed = int(seed)
        if OmegaConf.select(cfg, "dataset.split.seed") is not None:
            cfg.dataset.split.seed = int(seed)
    if OmegaConf.select(cfg, "dataset.downsample.seed") is not None:
        cfg.dataset.downsample.seed = int(seed)


def main(cfg: DictConfig):
    seed_list = _get_seed_list(cfg)
    if len(seed_list) == 1:
        _set_cfg_seed(cfg, seed_list[0])
        return _main_single_seed(cfg)

    base_output_dir = str(cfg.paths.output_dir)
    base_data_index_dir = str(cfg.paths.data_index_dir)
    print(f"[MultiSeed] Running seeds: {seed_list}")

    for seed in seed_list:
        cfg_s = copy.deepcopy(cfg)
        _set_cfg_seed(cfg_s, int(seed))
        seed_tag = f"seed_{int(seed)}"
        cfg_s.paths.output_dir = os.path.join(base_output_dir, seed_tag)
        cfg_s.paths.data_index_dir = os.path.join(base_data_index_dir, seed_tag)
        print("\n" + "#" * 24 + f" Seed: {int(seed)} " + "#" * 24)
        print(f"[MultiSeed] Output Dir: {cfg_s.paths.output_dir}")
        print(f"[MultiSeed] Index Dir: {cfg_s.paths.data_index_dir}")
        _main_single_seed(cfg_s)


def save_leaderboard(results, out_root):
    rebuild_leaderboards_from_summaries(out_root)


if __name__ == "__main__":
    yaml_path = sys.argv[1]
    cfg = load_config(yaml_path)
    cfg.paths.output_dir = cfg.paths.output_dir
    os.makedirs(cfg.paths.output_dir, exist_ok=True)
    main(cfg)
