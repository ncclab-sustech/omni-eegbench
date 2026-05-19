#!/usr/bin/env python
"""Inspect the local REVE path from config input to downstream logits.

This script is intentionally read-only: it does not train, save checkpoints, or
modify model weights. It prints the exact REVE implementation path used by the
benchmark so the code/settings can be reviewed in one place.

Examples:
  python scripts/inspect_reve_pipeline.py --config configs/linear_downsample_cross.yaml
  python scripts/inspect_reve_pipeline.py --config configs/linear_downsample_cross.yaml --run-forward --device cuda:0
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Any

import torch
from omegaconf import ListConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.metadata import load_dataset_metadata
from models.factory import get_model
from models.wrappers import set_full_finetune, set_linear_probe


def _line(title: str = "") -> None:
    if title:
        print(f"\n{'=' * 20} {title} {'=' * 20}")
    else:
        print("=" * 64)


def _fmt(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, indent=2)
    return str(value)


def _cfg_get(cfg, key: str, default: Any = None) -> Any:
    value = OmegaConf.select(cfg, key)
    return default if value is None else value


def _load_config_for_inspection(path: str, dataset: str | None):
    cfg = OmegaConf.load(path)
    dataset_name = OmegaConf.select(cfg, "dataset_name")
    selected = dataset
    if selected is None and isinstance(dataset_name, (list, tuple, ListConfig)):
        selected = str(dataset_name[0])
    if selected is not None:
        cfg.dataset_name = selected
    OmegaConf.resolve(cfg)
    return cfg


def _count_params(module: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def _module_source(module_or_cls: Any) -> str:
    try:
        return str(Path(inspect.getsourcefile(module_or_cls) or "").resolve())
    except Exception:
        return "<unknown>"


def _print_key_values(items: list[tuple[str, Any]]) -> None:
    width = max(len(k) for k, _ in items) if items else 0
    for key, value in items:
        print(f"{key:<{width}} : {_fmt(value)}")


def _fallback_metadata(args) -> dict[str, Any]:
    channels = [x.strip() for x in str(args.channels).split(",") if x.strip()]
    return {
        "dataset": {
            "name": args.dataset or "manual_reve_inspection",
            "downstream_task": "classification",
            "num_labels": int(args.num_labels),
            "channels": channels,
        },
        "processing": {
            "target_sampling_rate": float(args.sampling_rate),
            "window_sec": float(args.window_sec),
        },
        "statistics": {},
    }


def _metadata_summary(cfg, args) -> dict[str, Any]:
    try:
        meta = load_dataset_metadata(cfg)
    except Exception as exc:
        print(f"\n[metadata fallback] could not load dataset metadata: {type(exc).__name__}: {exc}")
        print("[metadata fallback] using manual/default channels and timing for model inspection.")
        meta = _fallback_metadata(args)
    dataset = meta.get("dataset", {})
    processing = meta.get("processing", {})
    channels = list(dataset.get("channels") or [])
    return {
        "dataset_name": dataset.get("name"),
        "num_labels": dataset.get("num_labels"),
        "num_channels": len(channels),
        "channels_preview": channels[:16],
        "sampling_rate": processing.get("target_sampling_rate"),
        "window_sec": processing.get("window_sec"),
        "time_points": int(round(float(processing.get("target_sampling_rate", 200)) * float(processing.get("window_sec", 0)))),
        "raw": meta,
    }


def _install_metadata_override(meta: dict[str, Any]) -> None:
    """Make factory.get_model use the metadata already printed by this script."""
    import models.factory as factory

    factory.load_dataset_metadata = lambda cfg: meta


def _apply_tuning_mode(model: torch.nn.Module, mode: str) -> None:
    mode = str(mode or "").lower()
    if mode in {"linear_probe", "linear_probing", "linear", "probe"}:
        set_linear_probe(model)
    elif mode in {"full_finetune", "finetune", "full"}:
        set_full_finetune(model)


def _summarize_config(cfg, model_name: str) -> None:
    _line("Config")
    _print_key_values(
        [
            ("config model.name", _cfg_get(cfg, "model.name")),
            ("requested inspect model", model_name),
            ("paths.reve", _cfg_get(cfg, "paths.reve")),
            ("paths.reve_positions", _cfg_get(cfg, "paths.reve_positions")),
            ("model.reve.use_official_positions", _cfg_get(cfg, "model.reve.use_official_positions")),
            ("model.reve.norm", _cfg_get(cfg, "model.reve.norm")),
            ("model.reve.max_patches", _cfg_get(cfg, "model.reve.max_patches")),
            ("model.reve.montage_path", _cfg_get(cfg, "model.reve.montage_path")),
            ("model.reve.electrode_vocab_path", _cfg_get(cfg, "model.reve.electrode_vocab_path")),
            ("interpolation.mode", _cfg_get(cfg, "model.interpolation.mode")),
            ("interpolation.align_corners", _cfg_get(cfg, "model.interpolation.align_corners")),
            ("train.tuning_mode", _cfg_get(cfg, "train.tuning_mode")),
            ("train.use_zscore", _cfg_get(cfg, "train.use_zscore")),
            ("train.batch_size", _cfg_get(cfg, "train.batch_size")),
            ("train.epochs", _cfg_get(cfg, "train.epochs")),
            ("train.lr", _cfg_get(cfg, "train.lr")),
            ("train.weight_decay", _cfg_get(cfg, "train.weight_decay")),
            ("train.clip_grad", _cfg_get(cfg, "train.clip_grad")),
        ]
    )


def _summarize_metadata(summary: dict[str, Any]) -> None:
    _line("Dataset Input")
    _print_key_values(
        [
            ("dataset", summary["dataset_name"]),
            ("num_labels", summary["num_labels"]),
            ("num_channels", summary["num_channels"]),
            ("sampling_rate", summary["sampling_rate"]),
            ("window_sec", summary["window_sec"]),
            ("time_points", summary["time_points"]),
            ("channels_preview", summary["channels_preview"]),
        ]
    )


def _summarize_model(model: torch.nn.Module, cfg) -> None:
    _line("Model Objects")
    feature_extractor = model.feature_extractor
    backbone = getattr(feature_extractor, "backbone", None)
    head = model.probe_head
    total, trainable = _count_params(model)
    fe_total, fe_trainable = _count_params(feature_extractor)
    head_total, head_trainable = _count_params(head)
    _print_key_values(
        [
            ("downstream class", model.__class__.__name__),
            ("feature extractor class", feature_extractor.__class__.__name__),
            ("feature extractor source", _module_source(feature_extractor.__class__)),
            ("backbone class", backbone.__class__.__name__ if backbone is not None else None),
            ("backbone source", _module_source(backbone.__class__) if backbone is not None else None),
            ("head class", head.__class__.__name__),
            ("model params total/trainable", f"{total:,} / {trainable:,}"),
            ("feature params total/trainable", f"{fe_total:,} / {fe_trainable:,}"),
            ("head params total/trainable", f"{head_total:,} / {head_trainable:,}"),
            ("benchmark_metadata", getattr(model, "benchmark_metadata", {})),
        ]
    )

    _line("Optimizer Groups")
    groups = model.get_parameter_groups(
        lr=float(_cfg_get(cfg, "train.lr")),
        weight_decay=float(_cfg_get(cfg, "train.weight_decay")),
        cfg=cfg,
    )
    for idx, group in enumerate(groups):
        n_params = sum(p.numel() for p in group["params"])
        print(
            f"group[{idx}] name={group.get('name')} lr={group.get('lr')} "
            f"wd={group.get('weight_decay')} params={n_params:,}"
        )


def _summarize_reve_internals(model: torch.nn.Module, metadata: dict[str, Any]) -> None:
    _line("REVE Internals")
    fe = model.feature_extractor
    backbone = getattr(fe, "backbone", None)
    cfg = getattr(backbone, "config", None)

    patch_size = int(getattr(fe, "patch_size", getattr(backbone, "patch_size", getattr(cfg, "patch_size", 200))))
    patch_overlap = int(getattr(fe, "patch_overlap", getattr(backbone, "patch_overlap", getattr(cfg, "patch_overlap", 20))))
    stride = max(1, patch_size - patch_overlap)
    target_t = int(round(float(metadata["time_points"]) * float(getattr(fe, "target_freq", 200)) / float(getattr(fe, "input_freq", metadata["sampling_rate"] or 200))))
    padded_t = max(target_t, patch_size)
    remainder = (padded_t - patch_size) % stride
    if remainder:
        padded_t += stride - remainder
    inferred_patches = 1 + ((padded_t - patch_size) // stride)

    pos_cache = getattr(fe, "_positions_cache", None)
    position_sources = getattr(fe, "position_sources", None)
    missing_position_channels = getattr(fe, "missing_position_channels", None)
    dummy_position_channels = getattr(fe, "dummy_position_channels", None)

    has_fourier4d = hasattr(backbone, "fourier4d") if backbone is not None else False
    has_mlp4d = hasattr(backbone, "mlp4d") if backbone is not None else False
    has_cls_query = hasattr(backbone, "cls_query_token") if backbone is not None else False

    _print_key_values(
        [
            ("feature path", fe.__class__.__name__),
            ("input_freq -> target_freq", f"{getattr(fe, 'input_freq', None)} -> {getattr(fe, 'target_freq', None)}"),
            ("normalization module", getattr(fe, "norm", None).__class__.__name__),
            ("patch_size", patch_size),
            ("patch_overlap", patch_overlap),
            ("stride", stride),
            ("target_time_points_at_200Hz", target_t),
            ("padded_time_points", padded_t),
            ("inferred_num_patches", inferred_patches),
            ("configured max_patches", getattr(fe, "max_patches", None)),
            ("token_dim/embed_dim", getattr(fe, "token_dim", getattr(fe, "embed_dim", None))),
            ("feature output dim", getattr(fe, "embed_dim", None)),
            ("positions_cache shape", tuple(pos_cache.shape) if torch.is_tensor(pos_cache) else None),
            ("position_sources", position_sources),
            ("missing_position_channels", missing_position_channels),
            ("dummy_position_channels", dummy_position_channels),
            ("backbone has cls_query_token", has_cls_query),
            ("backbone has mlp4d", has_mlp4d),
            ("backbone has fourier4d", has_fourier4d),
            ("backbone config", cfg.to_dict() if hasattr(cfg, "to_dict") else vars(cfg) if cfg is not None else None),
        ]
    )

    print("\nForward formula visible in this local code:")
    if fe.__class__.__name__ == "REVENewFeatureExtractor":
        print("  x [B,C,T]")
        print("  -> norm(x)")
        print("  -> resample to 200 Hz if needed")
        print("  -> patchify with patch_size/patch_overlap")
        print("  -> backbone.to_patch_embedding(patches)")
        print("  -> add REVE positional embedding from positions + patch time")
        print("  -> backbone transformer + ln")
        print("  -> drop backbone CLS token, pad/crop patch tokens to max_patches")
        print("  -> trainable query attends over patch tokens")
        print("  -> concat [query_context + patch_tokens], flatten")
        print("  -> output_norm")
        print("  -> ProbeHead -> logits")
    else:
        print("  x [B,C,T] -> norm/resample -> backbone(x, positions) -> pool output -> ProbeHead -> logits")

    if not has_fourier4d:
        print("\nWARNING: this local backbone does not expose `fourier4d`.")
        print("         If the official REVE code uses FourierEmb4D + mlp4d, this local path is not a byte-for-byte official forward.")


def _run_forward(model: torch.nn.Module, metadata: dict[str, Any], device: str) -> None:
    _line("Dummy Forward")
    model.to(device)
    model.eval()
    fe = model.feature_extractor
    b = 1
    c = int(metadata["num_channels"])
    t = int(metadata["time_points"])
    x = torch.zeros(b, c, t, dtype=torch.float32, device=device)
    print(f"dummy input x: {tuple(x.shape)} on {device}")

    with torch.no_grad():
        if fe.__class__.__name__ == "REVENewFeatureExtractor":
            x1 = fe.norm(x)
            print(f"after norm: {tuple(x1.shape)}")
            if fe.input_freq != fe.target_freq:
                from models.wrappers import resample_along_time

                x1 = resample_along_time(
                    x1,
                    fe.input_freq,
                    fe.target_freq,
                    mode=fe.interpolation_mode,
                    align_corners=fe.interpolation_align_corners,
                )
                print(f"after resample: {tuple(x1.shape)}")
            positions = fe._positions_cache.to(device=x1.device, dtype=x1.dtype).unsqueeze(0).expand(x1.size(0), -1, -1)
            print(f"positions: {tuple(positions.shape)}")
            tokens = fe._extract_tokens(x1, positions)
            print(f"backbone tokens incl CLS: {tuple(tokens.shape)}")
            token_features = fe._pad_or_crop_tokens(tokens[:, 1:])
            print(f"patch tokens after pad/crop: {tuple(token_features.shape)}")
            query = fe.cls_query_token.to(dtype=token_features.dtype).expand(x1.size(0), -1, -1)
            scores = torch.matmul(query, token_features.transpose(-1, -2)) / (fe.token_dim ** 0.5)
            context = torch.matmul(torch.softmax(scores, dim=-1), token_features)
            features = torch.cat([context, token_features], dim=1).flatten(start_dim=1)
            features = fe.output_norm(features)
            print(f"feature vector: {tuple(features.shape)}")
            logits = model.probe_head(features)
        else:
            features = fe(x)
            print(f"feature vector: {tuple(features.shape)}")
            logits = model.probe_head(features)
        print(f"logits/output: {tuple(logits.shape)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the local REVE benchmark pipeline.")
    parser.add_argument("--config", default="configs/linear_downsample_cross.yaml")
    parser.add_argument("--dataset", default=None, help="Dataset name to inspect when config dataset_name is a list.")
    parser.add_argument("--model", default="reve", choices=["reve", "reve_old", "reve_new"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--run-forward", action="store_true", help="Run a dummy forward pass and print tensor shapes.")
    parser.add_argument("--channels", default="Fp1,Fp2,F3,F4,C3,C4,P3,P4,O1,O2,F7,F8,T7,T8,P7,P8,Fz,Cz,Pz")
    parser.add_argument("--sampling-rate", type=float, default=200.0)
    parser.add_argument("--window-sec", type=float, default=10.0)
    parser.add_argument("--num-labels", type=int, default=2)
    args = parser.parse_args()

    cfg = _load_config_for_inspection(args.config, args.dataset)
    cfg.model.name = args.model
    _summarize_config(cfg, args.model)

    metadata = _metadata_summary(cfg, args)
    _summarize_metadata(metadata)

    _install_metadata_override(metadata["raw"])
    model = get_model(cfg)
    _apply_tuning_mode(model, _cfg_get(cfg, "train.tuning_mode"))
    _summarize_model(model, cfg)
    _summarize_reve_internals(model, metadata)

    if args.run_forward:
        _run_forward(model, metadata, args.device)
    else:
        print("\nForward pass was not run. Add `--run-forward --device cuda:0` to print live tensor shapes.")


if __name__ == "__main__":
    main()
