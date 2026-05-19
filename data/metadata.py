from __future__ import annotations

import json
import os
import re
from typing import Any

from omegaconf import OmegaConf


def _read_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_parquet(path: str):
    try:
        import pandas as pd

        return pd.read_parquet(path)
    except ImportError:
        import pyarrow.parquet as pq

        return pq.read_table(path).to_pandas()


def _resolve_zarr_root(cfg) -> str:
    root = (
        OmegaConf.select(cfg, "paths.zarr_dataset_name")
        or OmegaConf.select(cfg, "paths.zarr_dataset_dir")
        or OmegaConf.select(cfg, "paths.dataset_name")
    )
    if root is None:
        raise ValueError("Missing zarr dataset path: paths.zarr_dataset_name or paths.zarr_dataset_dir")
    return os.path.abspath(str(root))


def _resolve_zarr_store(root: str) -> str:
    if root.endswith(".zarr"):
        return root
    base = os.path.basename(root.rstrip("/"))
    candidate = os.path.join(root, f"{base}.zarr")
    if os.path.isdir(candidate):
        return candidate
    matches = [os.path.join(root, name) for name in os.listdir(root) if name.endswith(".zarr")]
    if not matches:
        raise FileNotFoundError(f"No .zarr store found under {root}")
    return sorted(matches)[0]


_BIPOLAR_SUFFIXES = {"REF", "LE", "AR", "AVG", "M1", "M2", "A1", "A2"}
_EEG_ELECTRODE_RE = re.compile(
    r"^(?:FP|AF|F|FT|FC|C|CP|TP|T|P|PO|O|IZ|OZ|CZ|FZ|PZ|CPZ|FCZ|FPZ|AFZ)[A-Z0-9Z]*$",
    re.IGNORECASE,
)


def _strip_eeg_prefix(name: str) -> str:
    out = str(name).strip()
    if out.upper().startswith("EEG "):
        out = out[4:].strip()
    return out


def _looks_like_eeg_electrode(name: str) -> bool:
    key = _strip_eeg_prefix(name).strip().upper()
    return bool(_EEG_ELECTRODE_RE.match(key))


def normalize_bipolar_channel_name(name: str) -> str:
    """Map bipolar-like channel names to the leading referential electrode.

    Examples:
      FP1-F7 -> FP1
      EEG FP1-REF -> FP1

    This intentionally changes only the metadata channel name. The underlying
    signal remains unchanged.
    """
    raw = str(name).strip()
    if not raw or raw.upper().startswith("PAD_CH_"):
        return raw

    normalized = raw.replace("–", "-").replace("—", "-")
    if "-" not in normalized:
        return raw

    left, right = [part.strip() for part in normalized.split("-", 1)]
    if not left or not right:
        return raw

    right_key = _strip_eeg_prefix(right).upper()
    if _looks_like_eeg_electrode(left) and (
        _looks_like_eeg_electrode(right) or right_key in _BIPOLAR_SUFFIXES
    ):
        return _strip_eeg_prefix(left)
    return raw


def normalize_bipolar_channel_names(channels: list[str]) -> list[str]:
    return [normalize_bipolar_channel_name(ch) for ch in channels]


def load_dataset_metadata(cfg) -> dict[str, Any]:
    backend = str(OmegaConf.select(cfg, "dataset.backend") or "h5").lower().strip()
    if backend != "zarr":
        dataset_path = str(OmegaConf.select(cfg, "paths.dataset_name"))
        meta = _read_json(os.path.join(dataset_path, "dataset_info.json"))
        dataset = meta.get("dataset", {})
        if isinstance(dataset.get("channels"), list):
            dataset["channels"] = normalize_bipolar_channel_names([str(x) for x in dataset["channels"]])
        if isinstance(dataset.get("ch_names"), list):
            dataset["ch_names"] = normalize_bipolar_channel_names([str(x) for x in dataset["ch_names"]])
        return meta

    root = _resolve_zarr_root(cfg)
    store = _resolve_zarr_store(root)
    attrs = _read_json(os.path.join(store, ".zattrs"))

    cfg_path = os.path.join(root, "dataset_config.json")
    summary_path = os.path.join(root, "conversion_summary.json")
    index_path = os.path.join(root, "sample_index.parquet")
    label_vocab_path = os.path.join(root, "label_vocab.json")

    zcfg = _read_json(cfg_path) if os.path.exists(cfg_path) else {}
    summary = _read_json(summary_path) if os.path.exists(summary_path) else {}
    label_vocab = _read_json(label_vocab_path) if os.path.exists(label_vocab_path) else {}

    df = _read_parquet(index_path)
    label_col = "label_id" if "label_id" in df.columns else ("labels" if "labels" in df.columns else None)
    if label_col is not None:
        num_labels = int(df[label_col].max()) + 1
    elif label_vocab:
        num_labels = len(label_vocab)
    else:
        num_labels = 1

    channels = []
    if "channel_names_json" in df.columns:
        best = None
        best_len = -1
        for value in df["channel_names_json"].dropna().tolist():
            try:
                names = json.loads(str(value))
            except Exception:
                continue
            if len(names) > best_len:
                best = names
                best_len = len(names)
        channels = normalize_bipolar_channel_names([str(x) for x in (best or [])])

    c_max = int(attrs.get("c_max") or (summary.get("shape") or [0, 0, 0])[1] or len(channels))
    if len(channels) < c_max:
        channels.extend([f"PAD_CH_{i}" for i in range(len(channels), c_max)])
    elif len(channels) > c_max:
        channels = channels[:c_max]

    sampling_rate = float(attrs.get("sampling_rate") or zcfg.get("target_sampling_rate") or summary.get("sampling_rate") or 200.0)
    window_sec = float(attrs.get("segment_seconds") or zcfg.get("segment_seconds") or 10.0)
    n_samples = int(attrs.get("n_samples") or summary.get("n_samples") or len(df))
    n_subjects = int(summary.get("n_subjects") or (df["subject_id"].nunique() if "subject_id" in df.columns else 1))

    return {
        "dataset": {
            "name": attrs.get("dataset_key") or zcfg.get("dataset_key") or os.path.basename(root.rstrip("/")),
            "downstream_task": "classification",
            "num_labels": num_labels,
            "channels": channels,
        },
        "processing": {
            "target_sampling_rate": sampling_rate,
            "window_sec": window_sec,
        },
        "statistics": {
            "valid_segments": n_samples,
            "successful_subjects": n_subjects,
        },
    }
