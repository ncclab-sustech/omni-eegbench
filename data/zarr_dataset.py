from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from data.zarr_split import ensure_zarr_index_aligned, load_zarr_split_manifest


def _read_parquet(path: str):
    try:
        import pandas as pd
        return pd.read_parquet(path)
    except ImportError:
        import pyarrow.parquet as pq
        return pq.read_table(path).to_pandas()


def _open_zarr_group(path: str):
    import zarr
    return zarr.open_group(path, mode="r")


def _env_flag(name: str, default: bool = True) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return str(value).strip().lower() not in {"0", "false", "no", "off"}


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _find_label_column(df):
    for key in ("label_id", "labels", "label", "target"):
        if key in df.columns:
            return key, lambda v: int(v)
    if "label_raw" in df.columns:
        return "label_raw", lambda v: max(0, int(v) - 1)
    raise RuntimeError(f"Could not find a supported label column in sample_index: {list(df.columns)}")


def _subject_split(subject_ids, split_seed: int):
    rng = np.random.default_rng(int(split_seed))
    subjects = np.array(sorted(set(str(s) for s in subject_ids)))
    rng.shuffle(subjects)
    n = len(subjects)
    if n < 3:
        raise RuntimeError(f"Need at least 3 subjects for Zarr subject split, got {n}.")
    n_train = max(1, int(n * 0.8))
    n_val = max(1, int(n * 0.1))
    if n_train + n_val >= n:
        n_val = max(1, n - n_train - 1)
    n_test = n - n_train - n_val
    if n_test <= 0:
        n_test = 1
        n_train = max(1, n - n_val - 1)
    return {
        "train": set(subjects[:n_train].tolist()),
        "val": set(subjects[n_train:n_train + n_val].tolist()),
        "test": set(subjects[n_train + n_val:].tolist()),
    }


def _stratified_take(indices_by_label, ratio: float, seed: int):
    rng = np.random.default_rng(int(seed))
    out = []
    for _, idxs in sorted(indices_by_label.items()):
        idxs = np.asarray(idxs, dtype=np.int64)
        rng.shuffle(idxs)
        k = max(1, int(round(len(idxs) * float(ratio))))
        out.extend(idxs[:k].tolist())
    out.sort()
    return out


class ZarrSplitDataset(Dataset):
    """Minimal Zarr-backed dataset for the 5.0 benchmark.

    It reads `sample_index.parquet` + `*.zarr/signals` and returns [C, T] tensors.
    The split is subject-based when no explicit split column is available.
    """

    def __init__(
        self,
        zarr_dataset_dir: str,
        split: str = "train",
        zero_mask_ratio: float = 0.0,
        seed: int = 42,
        ratio_shot: Optional[float] = None,
        split_seed: Optional[int] = None,
        split_indices: Optional[list[int]] = None,
        split_manifest_path: Optional[str] = None,
        raw_cache: Optional[bool] = None,
        raw_cache_max_gb: Optional[float] = None,
        zero_mask_channels: Optional[list[int]] = None,
    ):
        self.zarr_dataset_dir = os.path.abspath(zarr_dataset_dir)
        self.split = str(split)
        self.zero_mask_ratio = float(zero_mask_ratio)
        self.zero_mask_channels = None if zero_mask_channels is None else [int(x) for x in zero_mask_channels]
        self._rng = np.random.RandomState(int(seed))
        self._zarr_root = None
        self._signals = None
        self._raw_cache = None
        self.raw_cache = _env_flag("BENCHMARK_EEG_ZARR_RAW_CACHE", True) if raw_cache is None else bool(raw_cache)
        self.raw_cache_max_gb = (
            _env_float("BENCHMARK_EEG_ZARR_RAW_CACHE_GB", 8.0)
            if raw_cache_max_gb is None
            else float(raw_cache_max_gb)
        )

        self.zarr_path, self.index_path = self._resolve_paths(self.zarr_dataset_dir)
        self.index_path, df = ensure_zarr_index_aligned(self.zarr_path, self.index_path)
        if len(df) == 0:
            raise RuntimeError(f"Empty sample_index.parquet: {self.index_path}")

        label_col, label_fn = _find_label_column(df)
        df = df.copy()
        df["_label"] = [label_fn(v) for v in df[label_col].tolist()]
        if "subject_id" not in df.columns:
            raise RuntimeError("sample_index.parquet must contain subject_id for Zarr subject split.")

        self.channel_names = None
        if "channel_names_json" in df.columns:
            first_valid = next((v for v in df["channel_names_json"].tolist() if v is not None and str(v) != "nan"), None)
            if first_valid is not None:
                try:
                    self.channel_names = json.loads(str(first_valid))
                except Exception:
                    self.channel_names = None

        if split_indices is not None and split_manifest_path is not None:
            raise ValueError("Pass either split_indices or split_manifest_path, not both.")

        selected = None
        if split_manifest_path is not None:
            manifest = load_zarr_split_manifest(split_manifest_path)
            manifest_splits = manifest.get("splits", {})
            if self.split not in manifest_splits:
                raise ValueError(f"Unknown split={self.split!r}; expected one of {list(manifest_splits)}")
            selected = [int(x) for x in manifest_splits[self.split]]
        elif split_indices is not None:
            selected = [int(x) for x in split_indices]
        else:
            explicit_split = None
            if "split" in df.columns:
                uniq = {str(x).strip().lower() for x in df["split"].dropna().tolist()}
                if {"train", "val", "test"}.issubset(uniq):
                    explicit_split = {k: df[df["split"].astype(str).str.strip().str.lower() == k].index.tolist() for k in ("train", "val", "test")}

            if explicit_split is None:
                subject_splits = _subject_split(df["subject_id"].astype(str).tolist(), split_seed if split_seed is not None else seed)
                generated_splits = {
                    name: df[df["subject_id"].astype(str).isin(subjects)].index.tolist()
                    for name, subjects in subject_splits.items()
                }
            else:
                generated_splits = explicit_split

            if self.split not in generated_splits:
                raise ValueError(f"Unknown split={self.split!r}; expected one of {list(generated_splits)}")
            selected = list(generated_splits[self.split])

        if self.split == "train" and ratio_shot is not None:
            by_label = defaultdict(list)
            for ridx in selected:
                by_label[int(df.loc[ridx, "_label"])].append(int(ridx))
            selected = _stratified_take(by_label, float(ratio_shot), seed)

        if not selected:
            raise RuntimeError(f"No Zarr samples selected for split={self.split} in {self.zarr_dataset_dir}")

        self.sample_rows = selected
        self.labels = [int(df.loc[ridx, "_label"]) for ridx in self.sample_rows]
        self.zarr_indices = list(range(len(self.sample_rows))) if "sample_id" not in df.columns else [int(ridx) for ridx in self.sample_rows]
        self.sample_to_zarr_index = [int(ridx) for ridx in self.sample_rows]
        self.num_classes = len(sorted(set(self.labels)))
        self._maybe_build_raw_cache()

    @staticmethod
    def _resolve_paths(root: str):
        if root.endswith('.zarr'):
            zarr_path = root
            index_path = os.path.join(os.path.dirname(root), 'sample_index.parquet')
        else:
            base = os.path.basename(root.rstrip('/'))
            candidate = os.path.join(root, f'{base}.zarr')
            zarr_path = candidate if os.path.isdir(candidate) else None
            if zarr_path is None:
                matches = [os.path.join(root, name) for name in os.listdir(root) if name.endswith('.zarr')]
                if not matches:
                    raise FileNotFoundError(f'No .zarr store found under {root}')
                zarr_path = sorted(matches)[0]
            index_path = os.path.join(root, 'sample_index.parquet')
        if not os.path.isdir(zarr_path):
            raise FileNotFoundError(f'Zarr store not found: {zarr_path}')
        if not os.path.exists(index_path):
            raise FileNotFoundError(f'sample_index.parquet not found: {index_path}')
        return zarr_path, index_path

    def __len__(self):
        return len(self.sample_rows)

    def _ensure_open(self):
        if self._zarr_root is None:
            self._zarr_root = _open_zarr_group(self.zarr_path)
            self._signals = self._zarr_root['signals']

    def _maybe_build_raw_cache(self):
        if not self.raw_cache:
            print(f"[ZarrSplitDataset] {self.split} raw-cache disabled.")
            return

        max_gb = self.raw_cache_max_gb
        self._ensure_open()
        sample_shape = tuple(self._signals.shape[1:])
        bytes_per_sample = int(np.prod(sample_shape)) * np.dtype(self._signals.dtype).itemsize
        est_gb = bytes_per_sample * len(self.sample_to_zarr_index) / (1024 ** 3)
        if est_gb > max_gb:
            print(
                f"[ZarrSplitDataset] {self.split} raw-cache skipped: "
                f"{len(self.sample_to_zarr_index)} samples need ~{est_gb:.2f} GB "
                f"(limit {max_gb:.2f} GB)."
            )
            return

        full_gb = (
            int(np.prod(self._signals.shape))
            * np.dtype(self._signals.dtype).itemsize
            / (1024 ** 3)
        )
        if full_gb <= max_gb:
            # Sequential full-store reads are much faster than thousands of random
            # single-sample reads for chunked zarr stores such as BCIC2A.
            full = np.asarray(self._signals[:], dtype=np.float32)
            cache = full[np.asarray(self.sample_to_zarr_index, dtype=np.int64)].copy()
        else:
            cache = np.empty((len(self.sample_to_zarr_index),) + sample_shape, dtype=np.float32)
            for out_idx, zarr_index in enumerate(self.sample_to_zarr_index):
                cache[out_idx] = np.asarray(self._signals[int(zarr_index)], dtype=np.float32)

        self._raw_cache = cache
        print(
            f"[ZarrSplitDataset] {self.split} raw-cache enabled: "
            f"{len(self.sample_to_zarr_index)} samples (~{est_gb:.2f} GB; full store ~{full_gb:.2f} GB)."
        )

    def _maybe_zero_mask(self, x: torch.Tensor) -> torch.Tensor:
        if self.zero_mask_channels is not None:
            if not self.zero_mask_channels:
                return x
            x = x.clone()
            valid = [idx for idx in self.zero_mask_channels if 0 <= idx < x.shape[0]]
            if valid:
                x[valid, :] = 0.0
            return x

        if self.split != 'train' or self.zero_mask_ratio <= 0:
            return x
        c = x.shape[0]
        k = max(1, int(c * self.zero_mask_ratio))
        ch_idx = self._rng.choice(c, size=k, replace=False)
        x = x.clone()
        x[ch_idx, :] = 0.0
        return x

    def __getitem__(self, idx):
        if self._raw_cache is not None:
            x = self._raw_cache[idx]
        else:
            self._ensure_open()
            zarr_index = int(self.sample_to_zarr_index[idx])
            x = np.asarray(self._signals[zarr_index], dtype=np.float32)
        y = int(self.labels[idx])
        xt = torch.from_numpy(x)
        xt = self._maybe_zero_mask(xt)
        return xt, torch.tensor(y).long()
