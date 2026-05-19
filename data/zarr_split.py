from __future__ import annotations

import json
import os
import hashlib
from collections import Counter, defaultdict
from typing import Any, Optional, Sequence

import numpy as np
from omegaconf import OmegaConf


def read_parquet(path: str):
    try:
        import pandas as pd

        return pd.read_parquet(path)
    except ImportError:
        import pyarrow.parquet as pq

        return pq.read_table(path).to_pandas()


def _write_parquet(df, path: str) -> None:
    try:
        df.to_parquet(path, index=False)
    except ImportError:
        import pyarrow as pa
        import pyarrow.parquet as pq

        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path)


def ratio_to_tag(r: float) -> str:
    return str(float(r)).replace(".", "p")


def resolve_zarr_paths(root: str) -> tuple[str, str]:
    root = os.path.abspath(str(root))
    if root.endswith(".zarr"):
        zarr_path = root
        index_path = os.path.join(os.path.dirname(root), "sample_index.parquet")
    else:
        base = os.path.basename(root.rstrip("/"))
        candidate = os.path.join(root, f"{base}.zarr")
        zarr_path = candidate if os.path.isdir(candidate) else None
        if zarr_path is None:
            matches = [os.path.join(root, name) for name in os.listdir(root) if name.endswith(".zarr")]
            if not matches:
                raise FileNotFoundError(f"No .zarr store found under {root}")
            zarr_path = sorted(matches)[0]
        index_path = os.path.join(root, "sample_index.parquet")
    if not os.path.isdir(zarr_path):
        raise FileNotFoundError(f"Zarr store not found: {zarr_path}")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"sample_index.parquet not found: {index_path}")
    return zarr_path, index_path


def _read_zarr_array_if_present(group, key: str, n_samples: int, default):
    if key not in group:
        return [default(i) if callable(default) else default for i in range(n_samples)]
    values = np.asarray(group[key][:])
    if len(values) != n_samples:
        return [default(i) if callable(default) else default for i in range(n_samples)]
    out = []
    for value in values.tolist():
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        out.append(value)
    return out


def _build_aligned_index_from_zarr(zarr_path: str, original_index_path: str, repaired_index_path: str):
    import pandas as pd
    import zarr

    group = zarr.open_group(zarr_path, mode="r")
    if "signals" not in group:
        raise RuntimeError(f"Zarr store has no `signals` array: {zarr_path}")
    n_samples = int(group["signals"].shape[0])
    attrs = dict(group.attrs)
    dataset_key = str(attrs.get("dataset_key") or os.path.basename(zarr_path).replace(".zarr", ""))

    original_df = read_parquet(original_index_path)
    first_channel_names = None
    if "channel_names_json" in original_df.columns:
        valid = original_df["channel_names_json"].dropna()
        if len(valid) > 0:
            first_channel_names = str(valid.iloc[0])

    labels = _read_zarr_array_if_present(group, "labels", n_samples, 0)
    subject_ids = _read_zarr_array_if_present(group, "subject_ids", n_samples, lambda i: f"subject_{i:06d}")
    channel_counts = _read_zarr_array_if_present(
        group,
        "channel_counts",
        n_samples,
        int(group["signals"].shape[1]) if group["signals"].ndim >= 2 else 0,
    )
    valid_time = _read_zarr_array_if_present(
        group,
        "valid_time_samples",
        n_samples,
        int(group["signals"].shape[-1]) if group["signals"].ndim >= 3 else 1,
    )
    qc_flags = _read_zarr_array_if_present(group, "qc_flags", n_samples, 0)

    data = {
        "sample_id": [f"{dataset_key}:zarr:{i}" for i in range(n_samples)],
        "dataset_key": [dataset_key] * n_samples,
        "subject_id": [str(x) for x in subject_ids],
        "session_id": ["zarr_repaired"] * n_samples,
        "segment_id": [f"segment_{i}" for i in range(n_samples)],
        "source_relpath": ["zarr_repaired"] * n_samples,
        "source_format": ["zarr"] * n_samples,
        "segment_start_sec": [float(i) for i in range(n_samples)],
        "segment_end_sec": [float(i + 1) for i in range(n_samples)],
        "label_id": [int(x) for x in labels],
        "label_raw": [int(x) for x in labels],
        "task_code": [dataset_key] * n_samples,
        "raw_sfreq": [float(attrs.get("sampling_rate") or 0.0)] * n_samples,
        "notch_hz": [attrs.get("notch_hz")] * n_samples,
        "qc_flags": [int(x) for x in qc_flags],
        "channel_count": [int(x) for x in channel_counts],
        "valid_time_samples": [int(x) for x in valid_time],
        "bandpass_high_effective": [attrs.get("bandpass_high")] * n_samples,
        "trial_id": [f"{str(subject_ids[i])}:segment_{i}" for i in range(n_samples)],
    }
    if first_channel_names is not None:
        data["channel_names_json"] = [first_channel_names] * n_samples

    repaired_df = pd.DataFrame(data)
    _write_parquet(repaired_df, repaired_index_path)
    return repaired_df


def _repaired_index_paths(zarr_path: str, index_path: str) -> list[str]:
    primary = os.path.join(os.path.dirname(index_path), "sample_index.repaired.parquet")
    cache_root = os.environ.get(
        "BENCHMARK_EEG_ZARR_INDEX_REPAIR_DIR",
        os.path.join("/tmp", "benchmark_eeg_zarr_index_repair"),
    )
    stem = os.path.basename(os.path.dirname(index_path.rstrip(os.sep))) or os.path.basename(zarr_path).replace(".zarr", "")
    digest = hashlib.sha1(os.path.abspath(zarr_path).encode("utf-8")).hexdigest()[:12]
    fallback = os.path.join(cache_root, f"{stem}.{digest}.sample_index.repaired.parquet")
    return [primary] if os.path.abspath(primary) == os.path.abspath(fallback) else [primary, fallback]


def ensure_zarr_index_aligned(zarr_path: str, index_path: str) -> tuple[str, Any]:
    """Return an index parquet aligned with zarr/signals.

    Broken conversions can leave `sample_index.parquet` out of sync with
    `signals`. In that case, build a repaired parquet from zarr arrays such
    as labels and subject_ids, without overwriting the original file.
    """
    import zarr

    group = zarr.open_group(zarr_path, mode="r")
    n_samples = int(group["signals"].shape[0])
    df = read_parquet(index_path)
    if len(df) == n_samples:
        return index_path, df

    candidates = _repaired_index_paths(zarr_path, index_path)
    for repaired_index_path in candidates:
        if os.path.exists(repaired_index_path):
            try:
                repaired_df = read_parquet(repaired_index_path)
            except Exception as exc:
                print(f"[ZarrIndexRepair] ignoring unreadable repaired index {repaired_index_path}: {exc}")
                continue
            if len(repaired_df) == n_samples:
                print(
                    f"[ZarrIndexRepair] using existing repaired index: "
                    f"{repaired_index_path} ({len(df)} original rows -> {n_samples} zarr samples)"
                )
                return repaired_index_path, repaired_df

    last_error = None
    for repaired_index_path in candidates:
        try:
            os.makedirs(os.path.dirname(repaired_index_path), exist_ok=True)
            print(
                f"[ZarrIndexRepair] sample_index row count does not match zarr signals: "
                f"{len(df)} rows vs {n_samples} samples. Rebuilding {repaired_index_path}."
            )
            repaired_df = _build_aligned_index_from_zarr(zarr_path, index_path, repaired_index_path)
            return repaired_index_path, repaired_df
        except OSError as exc:
            last_error = exc
            print(f"[ZarrIndexRepair] could not write {repaired_index_path}: {exc}")

    raise RuntimeError(f"Failed to build repaired zarr sample index for {zarr_path}: {last_error}")


def find_label_column(df):
    for key in ("label_id", "labels", "label", "target"):
        if key in df.columns:
            return key, lambda v: int(v)
    if "label_raw" in df.columns:
        return "label_raw", lambda v: max(0, int(v) - 1)
    raise RuntimeError(f"Could not find a supported label column in sample_index: {list(df.columns)}")


def _as_list(value) -> Optional[list]:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence):
        return list(value)
    return [value]


def _normalize_ratios(ratios) -> tuple[float, float, float]:
    if ratios is None:
        ratios = [0.8, 0.1, 0.1]
    values = [float(x) for x in list(ratios)]
    if len(values) != 3:
        raise ValueError(f"dataset.split.ratios must have 3 values, got {values}")
    if any(v < 0 for v in values) or sum(values) <= 0:
        raise ValueError(f"Invalid split ratios: {values}")
    total = sum(values)
    return values[0] / total, values[1] / total, values[2] / total


def _split_units(units: Sequence[str], ratios: tuple[float, float, float], seed: int) -> dict[str, set[str]]:
    rng = np.random.default_rng(int(seed))
    arr = np.array(sorted(set(str(u) for u in units)))
    rng.shuffle(arr)
    n = len(arr)
    if n < 3:
        raise RuntimeError(f"Need at least 3 split units, got {n}.")

    n_train = int(round(n * ratios[0]))
    n_val = int(round(n * ratios[1]))
    n_train = min(max(1, n_train), n - 2)
    n_val = min(max(1, n_val), n - n_train - 1)

    return {
        "train": set(arr[:n_train].tolist()),
        "val": set(arr[n_train:n_train + n_val].tolist()),
        "test": set(arr[n_train + n_val:].tolist()),
    }


def _filter_df(df, filters: dict[str, Any]):
    mask = np.ones(len(df), dtype=bool)

    include_labels = _as_list(filters.get("include_labels"))
    if include_labels is not None:
        include = {int(x) for x in include_labels}
        mask &= df["_label"].astype(int).isin(include).to_numpy()

    exclude_qc_flags = filters.get("exclude_qc_flags")
    if exclude_qc_flags is not None and "qc_flags" in df.columns:
        flags = _as_list(exclude_qc_flags)
        bitmask = 0
        for flag in flags:
            bitmask |= int(flag)
        mask &= ((df["qc_flags"].fillna(0).astype(int).to_numpy() & bitmask) == 0)

    min_channel_count = filters.get("min_channel_count")
    if min_channel_count is not None and "channel_count" in df.columns:
        mask &= df["channel_count"].fillna(0).astype(int).to_numpy() >= int(min_channel_count)

    if bool(filters.get("require_valid_time", False)) and "valid_time_samples" in df.columns:
        mask &= df["valid_time_samples"].fillna(0).astype(int).to_numpy() > 0

    return df[mask].copy()


def _trial_column(df) -> str:
    for key in ("trial_id", "session_id", "source_relpath"):
        if key in df.columns:
            return key
    raise RuntimeError(
        "Within-subject Zarr split needs one of: trial_id, session_id, source_relpath."
    )


def _cross_subject_split(df, ratios: tuple[float, float, float], seed: int):
    if "subject_id" not in df.columns:
        raise RuntimeError("sample_index.parquet must contain subject_id for cross-subject Zarr split.")
    split_subjects = _split_units(df["subject_id"].astype(str).tolist(), ratios, seed)
    return {
        name: sorted(df.index[df["subject_id"].astype(str).isin(units)].astype(int).tolist())
        for name, units in split_subjects.items()
    }


def _predefined_split(df):
    if "split" not in df.columns:
        raise RuntimeError("sample_index.parquet must contain split for predefined Zarr split.")

    normalized = df["split"].astype(str).str.strip().str.lower()
    aliases = {
        "training": "train",
        "validation": "val",
        "valid": "val",
        "dev": "val",
        "testing": "test",
    }
    normalized = normalized.map(lambda x: aliases.get(x, x))
    valid = {"train", "val", "test"}
    unknown = sorted(set(normalized.tolist()) - valid)
    if unknown:
        raise RuntimeError(f"Unknown predefined split value(s) in sample_index.parquet: {unknown}")

    return {
        split_name: sorted(df.index[normalized == split_name].astype(int).tolist())
        for split_name in ("train", "val", "test")
    }


def _within_subject_split(
    df,
    ratios: tuple[float, float, float],
    seed: int,
    small_subject_policy: str,
):
    if "subject_id" not in df.columns:
        raise RuntimeError("sample_index.parquet must contain subject_id for within-subject Zarr split.")
    trial_col = _trial_column(df)
    rng = np.random.default_rng(int(seed))
    out = {"train": [], "val": [], "test": []}

    for subject_id, sub_df in df.groupby(df["subject_id"].astype(str), sort=True):
        trials = np.array(sorted(set(str(x) for x in sub_df[trial_col].astype(str).tolist())))
        rng.shuffle(trials)
        if len(trials) < 3:
            if small_subject_policy == "error":
                raise RuntimeError(
                    f"Subject {subject_id!r} has only {len(trials)} trial(s), cannot split within-subject."
                )
            if small_subject_policy == "skip":
                continue
            if small_subject_policy != "warn_train":
                raise ValueError(f"Unknown small_subject_policy={small_subject_policy!r}")
            print(
                f"[ZarrSplit] Subject {subject_id!r} has only {len(trials)} trial(s); "
                "placing all its samples in train."
            )
            out["train"].extend(sub_df.index.astype(int).tolist())
            continue

        subject_seed = int(hashlib.sha1(subject_id.encode("utf-8")).hexdigest()[:8], 16)
        trial_splits = _split_units(trials.tolist(), ratios, seed + subject_seed % 1000003)
        for split_name, split_trials in trial_splits.items():
            idxs = sub_df.index[sub_df[trial_col].astype(str).isin(split_trials)].astype(int).tolist()
            out[split_name].extend(idxs)

    return {k: sorted(v) for k, v in out.items()}


def _stratified_take(df, indices: Sequence[int], ratio: float, seed: int) -> list[int]:
    rng = np.random.default_rng(int(seed))
    by_label = defaultdict(list)
    for ridx in indices:
        by_label[int(df.loc[int(ridx), "_label"])].append(int(ridx))

    selected = []
    for _, idxs in sorted(by_label.items()):
        arr = np.asarray(idxs, dtype=np.int64)
        rng.shuffle(arr)
        k = max(1, int(round(len(arr) * float(ratio))))
        selected.extend(arr[:k].tolist())
    return sorted(selected)


def _stratified_ratio_splits(
    df,
    indices: Sequence[int],
    ratios: Sequence[float],
    seed: int,
) -> dict[float, list[int]]:
    """Build nested few-shot splits per label.

    For each label we shuffle once, then each ratio takes the prefix of that
    same shuffled order. This guarantees smaller shots are subsets of larger
    shots while preserving per-label proportions.
    """
    ratio_values = sorted(float(r) for r in ratios)
    if any(r <= 0 for r in ratio_values):
        raise ValueError(f"ratio_list values must be > 0, got {ratio_values}")

    rng = np.random.default_rng(int(seed))
    by_label = defaultdict(list)
    for ridx in indices:
        by_label[int(df.loc[int(ridx), "_label"])].append(int(ridx))

    shuffled_by_label = {}
    for label, idxs in sorted(by_label.items()):
        arr = np.asarray(idxs, dtype=np.int64)
        rng.shuffle(arr)
        shuffled_by_label[label] = arr

    out = {}
    for ratio in ratio_values:
        selected = []
        for arr in shuffled_by_label.values():
            k = max(1, int(round(len(arr) * ratio)))
            selected.extend(arr[:k].tolist())
        out[ratio] = sorted(selected)
    return out


def _downsample_train_indices(
    df,
    train_indices: Sequence[int],
    *,
    target_num: int,
    seed: int,
    unit: str = "subject_label",
):
    if int(target_num) <= 0:
        raise ValueError(f"target_num must be positive, got {target_num}")
    if unit != "subject_label":
        raise ValueError(f"Unsupported downsample unit={unit!r}; currently only subject_label is implemented.")
    if "subject_id" not in df.columns:
        raise RuntimeError("subject_id is required for subject_label downsampling.")

    rng = np.random.default_rng(int(seed))
    by_unit = defaultdict(list)
    for ridx in train_indices:
        row = df.loc[int(ridx)]
        key = (str(row["subject_id"]), int(row["_label"]))
        by_unit[key].append(int(ridx))

    selected = []
    per_unit = {}
    for key, idxs in sorted(by_unit.items()):
        arr = np.asarray(idxs, dtype=np.int64)
        before = int(len(arr))
        rng.shuffle(arr)
        keep = arr[: min(before, int(target_num))].tolist()
        selected.extend(keep)
        per_unit[f"{key[0]}::{key[1]}"] = {
            "before": before,
            "after": int(len(keep)),
        }

    return sorted(selected), per_unit


def _write_idx(path: str, indices: Sequence[int]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for idx in indices:
            f.write(f"{int(idx)}\n")


def _label_counts(df, indices: Sequence[int]) -> dict[str, int]:
    return {str(k): int(v) for k, v in Counter(df.loc[list(indices), "_label"].astype(int).tolist()).items()}


def _split_summary(df, split_indices: dict[str, list[int]], mode: str, trial_col: Optional[str]):
    counts = {}
    for split_name, idxs in split_indices.items():
        part = df.loc[idxs]
        counts[split_name] = {
            "samples": int(len(part)),
            "subjects": int(part["subject_id"].nunique()) if "subject_id" in part.columns else None,
            "trials": int(part[trial_col].nunique()) if trial_col is not None and trial_col in part.columns else None,
            "labels": _label_counts(df, idxs) if idxs else {},
        }

    leakage = {}
    if "subject_id" in df.columns:
        subj_sets = {
            name: set(df.loc[idxs, "subject_id"].astype(str).tolist())
            for name, idxs in split_indices.items()
        }
        leakage["subject_overlap"] = {
            "train_val": bool(subj_sets["train"] & subj_sets["val"]),
            "train_test": bool(subj_sets["train"] & subj_sets["test"]),
            "val_test": bool(subj_sets["val"] & subj_sets["test"]),
            "allowed": mode == "within_subject",
        }
    if trial_col is not None and trial_col in df.columns and "subject_id" in df.columns:
        trial_sets = {}
        for name, idxs in split_indices.items():
            part = df.loc[idxs]
            trial_sets[name] = set(
                (str(s), str(t))
                for s, t in zip(part["subject_id"].tolist(), part[trial_col].tolist())
            )
        leakage["subject_trial_overlap"] = {
            "train_val": bool(trial_sets["train"] & trial_sets["val"]),
            "train_test": bool(trial_sets["train"] & trial_sets["test"]),
            "val_test": bool(trial_sets["val"] & trial_sets["test"]),
            "allowed": False,
        }
    return counts, leakage


def load_zarr_split_manifest(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_zarr_split_manifest(
    zarr_dataset_dir: str,
    *,
    output_dir: str,
    mode: str,
    ratios=(0.8, 0.1, 0.1),
    seed: int = 42,
    filters: Optional[dict[str, Any]] = None,
    ratio_list: Optional[Sequence[float]] = None,
    ratio_seed: Optional[int] = None,
    small_subject_policy: str = "warn_train",
    downsample: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    zarr_path, index_path = resolve_zarr_paths(zarr_dataset_dir)
    index_path, df = ensure_zarr_index_aligned(zarr_path, index_path)
    if len(df) == 0:
        raise RuntimeError(f"Empty sample_index.parquet: {index_path}")

    label_col, label_fn = find_label_column(df)
    df = df.copy()
    df["_zarr_index"] = np.arange(len(df), dtype=np.int64)
    df["_label"] = [label_fn(v) for v in df[label_col].tolist()]
    df.index = df["_zarr_index"].astype(int)

    filters = filters or {}
    df = _filter_df(df, filters)
    if len(df) == 0:
        raise RuntimeError("All Zarr samples were filtered out.")

    mode = str(mode).strip().lower()
    ratios = _normalize_ratios(ratios)

    if mode == "cross_subject":
        split_indices = _cross_subject_split(df, ratios, seed)
        trial_col = _trial_column(df) if any(k in df.columns for k in ("trial_id", "session_id", "source_relpath")) else None
    elif mode == "within_subject":
        trial_col = _trial_column(df)
        split_indices = _within_subject_split(df, ratios, seed, small_subject_policy)
    elif mode in {"predefined", "from_index", "index"}:
        split_indices = _predefined_split(df)
        trial_col = _trial_column(df) if any(k in df.columns for k in ("trial_id", "session_id", "source_relpath")) else None
    else:
        raise ValueError(
            f"Unknown Zarr split mode={mode!r}; expected cross_subject, within_subject, or predefined."
        )

    for split_name, idxs in split_indices.items():
        if not idxs:
            raise RuntimeError(f"Zarr split {split_name!r} is empty after mode={mode}.")

    downsample = downsample or {}
    downsample_mode = str(downsample.get("mode", "off")).strip().lower()
    downsample_enabled = downsample_mode in {"on", "true", "always", "yes"}
    downsample_info = {
        "enabled": bool(downsample_enabled),
        "mode": downsample_mode,
        "apply_to": "train",
        "unit": str(downsample.get("unit", "subject_label")),
        "target_num": None,
        "seed": None,
        "train_before": int(len(split_indices["train"])),
        "train_after": int(len(split_indices["train"])),
    }
    if downsample_enabled:
        target_num = int(downsample.get("target_num", 40))
        downsample_seed = int(downsample.get("seed", seed))
        train_after, per_unit = _downsample_train_indices(
            df,
            split_indices["train"],
            target_num=target_num,
            seed=downsample_seed,
            unit=str(downsample.get("unit", "subject_label")),
        )
        split_indices["train"] = train_after
        downsample_info.update({
            "target_num": target_num,
            "seed": downsample_seed,
            "train_after": int(len(train_after)),
            "per_unit": per_unit,
        })
        print(
            f"[ZarrSplit] downsample train subject_label target_num={target_num}: "
            f"{downsample_info['train_before']} -> {downsample_info['train_after']}"
        )

    counts, leakage = _split_summary(df, split_indices, mode, trial_col)
    os.makedirs(output_dir, exist_ok=True)
    for split_name, idxs in split_indices.items():
        _write_idx(os.path.join(output_dir, f"{split_name}_idx.txt"), idxs)

    ratio_files = {}
    if ratio_list:
        ratio_seed = int(seed if ratio_seed is None else ratio_seed)
        ratio_splits = _stratified_ratio_splits(df, split_indices["train"], ratio_list, ratio_seed)
        for ratio, selected in ratio_splits.items():
            tag = ratio_to_tag(float(ratio))
            filename = f"train_idx_ratio_{tag}.txt"
            _write_idx(os.path.join(output_dir, filename), selected)
            ratio_files[str(float(ratio))] = {
                "file": filename,
                "samples": int(len(selected)),
                "labels": _label_counts(df, selected),
                "nested": True,
            }

    manifest = {
        "version": 1,
        "backend": "zarr",
        "zarr_dataset_dir": os.path.abspath(str(zarr_dataset_dir)),
        "sample_index": os.path.abspath(index_path),
        "mode": mode,
        "seed": int(seed),
        "ratios": [float(x) for x in ratios],
        "filters": filters,
        "trial_column": trial_col,
        "small_subject_policy": small_subject_policy,
        "splits": {k: [int(x) for x in v] for k, v in split_indices.items()},
        "counts": counts,
        "ratio_files": ratio_files,
        "ratio_nested": bool(ratio_list),
        "downsample": downsample_info,
        "leakage_checks": leakage,
    }
    manifest_path = os.path.join(output_dir, "split_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[ZarrSplit] mode={mode} seed={seed} ratios={manifest['ratios']}")
    print(f"[ZarrSplit] counts={json.dumps(counts, ensure_ascii=False)}")
    print(f"[ZarrSplit] manifest={manifest_path}")
    return manifest


def build_zarr_split_manifest_from_cfg(cfg, *, ratio_list: Optional[Sequence[float]] = None) -> dict[str, Any]:
    zarr_root = (
        OmegaConf.select(cfg, "paths.zarr_dataset_name")
        or OmegaConf.select(cfg, "paths.zarr_dataset_dir")
        or OmegaConf.select(cfg, "paths.dataset_name")
    )
    if zarr_root is None:
        raise ValueError("Missing Zarr dataset path in config.")

    mode = (
        OmegaConf.select(cfg, "dataset.eval_mode")
        or OmegaConf.select(cfg, "subject_test_mode")
        or "cross_subject"
    )
    split_cfg = OmegaConf.select(cfg, "dataset.split") or {}
    filter_cfg = OmegaConf.select(cfg, "dataset.filter") or {}
    downsample_cfg = OmegaConf.select(cfg, "dataset.downsample") or {}
    split_dict = OmegaConf.to_container(split_cfg, resolve=True) if not isinstance(split_cfg, dict) else split_cfg
    filter_dict = OmegaConf.to_container(filter_cfg, resolve=True) if not isinstance(filter_cfg, dict) else filter_cfg
    downsample_dict = OmegaConf.to_container(downsample_cfg, resolve=True) if not isinstance(downsample_cfg, dict) else downsample_cfg

    return build_zarr_split_manifest(
        str(zarr_root),
        output_dir=str(OmegaConf.select(cfg, "paths.data_index_dir")),
        mode=str(mode),
        ratios=split_dict.get("ratios", [0.8, 0.1, 0.1]),
        seed=int(split_dict.get("seed", OmegaConf.select(cfg, "dataset.zarr_split_seed") or cfg.seed)),
        filters=filter_dict,
        ratio_list=ratio_list,
        ratio_seed=int(OmegaConf.select(cfg, "seed")),
        small_subject_policy=str(split_dict.get("small_subject_policy", "warn_train")),
        downsample=downsample_dict,
    )
