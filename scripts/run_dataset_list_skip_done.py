from __future__ import annotations

import argparse
import copy
import gc
import traceback
import os
import sys
from typing import Any

from omegaconf import DictConfig, ListConfig, OmegaConf

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from main import main as run_main

try:
    import torch
except Exception:
    torch = None


def _csv_list(value: str) -> list[str]:
    return [x.strip() for x in str(value).split(",") if x.strip()]


def _as_dataset_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return _csv_list(value)
    if isinstance(value, (list, tuple, ListConfig)) or OmegaConf.is_list(value):
        return [str(x) for x in value]
    return [str(value)]


def _require(cfg: DictConfig, key: str) -> None:
    if OmegaConf.select(cfg, key) is None:
        raise ValueError(f"[Config] Missing required key: `{key}`")


def _validate_cfg(cfg: DictConfig) -> None:
    for key in ("paths", "dataset", "model", "train"):
        _require(cfg, key)
    for key in (
        "paths.data_index_dir",
        "paths.output_dir",
        "model.names",
        "model.patch_size",
        "model.head.type",
        "train.tuning_mode",
        "train.batch_size",
        "train.epochs",
        "train.num_workers",
        "train.lr",
        "train.weight_decay",
    ):
        _require(cfg, key)


def _build_dataset_cfg(base_cfg: DictConfig, dataset_name: str) -> DictConfig:
    cfg = copy.deepcopy(base_cfg)
    cfg.dataset_name = str(dataset_name)
    OmegaConf.resolve(cfg)
    _validate_cfg(cfg)
    return cfg


def _ratio_to_tag(r: float) -> str:
    s = str(float(r))
    return s.replace(".", "p")


def _get_ratio_tags(cfg: DictConfig) -> list[str]:
    raw = OmegaConf.select(cfg, "train.ratio_shot")
    if raw is None:
        return ["full"]
    if isinstance(raw, (list, tuple, ListConfig)) or OmegaConf.is_list(raw):
        return [_ratio_to_tag(float(r)) for r in raw]
    return [_ratio_to_tag(float(raw))]


def _get_seed_list(cfg: DictConfig) -> list[int]:
    seed_list = OmegaConf.select(cfg, "train.seed_list")
    if seed_list is None:
        seed_list = OmegaConf.select(cfg, "seed_list")
    if seed_list is None:
        return [int(cfg.seed)]
    if isinstance(seed_list, (list, tuple, ListConfig)) or OmegaConf.is_list(seed_list):
        return [int(s) for s in seed_list]
    return [int(seed_list)]


def _is_dataset_complete(cfg: DictConfig) -> tuple[bool, list[str]]:
    """Return (is_complete, list_of_missing_paths).

    Mirrors the directory layout that main.py produces:
      multi-seed:  {output_dir}/seed_{s}[_downsample_t{N}]/ratio_{tag}/{model}/summary.json
      single-seed: {output_dir}[_downsample_t{N}]/ratio_{tag}/{model}/summary.json
    """
    base_output_dir = str(cfg.paths.output_dir)
    backend = str(OmegaConf.select(cfg, "dataset.backend") or "zarr").lower()
    downsample_mode = str(OmegaConf.select(cfg, "dataset.downsample.mode") or "off").strip().lower()
    use_downsample = downsample_mode in {"on", "true", "always", "yes"}
    downsample_target_num = int(OmegaConf.select(cfg, "dataset.downsample.target_num") or 40)

    seed_list = _get_seed_list(cfg)
    ratio_tags = _get_ratio_tags(cfg)
    model_names = [str(n).lower() for n in cfg.model.names]

    def _downsample_suffix() -> str:
        if not use_downsample:
            return ""
        if backend == "zarr":
            return f"_downsample_t{downsample_target_num}"
        return "_downsample"

    suffix = _downsample_suffix()
    is_multi_seed = len(seed_list) > 1

    missing: list[str] = []

    for seed in seed_list:
        if is_multi_seed:
            seed_dir = os.path.join(base_output_dir, f"seed_{seed}{suffix}")
        else:
            seed_dir = base_output_dir + suffix

        for ratio_tag in ratio_tags:
            ratio_dir = os.path.join(seed_dir, f"ratio_{ratio_tag}")

            leaderboard = os.path.join(ratio_dir, "leaderboard_test.json")
            if not os.path.exists(leaderboard):
                missing.append(leaderboard)
                continue

            for model_name in model_names:
                summary = os.path.join(ratio_dir, model_name, "summary.json")
                if not os.path.exists(summary):
                    missing.append(summary)

    return (len(missing) == 0, missing)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run one config sequentially across multiple datasets, skipping those with complete results."
    )
    parser.add_argument("config", help="Base YAML config. `dataset_name` may be a scalar or a list.")
    parser.add_argument(
        "--datasets",
        default=None,
        help="Optional comma-separated dataset list overriding config.dataset_name.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print resolved per-dataset configs without training.")
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately if one dataset fails. By default failures are logged and the next dataset continues.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore completion check and run all datasets unconditionally.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    base_cfg = OmegaConf.load(args.config)
    datasets = _as_dataset_list(args.datasets) if args.datasets is not None else _as_dataset_list(
        OmegaConf.select(base_cfg, "dataset_name")
    )
    if not datasets:
        raise ValueError("No datasets found. Set top-level `dataset_name` or pass --datasets.")

    print(f"[DatasetList] config={args.config}")
    print(f"[DatasetList] datasets ({len(datasets)}): {datasets}")

    successes: list[str] = []
    skipped: list[str] = []
    failures: list[dict] = []

    for dataset_name in datasets:
        try:
            cfg = _build_dataset_cfg(base_cfg, dataset_name)
            print("\n" + "#" * 24 + f" Dataset: {dataset_name} " + "#" * 24)
            print(f"[DatasetList] output_dir={cfg.paths.output_dir}")
            print(f"[DatasetList] data_index_dir={cfg.paths.data_index_dir}")
            print(f"[DatasetList] zarr_dataset_name={OmegaConf.select(cfg, 'paths.zarr_dataset_name')}")
            print(f"[DatasetList] models={list(cfg.model.names)}")
            print(f"[DatasetList] seed_list={OmegaConf.select(cfg, 'train.seed_list')}")
            print(f"[DatasetList] ratio_shot={OmegaConf.select(cfg, 'train.ratio_shot')}")

            if not args.force:
                is_complete, missing = _is_dataset_complete(cfg)
                if is_complete:
                    print(f"[DatasetList] SKIP {dataset_name}: all results present.")
                    skipped.append(str(dataset_name))
                    continue
                sample = missing[:5]
                extra = len(missing) - len(sample)
                print(f"[DatasetList] {len(missing)} missing file(s), proceeding:")
                for p in sample:
                    print(f"  {p}")
                if extra > 0:
                    print(f"  ... and {extra} more")

            if not args.dry_run:
                run_main(cfg)
            successes.append(str(dataset_name))
        except (Exception, SystemExit) as e:
            failures.append({"dataset": str(dataset_name), "error": repr(e)})
            print(f"[DatasetList] ERROR dataset={dataset_name}: {e}")
            traceback.print_exc()
            gc.collect()
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
            if args.stop_on_error:
                raise
            print("[DatasetList] continuing with next dataset...")

    if args.dry_run:
        print("[DatasetList] dry-run complete.")
    print(f"\n[DatasetList] success  ({len(successes)}): {successes}")
    print(f"[DatasetList] skipped  ({len(skipped)}): {skipped}")
    print(f"[DatasetList] failed   ({len(failures)}): {failures}")


if __name__ == "__main__":
    main()
