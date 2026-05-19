from __future__ import annotations

import argparse
import copy
import gc
import os
import sys
import traceback
from typing import Any

from omegaconf import ListConfig, OmegaConf

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from main import main as run_main
from scripts.run_dataset_list import _build_dataset_cfg

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


def parse_args():
    parser = argparse.ArgumentParser(description="Run linear probing sequentially across multiple datasets.")
    parser.add_argument("config", nargs="?", default="configs/example.yaml", help="Base linear_prob YAML config.")
    parser.add_argument("--datasets", default=None, help="Comma-separated dataset list overriding config.dataset_name.")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved per-dataset configs without training.")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop immediately if one dataset fails.")
    parser.add_argument(
        "--keep-ratio-shot",
        action="store_true",
        help="Keep config.train.ratio_shot. By default linear_prob uses full train data (ratio_shot=null).",
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

    print(f"[LinearProbList] config={args.config}")
    print(f"[LinearProbList] datasets={datasets}")

    successes = []
    failures = []
    for dataset_name in datasets:
        try:
            cfg = _build_dataset_cfg(base_cfg, dataset_name)
            cfg.exp_name = str(OmegaConf.select(cfg, "exp_name") or "linear_prob")
            cfg.train.tuning_mode = "linear_probing"
            if not args.keep_ratio_shot:
                cfg.train.ratio_shot = None

            print("\n" + "#" * 24 + f" Dataset: {dataset_name} " + "#" * 24)
            print(f"[LinearProbList] output_dir={cfg.paths.output_dir}")
            print(f"[LinearProbList] data_index_dir={cfg.paths.data_index_dir}")
            print(f"[LinearProbList] zarr_dataset_name={OmegaConf.select(cfg, 'paths.zarr_dataset_name')}")
            print(f"[LinearProbList] models={list(cfg.model.names)}")
            print(f"[LinearProbList] seed_list={OmegaConf.select(cfg, 'train.seed_list')}")
            print(f"[LinearProbList] ratio_shot={OmegaConf.select(cfg, 'train.ratio_shot')}")

            if not args.dry_run:
                run_main(cfg)
            successes.append(str(dataset_name))
        except (Exception, SystemExit) as e:
            failures.append({"dataset": str(dataset_name), "error": repr(e)})
            print(f"[LinearProbList] ERROR dataset={dataset_name}: {e}")
            traceback.print_exc()
            gc.collect()
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
            if args.stop_on_error:
                raise
            print("[LinearProbList] continuing with next dataset...")

    if args.dry_run:
        print("[LinearProbList] dry-run complete.")
    print(f"[LinearProbList] success ({len(successes)}): {successes}")
    print(f"[LinearProbList] failed ({len(failures)}): {failures}")


if __name__ == "__main__":
    main()
