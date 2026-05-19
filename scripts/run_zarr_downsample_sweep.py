from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys
from typing import Iterable

from omegaconf import OmegaConf

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from configs.config import load_config
from main import main as run_main


def _csv_list(value: str) -> list[str]:
    return [x.strip() for x in str(value).split(",") if x.strip()]


def _int_list(value: str) -> list[int]:
    return [int(x) for x in _csv_list(value)]


def _list_value(value, cast=str) -> list:
    if value is None:
        return []
    if isinstance(value, str):
        values = _csv_list(value)
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        values = [value]
    return [cast(v) for v in values]


def _cfg_value(cfg, key: str, default=None):
    if cfg is None:
        return default
    value = OmegaConf.select(cfg, key)
    if value is None:
        return default
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _mean(values: list[float]) -> float | None:
    values = [float(v) for v in values if v is not None]
    if not values:
        return None
    return sum(values) / len(values)


def _std(values: list[float]) -> float | None:
    values = [float(v) for v in values if v is not None]
    if len(values) == 0:
        return None
    if len(values) == 1:
        return 0.0
    m = sum(values) / len(values)
    return (sum((v - m) ** 2 for v in values) / (len(values) - 1)) ** 0.5


def _set_dataset_paths(cfg, dataset: str, zarr_root_template: str) -> None:
    root = zarr_root_template.format(dataset=dataset)
    cfg.dataset_name = dataset
    cfg.paths.dataset_dir = root
    cfg.paths.dataset_name = os.path.join(root, dataset)
    cfg.paths.zarr_dataset_dir = root
    cfg.paths.zarr_dataset_name = root


def _summary_path(output_base: str, target_num: int, model: str) -> str:
    return os.path.join(
        f"{output_base}_downsample_t{int(target_num)}",
        "ratio_full",
        model,
        "summary.json",
    )


def _read_summary(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_jsonl(path: str, records: Iterable[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_csv(path: str, records: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = [
        "dataset",
        "eval_mode",
        "model",
        "target_num",
        "seed",
        "tuning_mode",
        "batch_size",
        "epochs",
        "best_val",
        "test_accuracy",
        "test_balanced_accuracy",
        "test_f1_macro",
        "test_loss",
        "train_before_downsample",
        "train_after_downsample",
        "summary_path",
        "manifest_path",
        "output_dir",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def aggregate_records(records: list[dict]) -> list[dict]:
    groups = {}
    for record in records:
        if record.get("status") != "ok":
            continue
        key = (
            record.get("dataset"),
            record.get("eval_mode"),
            record.get("model"),
            int(record.get("target_num")),
            record.get("tuning_mode"),
        )
        groups.setdefault(key, []).append(record)

    metrics = [
        "best_val",
        "test_accuracy",
        "test_balanced_accuracy",
        "test_f1_macro",
        "test_loss",
        "train_after_downsample",
    ]
    out = []
    for (dataset, eval_mode, model, target_num, tuning_mode), rows in sorted(groups.items()):
        agg = {
            "dataset": dataset,
            "eval_mode": eval_mode,
            "model": model,
            "target_num": target_num,
            "tuning_mode": tuning_mode,
            "n_seeds": len(rows),
            "seeds": ",".join(str(r.get("seed")) for r in rows),
        }
        for metric in metrics:
            vals = [r.get(metric) for r in rows if r.get(metric) is not None]
            agg[f"{metric}_mean"] = _mean(vals)
            agg[f"{metric}_std"] = _std(vals)
        out.append(agg)
    return out


def _write_aggregate_csv(path: str, records: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = [
        "dataset",
        "eval_mode",
        "model",
        "target_num",
        "tuning_mode",
        "n_seeds",
        "seeds",
        "best_val_mean",
        "best_val_std",
        "test_accuracy_mean",
        "test_accuracy_std",
        "test_balanced_accuracy_mean",
        "test_balanced_accuracy_std",
        "test_f1_macro_mean",
        "test_f1_macro_std",
        "test_loss_mean",
        "test_loss_std",
        "train_after_downsample_mean",
        "train_after_downsample_std",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def _dataset_report_dir(args, dataset: str) -> str:
    return os.path.join(args.output_root, dataset, args.eval_mode, args.tuning_mode)


def _combined_report_dir(args) -> str:
    return os.path.join(args.output_root, "_combined", args.eval_mode, args.tuning_mode)


def _write_reports(report_dir: str, records: list[dict]) -> None:
    aggregate = aggregate_records(records)
    _write_jsonl(os.path.join(report_dir, "records.jsonl"), records)
    _write_csv(os.path.join(report_dir, "records.csv"), records)
    _write_jsonl(os.path.join(report_dir, "aggregate.jsonl"), aggregate)
    _write_aggregate_csv(os.path.join(report_dir, "aggregate.csv"), aggregate)


def build_cfg_for_run(base_cfg, args, dataset: str, target_num: int):
    cfg = copy.deepcopy(base_cfg)
    _set_dataset_paths(cfg, dataset, args.zarr_root_template)

    cfg.project_name = args.project_name
    cfg.exp_name = args.exp_name
    cfg.subject_test_mode = args.eval_mode
    OmegaConf.update(cfg, "dataset.backend", "zarr", force_add=True)
    OmegaConf.update(cfg, "dataset.eval_mode", args.eval_mode, force_add=True)
    OmegaConf.update(cfg, "dataset.downsample.mode", "on", force_add=True)
    OmegaConf.update(cfg, "dataset.downsample.target_num", int(target_num), force_add=True)
    OmegaConf.update(cfg, "dataset.downsample.seed", int(args.current_seed), force_add=True)
    OmegaConf.update(cfg, "dataset.downsample.unit", "subject_label", force_add=True)
    OmegaConf.update(cfg, "dataset.downsample.apply_to", "train", force_add=True)
    run_split_seed = int(args.current_seed) if bool(getattr(args, "vary_split_seed", False)) else int(args.split_seed)
    OmegaConf.update(cfg, "dataset.zarr_split_seed", run_split_seed, force_add=True)
    OmegaConf.update(cfg, "dataset.split.seed", run_split_seed, force_add=True)
    OmegaConf.update(cfg, "dataset.split.ratios", [float(x) for x in args.split_ratios], force_add=True)
    OmegaConf.update(cfg, "dataset.split.reuse_manifest", False, force_add=True)

    cfg.model.names = list(args.models)
    cfg.train.tuning_mode = args.tuning_mode
    cfg.train.ratio_shot = None
    OmegaConf.update(cfg, "train.vary_split_seed", bool(getattr(args, "vary_split_seed", False)), force_add=True)
    # Clear seed_list so main() runs exactly one seed (cfg.seed) per call.
    # Without this, a base config with train.seed_list triggers the MultiSeed loop
    # inside main(), creating an extra seed_N/ subdirectory layer in the output path.
    OmegaConf.update(cfg, "train.seed_list", None, force_add=True)
    cfg.train.use_wandb = False
    cfg.seed = int(args.current_seed)
    if args.cuda is not None:
        cfg.cuda = int(args.cuda)
    if args.batch_size is not None:
        cfg.train.batch_size = int(args.batch_size)
    if args.epochs is not None:
        cfg.train.epochs = int(args.epochs)
    if args.num_workers is not None:
        cfg.train.num_workers = int(args.num_workers)
    if args.eval_freq is not None:
        cfg.train.eval_freq = int(args.eval_freq)

    run_root = os.path.join(
        args.output_root,
        dataset,
        args.eval_mode,
        args.tuning_mode,
        f"seed_{int(args.current_seed)}",
    )
    index_root = os.path.join(
        args.output_root,
        "_indices",
        dataset,
        args.eval_mode,
        f"seed_{int(args.current_seed)}",
    )
    cfg.paths.output_dir = run_root
    cfg.paths.data_index_dir = index_root
    if OmegaConf.select(cfg, "paths.model_output_dir") is not None:
        cfg.paths.model_output_dir = None
    return cfg, run_root, index_root


def collect_records(args, dataset: str, target_num: int, output_base: str, index_base: str) -> list[dict]:
    manifest_path = os.path.join(
        f"{index_base}_downsample_t{int(target_num)}",
        "split_manifest.json",
    )
    manifest = {}
    if os.path.exists(manifest_path):
        manifest = _read_summary(manifest_path)
    downsample = manifest.get("downsample", {})

    records = []
    for model in args.models:
        summary_path = _summary_path(output_base, target_num, model)
        if not os.path.exists(summary_path):
            records.append({
                "dataset": dataset,
                "eval_mode": args.eval_mode,
                "model": model,
                "target_num": int(target_num),
                "seed": int(args.current_seed),
                "tuning_mode": args.tuning_mode,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "summary_path": summary_path,
                "manifest_path": manifest_path,
                "output_dir": os.path.dirname(summary_path),
                "status": "missing_summary",
            })
            continue

        summary = _read_summary(summary_path)
        test_metrics = summary.get("test_metrics") or {}
        records.append({
            "dataset": dataset,
            "eval_mode": args.eval_mode,
            "model": model,
            "target_num": int(target_num),
            "seed": int(args.current_seed),
            "tuning_mode": args.tuning_mode,
            "batch_size": int(args.batch_size) if args.batch_size is not None else None,
            "epochs": int(args.epochs) if args.epochs is not None else None,
            "best_val": summary.get("best_val"),
            "test_accuracy": test_metrics.get("accuracy"),
            "test_balanced_accuracy": test_metrics.get("balanced_accuracy"),
            "test_f1_macro": test_metrics.get("f1_macro"),
            "test_loss": summary.get("test_loss"),
            "train_before_downsample": downsample.get("train_before"),
            "train_after_downsample": downsample.get("train_after"),
            "summary_path": summary_path,
            "manifest_path": manifest_path,
            "output_dir": os.path.dirname(summary_path),
            "status": "ok",
        })
    return records


def parse_args():
    parser = argparse.ArgumentParser(description="Run Zarr train downsample target_num sweeps.")
    parser.add_argument("--sweep-config", default=None, help="YAML file with sweep settings.")
    parser.add_argument("--config", default=None, help="Base training config; overrides sweep_config.base_config.")
    parser.add_argument("--datasets", default=None, help="Comma-separated dataset names, e.g. SSVEP,Physionet_MI")
    parser.add_argument("--models", default=None, help="Comma-separated model names, e.g. eegnet,biot,cbramod")
    parser.add_argument("--target-nums", default=None, help="Comma-separated target_num values, e.g. 5,10,20,40")
    parser.add_argument("--zarr-root-template", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--eval-mode", default=None, choices=["cross_subject", "within_subject"])
    parser.add_argument("--split-ratios", default=None)
    parser.add_argument("--split-seed", type=int, default=None, help="Fixed seed for train/val/test split; keep constant across --seeds.")
    parser.add_argument("--vary-split-seed", action="store_true", help="Use each run seed for train/val/test split; overrides fixed split-seed behavior.")
    parser.add_argument("--seeds", default=None, help="Comma-separated seeds, e.g. 42,2024,3407")
    parser.add_argument("--tuning-mode", default=None, choices=["linear_probing", "full_finetune", "zero_shot"])
    parser.add_argument("--project-name", default=None)
    parser.add_argument("--exp-name", default=None)
    parser.add_argument("--cuda", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--eval-freq", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Only print planned runs; do not train.")
    args = parser.parse_args()

    sweep_cfg = OmegaConf.load(args.sweep_config) if args.sweep_config else None

    args.config = args.config or _cfg_value(sweep_cfg, "base_config", "configs/example.yaml")
    args.datasets = _list_value(args.datasets if args.datasets is not None else _cfg_value(sweep_cfg, "datasets"), str)
    args.models = [m.lower() for m in _list_value(args.models if args.models is not None else _cfg_value(sweep_cfg, "models"), str)]
    args.target_nums = _list_value(
        args.target_nums if args.target_nums is not None else _cfg_value(sweep_cfg, "target_nums"),
        int,
    )
    args.seeds = _list_value(args.seeds if args.seeds is not None else _cfg_value(sweep_cfg, "seeds", [42]), int)
    args.zarr_root_template = args.zarr_root_template or _cfg_value(
        sweep_cfg, "zarr_root_template", "/zongsheng-group/wsn/{dataset}_wsn/"
    )
    args.output_root = args.output_root or _cfg_value(sweep_cfg, "output_root", "./downsample_sweeps")
    args.eval_mode = args.eval_mode or _cfg_value(sweep_cfg, "eval_mode", "cross_subject")
    args.split_ratios = _list_value(
        args.split_ratios if args.split_ratios is not None else _cfg_value(sweep_cfg, "split_ratios", [0.8, 0.1, 0.1]),
        float,
    )
    args.split_seed = args.split_seed if args.split_seed is not None else int(_cfg_value(sweep_cfg, "split_seed", 42))
    args.vary_split_seed = bool(args.vary_split_seed or _cfg_value(sweep_cfg, "vary_split_seed", False))
    args.tuning_mode = args.tuning_mode or _cfg_value(sweep_cfg, "tuning_mode", "linear_probing")
    args.project_name = args.project_name or _cfg_value(sweep_cfg, "project_name", "EEG_Benchmark_5_DownsampleSweep")
    args.exp_name = args.exp_name or _cfg_value(sweep_cfg, "exp_name", "downsample_sweep")
    args.cuda = args.cuda if args.cuda is not None else _cfg_value(sweep_cfg, "cuda")
    args.batch_size = args.batch_size if args.batch_size is not None else _cfg_value(sweep_cfg, "batch_size")
    args.epochs = args.epochs if args.epochs is not None else _cfg_value(sweep_cfg, "epochs")
    args.num_workers = args.num_workers if args.num_workers is not None else _cfg_value(sweep_cfg, "num_workers", 0)
    args.eval_freq = args.eval_freq if args.eval_freq is not None else _cfg_value(sweep_cfg, "eval_freq")
    args.dry_run = bool(args.dry_run or _cfg_value(sweep_cfg, "dry_run", False))

    if not args.datasets:
        raise ValueError("No datasets specified. Set `datasets` in --sweep-config or pass --datasets.")
    if not args.models:
        raise ValueError("No models specified. Set `models` in --sweep-config or pass --models.")
    if not args.target_nums:
        raise ValueError("No target_nums specified. Set `target_nums` in --sweep-config or pass --target-nums.")
    if len(args.split_ratios) != 3:
        raise ValueError("--split-ratios must contain exactly 3 values")
    return args


def main():
    args = parse_args()
    base_cfg = load_config(args.config)
    all_records = []
    records_by_dataset = {dataset: [] for dataset in args.datasets}

    print("[Sweep] datasets=", args.datasets)
    print("[Sweep] models=", args.models)
    print("[Sweep] target_nums=", args.target_nums)
    print("[Sweep] seeds=", args.seeds)
    print("[Sweep] output_root=", args.output_root)
    print("[Sweep] vary_split_seed=", args.vary_split_seed, "split_seed=", args.split_seed)

    for dataset in args.datasets:
        for seed in args.seeds:
            args.current_seed = int(seed)
            for target_num in args.target_nums:
                cfg, output_base, index_base = build_cfg_for_run(base_cfg, args, dataset, target_num)
                print(
                    f"[Sweep] dataset={dataset} seed={seed} target_num={target_num} "
                    f"models={args.models} output_base={output_base}"
                )
                if not args.dry_run:
                    run_main(cfg)
                    new_records = collect_records(args, dataset, target_num, output_base, index_base)
                    all_records.extend(new_records)
                    records_by_dataset.setdefault(dataset, []).extend(new_records)
                    _write_reports(_dataset_report_dir(args, dataset), records_by_dataset[dataset])
                    _write_reports(_combined_report_dir(args), all_records)

    if args.dry_run:
        print("[Sweep] dry-run complete.")
    else:
        for dataset in args.datasets:
            report_dir = _dataset_report_dir(args, dataset)
            print(f"[Sweep] wrote dataset records: {os.path.join(report_dir, 'records.csv')}")
            print(f"[Sweep] wrote dataset aggregate: {os.path.join(report_dir, 'aggregate.csv')}")
        combined_dir = _combined_report_dir(args)
        print(f"[Sweep] wrote combined records: {os.path.join(combined_dir, 'records.csv')}")
        print(f"[Sweep] wrote combined aggregate: {os.path.join(combined_dir, 'aggregate.csv')}")


if __name__ == "__main__":
    main()
