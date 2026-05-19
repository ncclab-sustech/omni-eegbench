from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import os
import traceback
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, List

import torch
from omegaconf import OmegaConf

from configs.config import load_config
from data.make_indices_fixed import generate_indices
from data.make_indices_downsample_fixed import generate_indices_downsample
from main import _build_loaders, _resolve_device, _set_seed, run_one_model
from models.factory import get_model
from tuning import setup_training_mode


DATASET_ROOT = "/zongsheng-group/new_h5"
DEFAULT_DATASETS = ["AD65", "BCI2A", "Physionet_MI", "SEEDIV", "TUSL"]
DEFAULT_MODELS = ["cbramod", "biot", "brainomni", "femba", "neurogpt", "labram", "eegmamba", "neurolm", "bendr"]
RESULT_ROOT = "./matrix_results"
DISEASE_DATASETS = {"AD65", "TUSL"}
MATRIX_EVAL_FREQ = 5
MATRIX_NUM_WORKERS = 0

DATASET_ALIASES = {
    "BCI2A": "BCIC2A",
}


@dataclass(frozen=True)
class Experiment:
    dataset_label: str
    dataset_dir_name: str
    model: str
    subject_mode: str
    use_downsample: bool
    ratio_shot: float | None
    noise_ratio: float | None

    @property
    def setting_name(self) -> str:
        mode_tag = "linear_within" if self.subject_mode == "within_subject" else "linear_cross"
        ds_tag = "downsample" if self.use_downsample else "no_downsample"
        base = f"{mode_tag}_{ds_tag}"
        if self.ratio_shot is not None:
            return f"{base}_kshot_{str(float(self.ratio_shot)).replace('.', 'p')}"
        if self.noise_ratio is not None:
            return f"{base}_channelmask_{str(float(self.noise_ratio)).replace('.', 'p')}"
        return base

    @property
    def output_dir(self) -> str:
        return os.path.join(RESULT_ROOT, self.dataset_label, self.model, self.setting_name)

    @property
    def dataset_path(self) -> str:
        return os.path.join(DATASET_ROOT, self.dataset_dir_name)

    @property
    def requires_channel_mask(self) -> bool:
        return self.noise_ratio is not None


def resolve_dataset_dir_name(dataset_label: str) -> str:
    actual = DATASET_ALIASES.get(dataset_label, dataset_label)
    path = os.path.join(DATASET_ROOT, actual)
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Dataset directory not found: {path}")
    return actual


def build_experiments(datasets: Iterable[str], models: Iterable[str]) -> List[Experiment]:
    exps: List[Experiment] = []
    for dataset in datasets:
        dir_name = resolve_dataset_dir_name(dataset)
        for model in models:
            subject_modes = ["cross_subject"] if dataset in DISEASE_DATASETS else ["within_subject", "cross_subject"]
            if dataset in DISEASE_DATASETS:
                print(f"[Plan] Skip within-subject for disease dataset: {dataset}")
            for subject_mode in subject_modes:
                for use_downsample in [True, False]:
                    exps.append(
                        Experiment(
                            dataset_label=dataset,
                            dataset_dir_name=dir_name,
                            model=model,
                            subject_mode=subject_mode,
                            use_downsample=use_downsample,
                            ratio_shot=None,
                            noise_ratio=None,
                        )
                    )

            for ratio in [0.05, 0.1, 0.3, 0.5]:
                exps.append(
                    Experiment(
                        dataset_label=dataset,
                        dataset_dir_name=dir_name,
                        model=model,
                        subject_mode="cross_subject",
                        use_downsample=True,
                        ratio_shot=ratio,
                        noise_ratio=None,
                    )
                )
            for noise in [0.2, 0.4, 0.6]:
                exps.append(
                    Experiment(
                        dataset_label=dataset,
                        dataset_dir_name=dir_name,
                        model=model,
                        subject_mode="cross_subject",
                        use_downsample=True,
                        ratio_shot=None,
                        noise_ratio=noise,
                    )
                )
    return exps


def ensure_indices(dataset_path: str, dataset_dir_name: str, subject_mode: str, use_downsample: bool, seed: int, target_num: int):
    suffix = "_downsample" if use_downsample else ""
    index_dir = os.path.join("indices", f"{dataset_dir_name}_indices_{subject_mode}{suffix}")
    need_generate = (
        (not os.path.exists(index_dir))
        or (not os.listdir(index_dir))
        or (not os.path.exists(os.path.join(index_dir, "train_idx.txt")))
    )
    if not need_generate:
        return index_dir

    os.makedirs(index_dir, exist_ok=True)
    ratio_list = (0.02, 0.05, 0.1, 0.3, 0.5)
    if use_downsample:
        generate_indices_downsample(
            data_dir=dataset_path,
            base_save_dir="indices",
            trial_split_ratio=(0.4, 0.3, 0.3),
            target_num=target_num,
            ratio_list=ratio_list,
            seed=seed,
            custom_label_map=None,
        )
    else:
        generate_indices(
            data_dir=dataset_path,
            base_save_dir="indices",
            trial_split_ratio=(0.4, 0.3, 0.3),
            ratio_list=ratio_list,
            seed=seed,
            custom_label_map=None,
        )
    return index_dir


def make_cfg_for_experiment(base_cfg, exp: Experiment):
    cfg = copy.deepcopy(base_cfg)
    cfg.model.names = [exp.model]
    cfg.model.name = exp.model
    cfg.train.tuning_mode = "linear_probing"
    cfg.train.eval_freq = max(int(OmegaConf.select(cfg, "train.eval_freq") or 1), MATRIX_EVAL_FREQ)
    cfg.train.num_workers = MATRIX_NUM_WORKERS
    cfg.paths.dataset_name = exp.dataset_path
    cfg.paths.data_index_dir = os.path.join("indices", f"{exp.dataset_dir_name}_indices_{exp.subject_mode}")
    cfg.paths.output_dir = exp.output_dir
    cfg.paths.model_output_dir = exp.output_dir
    cfg.dataset.downsample.mode = "on" if exp.use_downsample else "off"
    cfg.train.ratio_shot = exp.ratio_shot
    if exp.requires_channel_mask:
        cfg.dataset.noise.enable = True
        cfg.dataset.noise.mode = "channel_mask"
        cfg.dataset.noise.ratio = float(exp.noise_ratio)
        cfg.dataset.noise.apply_to = "train"
    else:
        cfg.dataset.noise.enable = False
        cfg.dataset.noise.ratio = 0.0
        cfg.dataset.noise.apply_to = "train"
    return cfg


def load_dataset_info(dataset_path: str):
    with open(os.path.join(dataset_path, "dataset_info.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def preflight_print(exp: Experiment):
    info = load_dataset_info(exp.dataset_path)
    sr = float(info["processing"]["target_sampling_rate"])
    channels = info["dataset"]["channels"]
    print(
        f"[Preflight] dataset={exp.dataset_label} model={exp.model} "
        f"sr={sr} channels={len(channels)} setting={exp.setting_name}"
    )
    if exp.model == "cbramod" and int(round(sr)) != 200:
        print(f"  [Info] CBraMod will resample {sr} Hz -> 200 Hz in the wrapper.")
    if exp.model == "brainomni" and int(round(sr)) != 256:
        print(f"  [Info] BrainOmni will resample {sr} Hz -> 256 Hz in the wrapper.")
    if exp.model == "biot":
        print("  [Info] BIOT uses referential names when possible and zero-fills missing BIOT-18 pairs in permissive mode.")


def summary_exists(exp: Experiment) -> bool:
    return os.path.isfile(os.path.join(exp.output_dir, "summary.json"))


def write_jsonl(path: str, record: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def rebuild_setting_leaderboards(dataset_root: str, setting_name: str):
    rows_val = []
    rows_test = []

    for model_name in sorted(os.listdir(dataset_root)):
        model_dir = os.path.join(dataset_root, model_name)
        if not os.path.isdir(model_dir) or model_name.startswith("_"):
            continue
        summary_path = os.path.join(model_dir, setting_name, "summary.json")
        if not os.path.isfile(summary_path):
            continue
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
        except Exception as e:
            print(f"[Leaderboard] Warning: failed to read {summary_path}: {e}")
            continue

        best_val = summary.get("best_val")
        if best_val is not None:
            rows_val.append((summary.get("model", model_name), float(best_val), summary.get("best_path")))
        test_metrics = summary.get("test_metrics") or {}
        if "accuracy" in test_metrics:
            rows_test.append((summary.get("model", model_name), float(test_metrics["accuracy"]), summary.get("best_path")))

    rows_val.sort(key=lambda x: x[1], reverse=True)
    rows_test.sort(key=lambda x: x[1], reverse=True)

    lb_root = os.path.join(dataset_root, "_leaderboards", setting_name)
    os.makedirs(lb_root, exist_ok=True)
    for mode, rows in [("val", rows_val), ("test", rows_test)]:
        payload = [
            {"rank": i + 1, "model": m, f"{mode}_accuracy": acc, "best_ckpt": ckpt}
            for i, (m, acc, ckpt) in enumerate(rows)
        ]
        json_path = os.path.join(lb_root, f"leaderboard_{mode}.json")
        txt_path = os.path.join(lb_root, f"leaderboard_{mode}.txt")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"Leaderboard ({setting_name} {mode} accuracy)\n")
            if not rows:
                f.write("No results found.\n")
            for i, (m, acc, ckpt) in enumerate(rows, 1):
                f.write(f"{i:02d}. {m:<16} acc={acc:.4f} ckpt={ckpt or '-'}\n")


def run_experiment(base_cfg, exp: Experiment, device: torch.device):
    cfg = make_cfg_for_experiment(base_cfg, exp)
    os.makedirs(exp.output_dir, exist_ok=True)

    target_num = int(OmegaConf.select(cfg, "dataset.downsample.target_num") or 40)
    index_dir = ensure_indices(exp.dataset_path, exp.dataset_dir_name, exp.subject_mode, exp.use_downsample, int(cfg.seed), target_num)
    cfg.paths.data_index_dir = index_dir

    preflight_print(exp)
    train_loader, val_loader, test_loader = _build_loaders(cfg, device)
    result = run_one_model(cfg, exp.model, device, train_loader, val_loader, test_loader)
    return result


def prepare_indices_for_experiments(base_cfg, experiments: List[Experiment]):
    prepared = set()
    for exp in experiments:
        target_num = int(OmegaConf.select(base_cfg, "dataset.downsample.target_num") or 40)
        key = (exp.dataset_path, exp.dataset_dir_name, exp.subject_mode, exp.use_downsample, int(base_cfg.seed), target_num)
        if key in prepared:
            continue
        print(
            f"[Indices] ensure dataset={exp.dataset_label} subject_mode={exp.subject_mode} "
            f"downsample={exp.use_downsample}"
        )
        ensure_indices(
            dataset_path=exp.dataset_path,
            dataset_dir_name=exp.dataset_dir_name,
            subject_mode=exp.subject_mode,
            use_downsample=exp.use_downsample,
            seed=int(base_cfg.seed),
            target_num=target_num,
        )
        prepared.add(key)


def _run_worker(
    gpu_id: int,
    config_path: str,
    tasks: List[Experiment],
    skip_existing: bool,
    result_queue,
):
    base_cfg = load_config(config_path)
    base_cfg.cuda = int(gpu_id)
    device = _resolve_device(base_cfg)
    _set_seed(int(base_cfg.seed))

    for exp in tasks:
        if skip_existing and summary_exists(exp):
            result_queue.put({
                "status": "skipped",
                "dataset": exp.dataset_label,
                "model": exp.model,
                "setting": exp.setting_name,
                "output_dir": exp.output_dir,
                "gpu": gpu_id,
            })
            continue
        try:
            print(f"[GPU {gpu_id}] Run {exp.dataset_label}/{exp.model}/{exp.setting_name}")
            run_experiment(base_cfg, exp, device)
            result_queue.put({
                "status": "success",
                "dataset": exp.dataset_label,
                "model": exp.model,
                "setting": exp.setting_name,
                "output_dir": exp.output_dir,
                "gpu": gpu_id,
            })
        except Exception as e:
            result_queue.put({
                "status": "failed",
                "dataset": exp.dataset_label,
                "dataset_dir_name": exp.dataset_dir_name,
                "model": exp.model,
                "setting": exp.setting_name,
                "output_dir": exp.output_dir,
                "gpu": gpu_id,
                "error": str(e),
                "traceback": traceback.format_exc(),
            })


def _run_parallel(config_path: str, experiments: List[Experiment], gpu_ids: List[int], skip_existing: bool):
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    shards = [[] for _ in gpu_ids]

    groups = []
    current_key = None
    current_group = []
    for exp in experiments:
        key = (exp.dataset_label, exp.subject_mode, exp.use_downsample)
        if current_key is None or key == current_key:
            current_group.append(exp)
            current_key = key
        else:
            groups.append(current_group)
            current_group = [exp]
            current_key = key
    if current_group:
        groups.append(current_group)

    shard_sizes = [0 for _ in gpu_ids]
    for group in groups:
        target = min(range(len(gpu_ids)), key=lambda i: shard_sizes[i])
        shards[target].extend(group)
        shard_sizes[target] += len(group)

    procs = []
    for gpu_id, shard in zip(gpu_ids, shards):
        if not shard:
            continue
        p = ctx.Process(
            target=_run_worker,
            args=(gpu_id, config_path, shard, skip_existing, result_queue),
        )
        p.start()
        procs.append(p)

    results = []
    expected = sum(len(shard) for shard in shards)
    for _ in range(expected):
        results.append(result_queue.get())

    for p in procs:
        p.join()
    return results


def smoke_check(base_cfg, experiments: List[Experiment], device: torch.device, batch_size: int = 2):
    print("[Smoke] Starting simple forward-pass check.")
    seen_models = set()
    smoke_results = []
    for exp in experiments:
        if exp.model in seen_models:
            continue
        if exp.dataset_label != "AD65" or exp.subject_mode != "cross_subject" or not exp.use_downsample:
            continue
        seen_models.add(exp.model)
        cfg = make_cfg_for_experiment(base_cfg, exp)
        cfg.train.batch_size = min(int(cfg.train.batch_size), batch_size)
        cfg.train.num_workers = 0
        target_num = int(OmegaConf.select(cfg, "dataset.downsample.target_num") or 40)
        index_dir = ensure_indices(exp.dataset_path, exp.dataset_dir_name, exp.subject_mode, exp.use_downsample, int(cfg.seed), target_num)
        cfg.paths.data_index_dir = index_dir
        try:
            train_loader, _, _ = _build_loaders(cfg, device)
            samples, _ = next(iter(train_loader))
            samples = samples[:batch_size].to(device)
            model = get_model(cfg).to(device)
            model = setup_training_mode(model, cfg)
            model.eval()
            with torch.no_grad():
                out = model(samples)
            smoke_results.append({"model": exp.model, "status": "ok", "output_shape": tuple(out.shape)})
            print(f"[Smoke] {exp.model}: ok output_shape={tuple(out.shape)}")
        except Exception as e:
            smoke_results.append({"model": exp.model, "status": "fail", "error": str(e)})
            print(f"[Smoke] {exp.model}: fail -> {e}")
    return smoke_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Base yaml config")
    parser.add_argument("--smoke-only", action="store_true", help="Only run a lightweight forward-pass smoke check")
    parser.add_argument("--skip-existing", action="store_true", help="Skip experiments whose summary.json already exists")
    parser.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS)
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--gpus", nargs="*", type=int, help="GPU ids to use in parallel, e.g. --gpus 0 1")
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    experiments = build_experiments(args.datasets, args.models)
    print(f"[Plan] Total experiments: {len(experiments)}")
    for exp in experiments:
        print(f"[Experiment] {exp.dataset_label}/{exp.model}/{exp.setting_name} -> {exp.output_dir}")

    if args.smoke_only:
        device = _resolve_device(base_cfg)
        _set_seed(int(base_cfg.seed))
        smoke_check(base_cfg, experiments, device)
        return

    prepare_indices_for_experiments(base_cfg, experiments)
    errors_path = os.path.join(RESULT_ROOT, "matrix_errors.jsonl")
    stats = {"success": 0, "failed": 0, "skipped": 0}
    per_dataset = defaultdict(lambda: {"success": 0, "failed": 0, "skipped": 0})
    gpu_ids = args.gpus if args.gpus else [int(OmegaConf.select(base_cfg, "cuda") or 0)]
    print(f"[Schedule] Using GPUs: {gpu_ids}")

    if len(gpu_ids) == 1:
        base_cfg.cuda = int(gpu_ids[0])
        device = _resolve_device(base_cfg)
        _set_seed(int(base_cfg.seed))
        for exp in experiments:
            if args.skip_existing and summary_exists(exp):
                print(f"[Skip] Existing summary found: {exp.output_dir}")
                stats["skipped"] += 1
                per_dataset[exp.dataset_label]["skipped"] += 1
                continue
            try:
                print(f"[Run][GPU {gpu_ids[0]}] {exp.dataset_label}/{exp.model}/{exp.setting_name}")
                run_experiment(base_cfg, exp, device)
                stats["success"] += 1
                per_dataset[exp.dataset_label]["success"] += 1
            except Exception as e:
                stats["failed"] += 1
                per_dataset[exp.dataset_label]["failed"] += 1
                record = {
                    "dataset": exp.dataset_label,
                    "dataset_dir_name": exp.dataset_dir_name,
                    "model": exp.model,
                    "setting": exp.setting_name,
                    "output_dir": exp.output_dir,
                    "gpu": gpu_ids[0],
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                }
                write_jsonl(errors_path, record)
                print(f"[Error] {exp.dataset_label}/{exp.model}/{exp.setting_name}: {e}")
    else:
        results = _run_parallel(args.config, experiments, gpu_ids, args.skip_existing)
        for record in results:
            status = record["status"]
            dataset = record["dataset"]
            stats[status] += 1
            per_dataset[dataset][status] += 1
            if status == "failed":
                write_jsonl(errors_path, record)
                print(
                    f"[Error][GPU {record['gpu']}] "
                    f"{record['dataset']}/{record['model']}/{record['setting']}: {record['error']}"
                )
            elif status == "skipped":
                print(f"[Skip][GPU {record['gpu']}] Existing summary found: {record['output_dir']}")

    rebuild_targets = sorted({(exp.dataset_label, exp.setting_name) for exp in experiments})
    for dataset_label, setting_name in rebuild_targets:
        rebuild_setting_leaderboards(os.path.join(RESULT_ROOT, dataset_label), setting_name)

    print(f"[Done] success={stats['success']} failed={stats['failed']} skipped={stats['skipped']}")
    for dataset, ds_stats in sorted(per_dataset.items()):
        print(
            f"[Dataset Summary] {dataset}: "
            f"success={ds_stats['success']} failed={ds_stats['failed']} skipped={ds_stats['skipped']}"
        )


if __name__ == "__main__":
    main()
