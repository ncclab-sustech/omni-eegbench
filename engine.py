# engine.py
import math
import sys
import torch
from typing import Iterable, Tuple, Any
import os
import json
from utils import MetricLogger, SmoothedValue, get_metrics
from omegaconf import OmegaConf
from data.metadata import load_dataset_metadata

import torch
import torch.nn.functional as F
import numpy as np


def _balanced_accuracy_from_tensors(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    y_true = y_true.reshape(-1).detach().cpu()
    y_pred = y_pred.reshape(-1).detach().cpu()

    if y_true.numel() == 0:
        return 0.0

    recalls = []
    for cls in torch.unique(y_true):
        mask = (y_true == cls)
        denom = int(mask.sum().item())
        if denom == 0:
            continue
        recalls.append(float((y_pred[mask] == cls).sum().item()) / float(denom))

    if not recalls:
        return 0.0
    return float(sum(recalls) / len(recalls))


def _cfg_get(cfg: Any, key: str, default=None):
    if cfg is None:
        return default

    # OmegaConf DictConfig
    if OmegaConf is not None:
        try:
            v = OmegaConf.select(cfg, key)
            if v is not None:
                return v
        except Exception:
            pass

    # dict
    if isinstance(cfg, dict):
        # support nested a.b.c
        cur = cfg
        for p in key.split("."):
            if not isinstance(cur, dict) or p not in cur:
                return default
            cur = cur[p]
        return cur

    # object attribute
    # support nested a.b.c
    cur = cfg
    for p in key.split("."):
        if not hasattr(cur, p):
            return default
        cur = getattr(cur, p)
    return cur


def _maybe_set_cuda_from_cfg(device: torch.device, cfg=None) -> torch.device:
    cuda_id = _cfg_get(cfg, "cuda", None)
    if cuda_id is None:
        return device

    try:
        cuda_id = int(cuda_id)
    except Exception:
        return device

    if cuda_id >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(cuda_id)
        return torch.device(f"cuda:{cuda_id}")
    return torch.device("cpu")


def _infer_task(cfg) -> Tuple[str, int]:
    """
    task_mod: "Classification" | "Regression"
    nb_classes: from cfg.dataset.num_classes (required for classification)
    """
    dataset_json = load_dataset_metadata(cfg)
    
    task_mod = dataset_json['dataset']['downstream_task']
    
    if task_mod is None:
        task_mod = _cfg_get(cfg, "train.task_mod", "Classification")
    task_mod = str(task_mod)

    # your yaml uses dataset.num_classes
    nb_classes = dataset_json['dataset']['num_labels']
    # if nb_classes is None:
    #     # fallback to old naming if someone still uses it
    #     nb_classes = _cfg_get(cfg, "nb_classes", None)

    if task_mod != "Regression":
        if nb_classes is None:
            raise ValueError("[engine] Missing required config key: dataset.num_classes (for Classification).")
        nb_classes = int(nb_classes)
    else:
        # regression: allow nb_classes absent
        nb_classes = int(nb_classes) if nb_classes is not None else 1

    return task_mod, nb_classes


def _build_criterion(task_mod: str, nb_classes: int, cfg=None):
    """
    Default policy:
      - Regression -> MSE
      - Classification:
          - nb_classes <= 1 -> BCEWithLogits
          - nb_classes >= 2 -> CrossEntropy
    """
    if task_mod == "Regression":
        return torch.nn.MSELoss()

    if nb_classes <= 1:
        return torch.nn.BCEWithLogitsLoss()

    return torch.nn.CrossEntropyLoss()


def _apply_train_mode_for_tuning(model: torch.nn.Module, args=None) -> None:
    model.train(True)
    mode = str(_cfg_get(args, "train.tuning_mode", "") or "").lower()
    if mode in {"linear_probe", "linear_probing", "linear", "probe"}:
        feature_extractor = getattr(model, "feature_extractor", getattr(model, "extractor", None))
        if feature_extractor is not None:
            feature_extractor.eval()
            enable_probe_adapters = getattr(feature_extractor, "enable_linear_probe_trainables", None)
            if callable(enable_probe_adapters):
                enable_probe_adapters()


def train_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler=None,
    max_norm: float = 0.0,
    log_writer=None,
    args=None,   
):
    device = _maybe_set_cuda_from_cfg(device, args)
    use_amp = (device.type == "cuda")

    if optimizer is None:
        raise ValueError("[engine] optimizer is None. If you want zero-shot, skip train_one_epoch in main.py.")

    task_mod, nb_classes = _infer_task(args)
    criterion = _build_criterion(task_mod, nb_classes, args)

    # if max_norm not provided, fallback to yaml train.clip_grad
    if (max_norm is None) or (float(max_norm) == 0.0):
        clip_grad = _cfg_get(args, "train.clip_grad", 0.0)
        try:
            max_norm = float(clip_grad)
        except Exception:
            max_norm = 0.0

    _apply_train_mode_for_tuning(model, args)
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"
    print_freq = len(data_loader)+ 1

    for _, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # ---- target dtype/shape ----
        if task_mod == "Regression":
            targets = targets.float()
            if targets.dim() == 1:
                targets = targets.unsqueeze(-1)
        else:
            if nb_classes <= 1:
                # BCE expects float targets [B] or [B,1]
                targets = targets.float()
                if targets.dim() == 1:
                    targets = targets.unsqueeze(-1)
            else:
                # CE expects long targets [B]
                targets = targets.long()

        optimizer.zero_grad(set_to_none=True)
        
        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(samples)
            loss = criterion(outputs, targets)

        loss_value = float(loss.item())
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        if loss_scaler is not None:
            clip_val = max_norm if (max_norm and max_norm > 0) else None
            loss_scaler(
                loss,
                optimizer,
                clip_grad=clip_val,
                parameters=model.parameters(),
                create_graph=False,
                update_grad=True,
            )
        else:
            loss.backward()
            if max_norm and max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if log_writer is not None:
            log_writer.log(
                {
                    "train/loss": loss_value,
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "train/epoch": epoch,
                }
            )

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(
    data_loader,
    model,
    device,
    log_writer=None,
    epoch=None,
    args=None,   
    split: str = "eval",  #  "val" | "test" | "train" | "eval"
):
    device = _maybe_set_cuda_from_cfg(device, args)
    use_amp = (device.type == "cuda")

    task_mod, nb_classes = _infer_task(args)
    criterion = _build_criterion(task_mod, nb_classes, args)

    # ---- metrics list: MUST match utils.get_metrics keys ----
    if task_mod == "Regression":
        metrics_list = ["Pearson_Correlation", "R2_Score", "RMSE"]
    else:
        if nb_classes <= 1:
            metrics_list = ["accuracy", "balanced_accuracy", "pr_auc", "roc_auc"]
        else:
            metrics_list = ["accuracy", "f1_macro"]

    metric_logger = MetricLogger(delimiter="  ")
    split = str(split).lower().strip()
    header = f"{split.upper()}:"
    model.eval()

    all_outputs = []
    all_targets = []
    print_freq = len(data_loader)+ 1
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if task_mod == "Regression":
            targets = targets.float()
            if targets.dim() == 1:
                targets = targets.unsqueeze(-1)
        else:
            if nb_classes <= 1:
                targets = targets.float()
                if targets.dim() == 1:
                    targets = targets.unsqueeze(-1)
            else:
                targets = targets.long()

        with torch.cuda.amp.autocast(enabled=use_amp):
            output = model(samples)
            loss = criterion(output, targets)

        metric_logger.update(loss=float(loss.item()))

        # store outputs for metrics
        if task_mod == "Regression":
            out_for_metric = output
        else:
            if nb_classes <= 1:
                out_for_metric = torch.sigmoid(output)  
            else:
                out_for_metric = torch.softmax(output, dim=1)

        all_outputs.append(out_for_metric.detach().cpu())
        all_targets.append(targets.detach().cpu())

    all_outputs = torch.cat(all_outputs).numpy()
    all_targets = torch.cat(all_targets).numpy()

    try:
        is_binary = (task_mod != "Regression" and nb_classes <= 1)
        results = get_metrics(all_outputs, all_targets, metrics=metrics_list, is_binary=is_binary)
    except Exception as e:
        print(f"Warning: Metric calculation failed ({e}).")
        results = {}

    primary = "R2_Score" if task_mod == "Regression" else "accuracy"
    primary_val = float(results.get(primary, 0.0))
    print(f"* [{split}] {primary} {primary_val:.4f} Loss {metric_logger.loss.global_avg:.4f}")

    if log_writer is not None and epoch is not None:
        log_data = {f"{split}/loss": metric_logger.loss.global_avg}
        for k, v in results.items():
            log_data[f"{split}/{k}"] = v
        log_writer.log(log_data, step=epoch)

    return {
        "loss": metric_logger.loss.global_avg,
        "metrics": results,
    }



import torch
import torch.nn.functional as F

@torch.no_grad()
def evaluate_embed_nn(
    data_loader,
    model,
    device,
    log_writer=None,
    epoch=None,
    args=None,
    split: str = "eval",
    gallery_loader=None,
    exclude_self: bool = False,
    k: int = 1,
    normalize: bool = True,
    metric: str = "cosine",
    query_chunk: int = 512,
    gallery_chunk: int = 0,
):
    split = str(split).lower().strip()
    if gallery_loader is None:
        gallery_loader = data_loader

    # handle DP/DDP wrappers
    m = model.module if hasattr(model, "module") else model

    if not hasattr(m, "feature_extractor"):
        raise AttributeError("evaluate_embed_nn expects DownstreamModel with attribute `feature_extractor`.")

    m.eval()
    use_amp = (device.type == "cuda")

    def _to_2d(feat: torch.Tensor) -> torch.Tensor:
        """
        DownstreamModel doc says feat should be [B, D],
        but for safety handle [B, N, D] or others.
        """
        if feat.dim() == 2:
            return feat
        if feat.dim() == 3:
            # common token output -> mean pool tokens
            return feat.mean(dim=1)
        return feat.flatten(1)

    def _collect(loader):
        embs, labs = [], []
        for samples, targets in loader:
            samples = samples.to(device, non_blocking=True)
            t = targets.detach().cpu()

            with torch.cuda.amp.autocast(enabled=use_amp):
                feat = m.feature_extractor(samples)

            # extractor might return tuple/list/dict
            if isinstance(feat, (tuple, list)):
                feat = feat[0]
            elif isinstance(feat, dict):
                for key in ["feat", "feature", "features", "emb", "embedding", "x"]:
                    if key in feat and torch.is_tensor(feat[key]):
                        feat = feat[key]
                        break

            if not torch.is_tensor(feat):
                raise TypeError(f"feature_extractor output must be Tensor, got {type(feat)}")

            e = _to_2d(feat).detach().float().cpu()
            embs.append(e)
            labs.append(t)

        E = torch.cat(embs, dim=0)  # [N, D]
        y = torch.cat(labs, dim=0)  # [N] or [N,1]

        if y.ndim > 1 and y.shape[-1] == 1:
            y = y.squeeze(-1)
        if y.dtype in (torch.float16, torch.float32, torch.float64):
            y = y.round().long()
        else:
            y = y.long()

        if normalize:
            E = F.normalize(E, dim=1)

        return E, y

    # ---- collect embeddings ----
    G, y_g = _collect(gallery_loader)
    Q, y_q = _collect(data_loader)

    Ng, Nq = G.shape[0], Q.shape[0]
    k = int(k)
    topk_eval = max(k, 5)

    Q_dev = Q.to(device, non_blocking=True)

    if gallery_chunk <= 0:
        G_dev = G.to(device, non_blocking=True)
        y_g_dev = y_g.to(device, non_blocking=True)
    else:
        G_dev, y_g_dev = None, None

    all_preds = []
    all_truths = []
    correct_top5 = 0

    for qs in range(0, Nq, query_chunk):
        qe = min(qs + query_chunk, Nq)
        q_chunk = Q_dev[qs:qe]
        b = qe - qs

        if gallery_chunk <= 0:
            # cosine if normalize=True; otherwise dot product
            sims = q_chunk @ G_dev.T

            # exclude self only if same loader AND same order AND same size
            # if exclude_self and (gallery_loader is data_loader) and (Ng == Nq):
            if exclude_self and (Ng == Nq):
                rows = torch.arange(b, device=device)
                cols = torch.arange(qs, qe, device=device)
                sims[rows, cols] = -1e9

            topk_idx = torch.topk(sims, k=topk_eval, dim=1, largest=True).indices
            topk_lab = y_g_dev[topk_idx]

        else:
            best_vals = torch.full((b, topk_eval), -1e9, device=device)
            best_idx  = torch.full((b, topk_eval), -1, device=device, dtype=torch.long)

            for gs in range(0, Ng, gallery_chunk):
                ge = min(gs + gallery_chunk, Ng)
                g_chunk = G[gs:ge].to(device, non_blocking=True)

                sims = q_chunk @ g_chunk.T

                if exclude_self and (gallery_loader is data_loader) and (Ng == Nq):
                    os_ = max(qs, gs)
                    oe_ = min(qe, ge)
                    if os_ < oe_:
                        q_rows = torch.arange(os_ - qs, oe_ - qs, device=device)
                        g_cols = torch.arange(os_ - gs, oe_ - gs, device=device)
                        sims[q_rows, g_cols] = -1e9

                vals, idx = torch.topk(sims, k=min(topk_eval, ge - gs), dim=1, largest=True)
                idx = idx + gs

                merged_vals = torch.cat([best_vals, vals], dim=1)
                merged_idx  = torch.cat([best_idx, idx], dim=1)
                new_vals, new_pos = torch.topk(merged_vals, k=topk_eval, dim=1, largest=True)
                new_idx = torch.gather(merged_idx, 1, new_pos)

                best_vals, best_idx = new_vals, new_idx

            topk_idx = best_idx
            topk_lab = y_g.to(device, non_blocking=True)[topk_idx]

        # ---- predict ----
        if k == 1:
            y_pred = topk_lab[:, 0]
        else:
            vote_lab = topk_lab[:, :k]
            y_pred = torch.empty((b,), dtype=torch.long, device=device)
            for i in range(b):
                vals, counts = vote_lab[i].unique(return_counts=True)
                y_pred[i] = vals[counts.argmax()]

        y_true = y_q[qs:qe].to(device, non_blocking=True)
        all_preds.append(y_pred.detach().cpu())
        all_truths.append(y_true.detach().cpu())
        correct_top5 += (topk_lab[:, :5] == y_true.unsqueeze(1)).any(dim=1).sum().item()

    all_preds = torch.cat(all_preds, dim=0) if all_preds else torch.empty(0, dtype=torch.long)
    all_truths = torch.cat(all_truths, dim=0) if all_truths else torch.empty(0, dtype=torch.long)

    # Keep the legacy key name `accuracy`, but make it represent balanced accuracy.
    acc = _balanced_accuracy_from_tensors(all_truths, all_preds)
    top5_acc = correct_top5 / max(Nq, 1)

    print(f"* [{split}] embed-NN accuracy {acc:.4f} top5 {top5_acc:.4f} (Ng={Ng} Nq={Nq} k={k})")

    if log_writer is not None and epoch is not None:
        log_writer.log(
            {f"{split}/loss": 0.0, f"{split}/accuracy": float(acc), f"{split}/top5_accuracy": float(top5_acc)},
            step=epoch,
        )

    return {
        "loss": 0.0,
        "metrics": {
            "accuracy": float(acc),
            "top5_accuracy": float(top5_acc),
            "k": int(k),
            "normalize": bool(normalize),
            "exclude_self": bool(exclude_self),
            "Ng": int(Ng),
            "Nq": int(Nq),
        },
    }
