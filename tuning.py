# tuning.py
import torch


def _set_requires_grad(module: torch.nn.Module, flag: bool):
    """
    Safely set requires_grad:
    - Only float/complex tensors can require gradients.
    - int/bool tensors are always frozen to avoid RuntimeError.
    """
    for p in module.parameters():
        if p.dtype.is_floating_point or p.is_complex():
            p.requires_grad = flag
        else:
            p.requires_grad = False


def _get_fe_head(model: torch.nn.Module):
    """
    Support both naming styles:
      - wrappers.DownstreamModel: feature_extractor / probe_head
      - legacy: extractor / head
    """
    if hasattr(model, "feature_extractor") and hasattr(model, "probe_head"):
        return model.feature_extractor, model.probe_head
    if hasattr(model, "extractor") and hasattr(model, "head"):
        return model.extractor, model.head
    return None, None


def setup_training_mode(model: torch.nn.Module, cfg):
    """
    Control which params are trainable according to cfg.train.tuning_mode.

    Accept tuning_mode aliases:
      - linear_probing / linear_probe / linear / probe
      - full_finetune / finetune / full
      - zero_shot / zeroshot
    """
    mode = str(getattr(cfg.train, "tuning_mode", "")).lower()

    # ---- accept aliases ----
    if mode in {"linear_probe", "linear_probing", "linear", "probe"}:
        mode = "linear_probing"
    elif mode in {"full_finetune", "finetune", "full"}:
        mode = "full_finetune"
    elif mode in {"zero_shot", "zeroshot"}:
        mode = "zero_shot"

    if mode not in {"linear_probing", "full_finetune", "zero_shot"}:
        raise ValueError(
            f"Unknown train.tuning_mode={mode}. "
            "Must be one of: linear_probing | full_finetune | zero_shot "
            "(aliases: linear_probe/finetune/full/linear/probe)"
        )

    fe, head = _get_fe_head(model)

    if mode == "full_finetune":
        _set_requires_grad(model, True)
        print("[Tuning] full_finetune: backbone + head trainable")
        return model

    if mode == "zero_shot":
        _set_requires_grad(model, False)
        print("[Tuning] zero_shot: backbone + head frozen (eval only)")
        return model

    # ---- linear_probing ----
    if fe is not None and head is not None:
        _set_requires_grad(model, True)
        _set_requires_grad(fe, False)
        _set_requires_grad(head, True)
        enable_probe_adapters = getattr(fe, "enable_linear_probe_trainables", None)
        if callable(enable_probe_adapters):
            enable_probe_adapters()
        print("[Tuning] linear_probing: backbone frozen, head trainable")
        return model

    # ---- Fallback path: name-based ----
    _set_requires_grad(model, False)
    trainable = 0
    total = 0
    for n, p in model.named_parameters():
        total += 1
        if ("head" in n) or ("probe_head" in n):
            if p.dtype.is_floating_point or p.is_complex():
                p.requires_grad = True
                trainable += 1
            else:
                p.requires_grad = False

    print(f"[Tuning] linear_probing(fallback): trainable={trainable}/{total} (only '*head*'/'*probe_head*')")
    
    return model
