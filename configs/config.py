# configs/config.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, List, Optional

from omegaconf import OmegaConf, DictConfig


def _require(cfg: DictConfig, key: str) -> None:
    if OmegaConf.select(cfg, key) is None:
        raise ValueError(f"[Config] Missing required key: `{key}`")


def load_config(yaml_path: str) -> DictConfig:
    """
    Load yaml via OmegaConf:
      - supports ${...} interpolation out of box
      - returns DictConfig with dot access
    """
    cfg = OmegaConf.load(yaml_path)

    # ---- required top-level groups ----
    for k in ["paths", "dataset", "model", "train"]:
        _require(cfg, k)

    # ---- required keys used by your code ----
    _require(cfg, "paths.data_index_dir")
    _require(cfg, "paths.output_dir")

    #_require(cfg, "dataset.num_classes")
    #_require(cfg, "dataset.chn_names")

    _require(cfg, "model.names")
    _require(cfg, "model.patch_size")
    _require(cfg, "model.head.type")

    _require(cfg, "train.tuning_mode")
    _require(cfg, "train.batch_size")
    _require(cfg, "train.epochs")
    _require(cfg, "train.num_workers")
    _require(cfg, "train.lr")
    _require(cfg, "train.weight_decay")

    # force resolve interpolations early, so missing ${} throws now
    OmegaConf.resolve(cfg)
    return cfg
