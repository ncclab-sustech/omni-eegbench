import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
from typing import Optional

_DATA_CACHE = {}

class EEGIndexDataset(Dataset):
    def __init__(
        self,
        index_file,
        transform=None,
        split: str = "train",          
        zero_mask_ratio: float = 0.0,  
        seed: int = 42, 
        zero_mask_channels=None,
        # lzs修改
        remap_labels: bool = True,
        label2idx: dict | None = None,            
    ):
        with open(index_file, "r") as f:
            self.samples = [line.strip().split(",") for line in f.readlines() if line.strip()]

        self.transform = transform
        self.split = split
        self.remap_labels = remap_labels
        if self.remap_labels:
            # 从 samples 里取出所有原始 label
            raw_labels = [int(s[2]) for s in self.samples]
            if label2idx is None:
                uniq = sorted(set(raw_labels))
                self.label2idx = {lab: i for i, lab in enumerate(uniq)}
            else:
                self.label2idx = dict(label2idx)

            self.idx2label = {v: k for k, v in self.label2idx.items()}
            unknown = sorted(set(raw_labels) - set(self.label2idx.keys()))
            if len(unknown) > 0:
                raise ValueError(
                    f"[EEGIndexDataset] Found labels not in label2idx: {unknown}. "
                    f"Please build label2idx from train and pass it to this split."
                )

            self.num_classes = len(self.label2idx)
        cache_key = (
            str(index_file),
            bool(self.remap_labels),
            tuple(sorted(self.label2idx.items())) if self.remap_labels else None,
        )
        if cache_key in _DATA_CACHE:
            self.data = _DATA_CACHE[cache_key]
        else:
            h5_list = np.unique([i[0] for i in self.samples]).tolist()
            h5_list.sort()
            h5_cache = {h5_path: h5py.File(h5_path, "r") for h5_path in h5_list}
            try:
                self.data = [
                    (
                        torch.from_numpy(h5_cache[h5][p][()]).float(),
                        torch.tensor(self.label2idx[int(label)]).long(),
                    )
                    for h5, p, label in self.samples
                ]
            finally:
                for h5_file in h5_cache.values():
                    h5_file.close()
            _DATA_CACHE[cache_key] = self.data
        self.zero_mask_ratio = float(zero_mask_ratio)
        self.zero_mask_channels = None if zero_mask_channels is None else [int(x) for x in zero_mask_channels]

        self._rng = np.random.RandomState(seed)
        
    def __len__(self):
        return len(self.data)

    def _maybe_zero_mask(self, x: torch.Tensor) -> torch.Tensor:
        """
        x expected: [C, T]
        in train only: randomly choose ratio*C channels and set to 0.
        """
        if self.zero_mask_channels is not None:
            if not self.zero_mask_channels:
                return x
            x = x.clone()
            valid = [idx for idx in self.zero_mask_channels if 0 <= idx < x.shape[0]]
            if valid:
                x[valid, :] = 0.0
            return x

        if self.split != "train":
            return x
        r = self.zero_mask_ratio
        if r <= 0:
            return x

        # x: [C,T]
        C = x.shape[0]
        k = int(C * r)
        if k <= 0:
            k = 1  

        ch_idx = self._rng.choice(C, size=k, replace=False)
        x = x.clone()  
        x[ch_idx, :] = 0.0
        return x

    def __getitem__(self, idx):
        x,y = self.data[idx]

        # if x.ndim == 2:
        #     if x.shape[0] > x.shape[1]:
        #         x = x.transpose(0, 1)  

        x = self._maybe_zero_mask(x)

        if self.transform:
            x = self.transform(x)

        return x, y
