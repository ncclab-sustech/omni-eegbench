# 在 5.0_version 中加入新模型指南

这份文档说明如何把一个新的 EEG 模型接入当前 `5.0_version` benchmark。当前框架的核心约定是：

```text
输入 batch: x [B, C, T]
feature_extractor: x -> feat [B, D_flat]
ProbeHead: feat -> logits [B, num_classes]
最终模型: DownstreamModel(feature_extractor, probe_head)
```

只要新模型能通过 `models.factory.get_model(cfg)` 构造成这个结构，就可以被 `main.py`、`tuning.py`、`engine.py` 和 leaderboard 流程统一训练与评估。

## 需要改哪些文件

通常需要改 3 类文件：

| 文件 | 必须 | 作用 |
|---|---:|---|
| `models/<new_model>.py` | 是 | 放新模型 backbone 的实现，或封装官方代码 |
| `models/wrappers.py` | 通常是 | 把任意数据集的 `[B,C,T]` 适配成该模型需要的输入，并输出二维 `[B,D_flat]` 特征 |
| `models/factory.py` | 是 | 注册模型名、加载 checkpoint、构造 wrapper、接统一 head |
| `configs/*.yaml` | 是 | 加 `model.names`、checkpoint 路径、模型专属超参 |
| `main.py` | 可选 | 如果需要在 `CHECKPOINT_URLS` 中补下载来源 |

推荐最小接入方式：

1. 在 `models/<new_model>.py` 实现或导入 backbone。
2. 在 `models/wrappers.py` 写一个 `NewModelFeatureExtractor`。
3. 在 `models/factory.py` 中增加 `elif model_name == "<new_model>":` 分支。
4. 在 YAML 的 `paths` 和 `model` 下增加配置，并把模型名加入 `model.names`。
5. 用一个小配置先做 1 个 epoch 冒烟测试。

## 第 1 步：实现 backbone

如果新模型是自己写的，建议文件结构：

```python
# models/my_model.py
import torch
import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self, ...):
        super().__init__()
        ...

    def forward(self, x: torch.Tensor):
        ...
```

如果新模型来自官方仓库，优先保留官方模块结构，但需要注意：

- 不要让官方分类头直接参与 benchmark，分类头由 `ProbeHead` 统一创建。
- 如果 checkpoint 里包含预训练分类头，加载时应过滤掉 `head`、`classifier`、`fc` 等不匹配参数。
- backbone 的输出可以是 token、channel 或 window 级特征，wrapper 负责 pool 或 flatten 成二维 `[B,D_flat]`。

## 第 2 步：写 FeatureExtractor

新增 wrapper 的职责是把任意数据集输入统一适配到新模型：

- 输入必须是 `x [B, C, T]`。
- 中间可以保留 token/channel/window 维度，例如 `[B, N, D]`、`[B, C, D]`、`[B, C, W, D]`；监督训练路径下，交给 `ProbeHead` 的最终输出建议规约或 flatten 成二维 `feat [B, D_flat]`。
- 必须设置 `self.embed_dim = D_flat`，因为 factory 会用它创建 `ProbeHead`。
- 建议设置 `self.benchmark_metadata`，结果会写入 `summary.json`。
- 如果模型需要固定采样率、固定长度、固定通道顺序，应在 wrapper 内处理。

最小模板：

```python
# models/wrappers.py
class MyModelFeatureExtractor(nn.Module):
    """
    Input:  x [B, C, T]
    Output: feat [B, D_flat]
    """
    def __init__(
        self,
        model: nn.Module,
        input_freq: int,
        target_freq: int = 200,
        norm: Optional[nn.Module] = None,
        interpolation_mode: str = "linear",
        interpolation_align_corners: bool = False,
    ):
        super().__init__()
        self.model = model
        self.input_freq = int(input_freq)
        self.target_freq = int(target_freq)
        self.norm = norm if norm is not None else IdentityNorm()
        self.interpolation_mode = interpolation_mode
        self.interpolation_align_corners = bool(interpolation_align_corners)

        self.embed_dim = 512
        self.benchmark_metadata = {
            "implementation": "adapter",
            "variant": "my_model_adapter",
            "target_sampling_rate": self.target_freq,
            "channel_policy": "dataset-native channels",
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected x [B,C,T], got {tuple(x.shape)}")

        x = self.norm(x)
        if self.input_freq != self.target_freq:
            x = resample_along_time(
                x,
                self.input_freq,
                self.target_freq,
                mode=self.interpolation_mode,
                align_corners=self.interpolation_align_corners,
            )

        out = self.model(x)

        # 根据模型实际输出做 pooling / flatten，目标是 [B,D_flat]
        if out.ndim == 3:
            out = out.mean(dim=1)
        elif out.ndim > 3:
            out = out.flatten(1)
        return out
```

常用工具已经在 `models/wrappers.py` 里存在：

| 工具 | 用途 |
|---|---|
| `IdentityNorm` | 不做归一化 |
| `PerTrialZScore` | 每个 trial/channel 沿时间维做 z-score |
| `resample_along_time` | `[B,C,T]` 时间维重采样 |
| `LengthAdapter` | 固定长度 crop/pad |
| `build_coord_projection_matrix` | 按电极坐标做通道插值 |

## 第 3 步：在 factory 中注册

在 `models/factory.py` 顶部导入新模型和 wrapper：

```python
from models.my_model import MyModel
from models.wrappers import MyModelFeatureExtractor
```

如果需要模型别名或 variant，在 `MODEL_ALIAS_TO_BASE` 和 `_requested_model_and_variant` 的 `defaults` 中加：

```python
MODEL_ALIAS_TO_BASE = {
    ...
    "my_model_adapter": "my_model",
}

defaults = {
    ...
    "my_model": "adapter",
}
```

然后在 `get_model(cfg)` 里新增分支。注意尽量复用已经解析好的字段：

```python
elif model_name == "my_model":
    ckpt_path = str(_require(cfg, "paths.my_model"))

    raw = MyModel(...)
    load_checkpoint_any(raw, ckpt_path, model_type="my_model", strict=False)

    norm = PerTrialZScore() if use_zscore else IdentityNorm()
    extractor = MyModelFeatureExtractor(
        model=raw,
        input_freq=input_sampling_rate,
        target_freq=int(OmegaConf.select(cfg, "model.my_model.target_freq") or 200),
        norm=norm,
        interpolation_mode=interpolation_mode,
        interpolation_align_corners=interpolation_align_corners,
    )
```

factory 末尾已经统一完成：

```python
head = ProbeHead(in_dim=int(extractor.embed_dim), out_dim=num_classes, ...)
model = DownstreamModel(extractor, head)
```

因此新分支只需要正确构造 `extractor`。

## 第 4 步：配置 YAML

在实验配置中加入 checkpoint 路径和模型超参。例如：

```yaml
paths:
  ckpt_dir: "./checkpoints"
  my_model: "${paths.ckpt_dir}/my_model_pretrained.pth"

model:
  names: [my_model]
  patch_size: 200

  my_model:
    variant: "adapter"
    target_freq: 200
    backbone_lr: 3e-5
    head_lr: 5e-4
    weight_decay: 0.05
    no_decay_norm_bias: true
```

说明：

- `model.names` 是主循环读取的模型列表；`main.py` 会逐个设置 `cfg.model.name`。
- `paths.<model_name>` 通常在 factory 里用 `_require(cfg, "paths.<model_name>")` 读取。
- `model.<model_name>.backbone_lr` 和 `head_lr` 会被 `DownstreamModel.get_parameter_groups()` 自动读取。
- 如果是大规模预训练模型，`full_finetune` 下建议 backbone lr 小一些，例如 `1e-5` 到 `5e-5`。

## 第 5 步：checkpoint 加载策略

优先使用 `load_checkpoint_any()`，它已经支持：

- `.pth` / `.pt` / `.bin` / `.safetensors`
- 常见外层 key：`state_dict`、`model`、`module`、`net`、`params`、`weights`
- 常见 prefix：`module.`、`model.`、`state_dict.`、`net.`、`backbone.`
- 按 shape 过滤不匹配参数

如果新模型 checkpoint 有特殊命名，建议在 factory 分支里先处理 state dict，再调用 `load_state_dict`。例如：

```python
sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
sd = unwrap_state_dict(sd)
sd = strip_prefixes(sd, prefixes=("module.", "encoder."))
sd = drop_if_contains(sd, ("classifier", "head"))
sd = filter_by_shape(sd, raw, verbose=True)
raw.load_state_dict(sd, strict=False)
```

不要强行 `strict=True`，除非已经确认 checkpoint 与模型结构完全一致。

## 第 6 步：通道、采样率和长度适配

新模型最常见的问题都在这里。

通道策略可选：

| 情况 | 推荐做法 |
|---|---|
| 模型支持任意通道数 | 直接使用数据集原始 `C` |
| 模型要求固定通道顺序 | 在 wrapper 内根据 `chn_names` 重排，缺失通道可 zero-fill 或报错 |
| 模型要求标准 montage | 用 `build_coord_projection_matrix` 或参考 `NeuroGPTFeatureExtractor` |
| 模型要求 bipolar montage | 参考 `BIOTFeatureExtractor` / `BENDRFeatureExtractor` |

采样率策略：

- 数据集采样率来自 `load_dataset_metadata(cfg)["processing"]["target_sampling_rate"]`。
- 如果模型预训练采样率固定，在 wrapper 内用 `resample_along_time`。
- `interpolation.mode` 和 `interpolation.align_corners` 从 YAML 统一读取。

长度策略：

- 如果模型要求固定 `T`，使用 `LengthAdapter`。
- 如果模型要求 `T % patch_size == 0`，要在 wrapper 内 crop/pad 到合法长度。
- 如果模型以 patch/token/channel/window 为单位输出，最终要在 wrapper 内 pool 或 flatten 成 `[B,D_flat]`。

## 第 7 步：训练模式兼容

当前 `tuning.py` 支持：

| `train.tuning_mode` | 行为 |
|---|---|
| `linear_probing` | freeze feature extractor，只训练 `probe_head` |
| `full_finetune` | 训练 backbone + head |
| `zero_shot` | 全部冻结，只做 embedding nearest-neighbor 评估 |

如果你的 wrapper 里有少量 adapter 需要在线性探针时也训练，实现：

```python
def enable_linear_probe_trainables(self):
    self.channel_adapter.requires_grad_(True)
    self.channel_adapter.train(True)
```

如果还需要自定义 optimizer parameter groups，实现：

```python
def get_parameter_groups(self, lr: float, weight_decay: float, *, probe_head=None, cfg=None):
    ...
    return groups
```

可参考 `BENDRLearnedChannelAdapterFeatureExtractor`。

## 第 8 步：冒烟测试

建议先建一个临时小配置，例如把：

```yaml
model:
  names: [my_model]

train:
  seed_list: [42]
  epochs: 1
  batch_size: 2
  num_workers: 0
  use_wandb: false
```

然后运行：

```bash
python main.py configs/your_smoke_test.yaml
```

重点看日志：

- `[factory] Missing required config key`：YAML 缺字段。
- `Unknown model.name`：factory 没注册，或 `model.names` 名字不一致。
- `Cannot infer ... embed_dim`：wrapper 没设置 `self.embed_dim`。
- `Expected x [B,C,T]`：数据维度或 wrapper 输入假设不对。
- `shape mismatch` / `loaded ratio is low`：checkpoint 和模型结构不匹配。
- CUDA OOM：先调小 `batch_size`，必要时降低窗口长度或用 `linear_probing`。

## 推荐接入 checklist

- [ ] `models/<new_model>.py` 可以单独 import。
- [ ] wrapper 输入 `[B,C,T]`，最终输出二维 `[B,D_flat]`。
- [ ] wrapper 设置了 `self.embed_dim`。
- [ ] wrapper 处理了新模型需要的通道顺序、采样率、长度和归一化。
- [ ] `models/factory.py` 导入了 backbone 和 wrapper。
- [ ] `get_model(cfg)` 有 `elif model_name == "<new_model>"` 分支。
- [ ] YAML 有 `paths.<new_model>` 和 `model.<new_model>`。
- [ ] `model.names` 中的名字与 factory 分支完全一致，全部小写。
- [ ] 预训练分类头没有被错误加载为 benchmark head。
- [ ] 1 epoch 冒烟测试能跑完，并生成 `summary.json`。

## 最小代码改动示例

下面是一个完整骨架，实际接入时把 `MyModel(...)` 和 forward 细节替换成真实模型逻辑。

```python
# models/factory.py
from models.my_model import MyModel
from models.wrappers import MyModelFeatureExtractor

...

elif model_name == "my_model":
    ckpt_path = str(_require(cfg, "paths.my_model"))
    raw = MyModel(...)
    load_checkpoint_any(raw, ckpt_path, model_type="my_model", strict=False)

    extractor = MyModelFeatureExtractor(
        model=raw,
        input_freq=input_sampling_rate,
        target_freq=int(OmegaConf.select(cfg, "model.my_model.target_freq") or 200),
        norm=PerTrialZScore() if use_zscore else IdentityNorm(),
        interpolation_mode=interpolation_mode,
        interpolation_align_corners=interpolation_align_corners,
    )
```

```python
# models/wrappers.py
class MyModelFeatureExtractor(nn.Module):
    def __init__(self, model, input_freq, target_freq=200, norm=None, **kwargs):
        super().__init__()
        self.model = model
        self.input_freq = int(input_freq)
        self.target_freq = int(target_freq)
        self.norm = norm if norm is not None else IdentityNorm()
        self.embed_dim = 512

    def forward(self, x):
        x = self.norm(x)
        out = self.model(x)
        if out.ndim == 3:
            return out.mean(dim=1)
        if out.ndim > 3:
            return out.flatten(1)
        return out
```

```yaml
# configs/xxx.yaml
paths:
  my_model: "${paths.ckpt_dir}/my_model.pth"

model:
  names: [my_model]
  my_model:
    variant: "adapter"
    target_freq: 200
    backbone_lr: 3e-5
    head_lr: 5e-4
    weight_decay: 0.05
```

## 接入质量建议

新模型正式加入 benchmark 前，建议至少检查：

- 同一数据集上 `linear_probing` 能稳定跑完。
- `full_finetune` 下 optimizer groups 的 lr 符合预期。
- `summary.json` 中 `model_benchmark_metadata` 能说明 adapter、target channels、target sampling rate 和 checkpoint。
- 缺失通道不要静默失败：要么打印 warning，要么在 strict 模式下报错。
- checkpoint 加载日志中的 loaded ratio 不应异常低；如果低于 30%，要人工确认 key 映射是否正确。
