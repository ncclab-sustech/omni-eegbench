# 5.0 Version Architecture

## 入口

运行方式：

```bash
python main.py configs/linear_prob_zarr_reve.yaml
```

主入口文件：

| 文件 | 作用 |
|---|---|
| `main.py` | 实验主控：读配置、构造 dataloader、循环模型、训练、验证、测试、写结果 |
| `configs/linear_prob_zarr_reve.yaml` | 当前实验配置：数据路径、模型列表、checkpoint 路径、训练参数 |
| `configs/config.py` | 配置加载与 OmegaConf 解析 |

当前配置的关键字段：

| 配置项 | 含义 |
|---|---|
| `dataset.backend: zarr` | 使用 Zarr 数据后端 |
| `paths.zarr_dataset_name` | Zarr 数据集根目录 |
| `model.names` | 依次运行的模型列表 |
| `train.tuning_mode` | `linear_probing` / `full_finetune` / `zero_shot` |
| `train.ratio_shot` | Zarr train split 的按类别抽样比例；`null` 表示全量 |
| `dataset.downsample` | 当前只对 H5 后端生效；Zarr 后端会跳过 H5 index/downsample 生成 |

## 整体流程

```text
main.py
  -> load_config
  -> _build_loaders
  -> for ratio_shot
  -> for model.name in model.names
  -> run_one_model
      -> models.factory.get_model
      -> tuning.setup_training_mode
      -> engine.train_one_epoch
      -> engine.evaluate(val)
      -> save best checkpoint
      -> engine.evaluate(test)
      -> write summary.json / train_log.jsonl
  -> leaderboard_utils.rebuild_leaderboards_from_summaries
```

## 数据读入

### Zarr 后端

使用文件：

| 文件 | 作用 |
|---|---|
| `data/zarr_dataset.py` | `ZarrSplitDataset`，读取 `sample_index.parquet` 和 `.zarr/signals` |
| `data/metadata.py` | `load_dataset_metadata`，从 Zarr 元数据推断类别数、通道名、采样率、窗口长度 |
| `main.py::_build_loaders` | 构造 train/val/test 三个 DataLoader |

Zarr 数据要求：

```text
zarr_dataset_dir/
  sample_index.parquet
  *.zarr/
    signals
```

`ZarrSplitDataset` 输出：

```text
x: torch.Tensor [C, T]
y: torch.LongTensor scalar
```

split 逻辑：

| 情况 | 处理 |
|---|---|
| `sample_index.parquet` 有 `split` 列，且包含 `train/val/test` | 直接使用显式 split |
| 否则 | 按 `subject_id` 做 subject-level split |
| `train.ratio_shot != null` | 仅对 train split 按 label 分层抽样 |

Zarr cache：

| 环境变量 | 默认 | 作用 |
|---|---:|---|
| `BENCHMARK_EEG_ZARR_RAW_CACHE` | `true` | 是否把选中样本缓存到内存 |
| `BENCHMARK_EEG_ZARR_RAW_CACHE_GB` | `8.0` | 最大缓存容量 |

### H5 后端

使用文件：

| 文件 | 作用 |
|---|---|
| `data/dataset.py` | `EEGIndexDataset`，按 index 文件读取 H5 样本 |
| `data/make_indices_fixed.py` | 生成普通 train/val/test index |
| `data/make_indices_downsample_fixed.py` | 生成 downsample index |
| `main.py::_select_train_index_file` | 根据 `ratio_shot` 选择 train index |

H5 后端会使用：

```text
indices/.../train_idx.txt
indices/.../val_idx.txt
indices/.../test_idx.txt
```

Zarr 后端不会走这套 index 生成。

## Metadata

`models.factory.get_model` 和 `main.run_one_model` 都通过：

```python
load_dataset_metadata(cfg)
```

读取统一数据描述。

主要字段：

| 字段 | 用途 |
|---|---|
| `dataset.num_labels` | 分类头输出维度 |
| `dataset.channels` | 模型通道适配 |
| `dataset.downstream_task` | 分类/回归判断 |
| `processing.target_sampling_rate` | 输入采样率 |
| `processing.window_sec` | 输入窗口长度 |

## DataLoader

位置：`main.py::_build_loaders`

配置：

| 配置项 | 作用 |
|---|---|
| `train.batch_size` | batch size |
| `train.num_workers` | DataLoader worker 数 |
| `train.persistent_workers` | worker 是否常驻 |
| `train.prefetch_factor` | worker 预取 batch 数 |
| CUDA device | 自动启用 `pin_memory=True` |

## 模型构建

统一入口：

```python
models.factory.get_model(cfg)
```

统一输出结构：

```text
DownstreamModel
  feature_extractor: EEG [B,C,T] -> feature [B,D]
  probe_head: feature [B,D] -> logits [B,num_classes]
```

相关文件：

| 文件 | 作用 |
|---|---|
| `models/factory.py` | 根据 `model.name` 构造 backbone、加载 checkpoint、选择 wrapper |
| `models/wrappers.py` | 各模型输入适配、重采样、通道映射、特征池化 |
| `models/*.py` | 各 backbone 的具体实现 |

### 当前模型构建表

| `model.name` | Backbone | Wrapper | 输入适配/特征逻辑 |
|---|---|---|---|
| `eegnet` | `EEGNet` | backbone 自身 | 使用数据原始通道数；`PerTrialZScore` |
| `eegconformer` | `Conformer` | backbone 自身 | 使用数据原始通道数；`PerTrialZScore` |
| `labram` | `create_model("labram_base_patch200_200")` | `LaBraMFeatureExtractor` | 通道名映射到 LaBraM index；`FixedScaleTo01mV`；按 patch 输入 |
| `biot` | `BIOTClassifier` | `BIOTFeatureExtractor` | 构造 BIOT-18 bipolar montage；默认 `zero_fill_adapter` |
| `cbramod` | `CBraMod` | `CBraModFeatureExtractor` | 重采样到 200 Hz；reshape 为 `[B,C,P,200]`；mean pool |
| `reve` | HF `AutoModel` | `REVEFeatureExtractor` | 可用官方 position bank 或本地坐标；重采样到 200 Hz |
| `brainomni` | `BrainOmni` | `BrainOmniFeatureExtractor` | 通道名映射到坐标和 sensor type；重采样到 256 Hz；`model.encode` 后 flatten |
| `femba` | `FEMBA` | `FembaFeatureExtractor` | 坐标插值到 TUEG bipolar pair；当前为 `femba_adapter` |
| `neurogpt` | `EEGConformer` encoder | `NeuroGPTFeatureExtractor` | 插值到 22 通道；重采样到 250 Hz；chunk 后 encoder mean pool；当前不含 GPT 主体 |
| `neurolm` | `NeuroLM` | `NeuroLMFeatureExtractor` | 当前配置 `full_gpt`：按 200 点 patch 展平为 token 序列，经过 tokenizer 和 GPT2，mean pool |
| `eegmamba` | `EEGMamba` | `EEGMambaFeatureExtractor` | 适配到 canonical 19 通道；crop/pad 到目标长度；当前为 `19ch_adapter` |
| `bendr` | `BENDRClassification` | `BENDRFeatureExtractor` | 适配到 BENDR 19+1 输入策略；重采样到 256 Hz；当前为 `19plus1_adapter` |
| `brant` | `Brant` | `BrantPretrainFeatureExtractor` | 根据 checkpoint shape 推断时间参数；iEEG 参考模型，不在默认 scalp EEG 列表 |

### Variant / Adapter

`models.factory._requested_model_and_variant` 解析模型 variant。

当前默认/配置中的 adapter：

| 模型 | 当前 variant | 含义 |
|---|---|---|
| `biot` | `zero_fill_adapter` | 缺失 BIOT pair 时补 0 |
| `femba` | `adapter` | 坐标插值到 TUEG bipolar 目标 |
| `neurogpt` | `encoder_adapter` | 只用 EEG encoder，不用 GPT 主体 |
| `neurolm` | `full_gpt` | tokenizer 后进入 GPT2 主体 |
| `eegmamba` | `19ch_adapter` | 适配到 19 通道 |
| `bendr` | `19plus1_adapter` | 项目内 19+1 adapter |

每个 wrapper 可写入：

```python
model.benchmark_metadata
```

最终进入 `summary.json` 的 `model_benchmark_metadata` 字段。

## 训练模式

入口：

```python
tuning.setup_training_mode(model, cfg)
```

模式：

| `train.tuning_mode` | 行为 |
|---|---|
| `linear_probing` | 冻结 feature extractor，只训练 `probe_head` |
| `full_finetune` | feature extractor 和 head 全部训练 |
| `zero_shot` | 不训练；用 embedding nearest-neighbor 评估 |

底层 helper 在 `models/wrappers.py`：

| 函数 | 作用 |
|---|---|
| `set_linear_probe` | freeze feature extractor |
| `set_full_finetune` | 全部可训练 |
| `set_partial_finetune_last_n_transformer_blocks` | best-effort partial finetune |

## 训练与评估

使用文件：

| 文件 | 作用 |
|---|---|
| `engine.py::train_one_epoch` | 单 epoch 训练 |
| `engine.py::evaluate` | val/test 分类或回归评估 |
| `engine.py::evaluate_embed_nn` | zero-shot embedding NN 评估 |
| `utils.py::get_metrics` | accuracy、f1、AUC、回归指标等 |
| `utils.py::NativeScalerWithGradNormCount` | CUDA AMP scaler |

训练过程：

```text
for epoch:
  train_one_epoch
  if epoch hits eval_freq:
    evaluate(val)
    if val metric improves:
      save best checkpoint
load best checkpoint
evaluate(test)
```

保存指标：

| 任务 | save key |
|---|---|
| Classification | `accuracy` |
| Regression | `r2` |

BrainOmni 特例：

| 位置 | 行为 |
|---|---|
| `main.py::run_one_model` | 使用 `BrainOmni.get_parameters_groups`，backbone/head 分开学习率 |
| 配置 | `model.brainomni.backbone_lr`、`head_lr`、`weight_decay` |

## 输出结构

每个 ratio 下：

```text
result_5/{exp_name}/{dataset}_{split_mode}_{exp_name}/ratio_{tag}/
  {model}/
    train_log.jsonl
    summary.json
    best_{model}_{tuning_mode}.pth
  leaderboard_val.json
  leaderboard_val.txt
  leaderboard_test.json
  leaderboard_test.txt
```

`train_log.jsonl`：

| phase | 内容 |
|---|---|
| `train` | epoch、loss、lr |
| `val` | val loss、val metrics、best val |
| `test` | test loss、test metrics、best checkpoint path |

`summary.json`：

| 字段 | 内容 |
|---|---|
| `dataset` | 数据集名 |
| `model` | 模型名 |
| `model_benchmark_metadata` | wrapper 写入的模型/adapter 元信息 |
| `tuning_mode` | 训练模式 |
| `epochs` | epoch 数 |
| `save_key` | 用于选 best 的指标 |
| `best_val` | 最佳验证集指标 |
| `best_path` | 最佳 checkpoint |
| `test_metrics` | 测试集指标 |
| `test_loss` | 测试集 loss |
| `log_path` | 训练日志路径 |

leaderboard：

| 文件 | 来源 |
|---|---|
| `leaderboard_val.*` | 扫描各模型 `summary.json` 的 `best_val` |
| `leaderboard_test.*` | 扫描各模型 `summary.json` 的 `test_metrics.accuracy` |

生成位置：

```python
leaderboard_utils.rebuild_leaderboards_from_summaries(out_root)
```

## 当前注意点

| 项 | 状态 |
|---|---|
| Zarr downsample | `dataset.downsample` 不生效；Zarr 使用 `ratio_shot` |
| NeuroGPT | 当前只用 encoder adapter，不是完整 GPT 版本 |
| NeuroLM | 当前配置为 `full_gpt`，会经过 GPT2 主体 |
| Adapter 结果 | `model_benchmark_metadata` 会记录 adapter/variant 信息 |
| 网络盘性能 | Zarr raw-cache 可降低随机读开销；代码/import 仍受网络盘影响 |
