import os
import sys
import torch
import traceback

# ------------------------------------------------------------
# Make sure we can import from project root
# (run: python test.py  at Benchmarks/ )
# ------------------------------------------------------------
PROJECT_ROOT = os.getcwd()
sys.path.append(PROJECT_ROOT)

from models.factory import get_model


class MockConfig:
    """模拟真实的 Config 对象，用于测试"""
    def __init__(self, model_name, tuning_mode="linear_probing"):
        # ----- LaBraM specific (depends on your modeling_finetune.py impl) -----
        self.LABRAM_NUM_CH = 62
        self.LABRAM_NUM_T = 2

        # ----- core -----
        self.MODEL_NAME = model_name
        self.NUM_CLASSES = 4
        self.TUNING_MODE = tuning_mode

        # ----- common -----
        self.USE_ZSCORE = True
        self.PATCH_SIZE = 200

        # ----- REAL checkpoint paths -----
        ckpt_dir = os.path.join(PROJECT_ROOT, "checkpoints")
        self.PATH_BIOT = os.path.join(ckpt_dir, "EEG-six-datasets-18-channels.ckpt")
        self.PATH_LABRAM = os.path.join(ckpt_dir, "labram-base.pth")
        self.PATH_CBRAMOD = os.path.join(ckpt_dir, "pretrained_weights.pth")

        # ----- BIOT specific -----
        self.BIOT_EMB = 256
        self.BIOT_HOP = 100
        self.N_CHANNEL_OFFSET = 0
        self.BIOT_N_CHANNELS = 18
        self.CHANNEL_SELECT = None  # test 用 dummy 18ch，先不做 select

        # ----- head -----
        self.HEAD_TYPE = "mlp"
        self.HEAD_HIDDEN = 128
        self.HEAD_DROPOUT = 0.1
        self.HEAD_ACT = "gelu"


def make_dummy_input(model_name: str, patch_size: int = 200):
    """
    Return x with correct channel count per model.
    Note: This is ONLY a sanity forward test.
          BIOT montage correctness must be done in dataset/transform.
    """
    B, T = 2, 2 * patch_size  # 400, divisible by 200

    if model_name == "biot":
        C = 18
    else:
        C = 62

    return torch.randn(B, C, T)


def check_freeze_linear_probing(model):
    print("🔍 Checking Parameter Freezing (Linear Probing Mode)...")
    frozen_backbone = True
    trainable_head = False

    for name, p in model.feature_extractor.named_parameters():
        if p.requires_grad:
            print(f"   ⚠️ Backbone param {name} is NOT frozen!")
            frozen_backbone = False
            break

    for name, p in model.probe_head.named_parameters():
        if p.requires_grad:
            trainable_head = True
            break

    if frozen_backbone and trainable_head:
        print("✅ Freeze strategy works: Backbone frozen, Head trainable.")
    else:
        print(f"❌ Freeze strategy failed. Backbone frozen: {frozen_backbone}, Head trainable: {trainable_head}")


def _print_ckpt_paths(cfg: MockConfig):
    print("📦 Using checkpoints:")
    print(f"   LaBraM : {cfg.PATH_LABRAM}  (exists={os.path.exists(cfg.PATH_LABRAM)})")
    print(f"   BIOT   : {cfg.PATH_BIOT}    (exists={os.path.exists(cfg.PATH_BIOT)})")
    print(f"   CBraMod: {cfg.PATH_CBRAMOD} (exists={os.path.exists(cfg.PATH_CBRAMOD)})")


def test_model_flow(model_name: str):
    print(f"\n{'='*20} Testing {model_name.upper()} {'='*20}")

    cfg = MockConfig(model_name, tuning_mode="linear_probing")
    _print_ckpt_paths(cfg)

    # build
    try:
        model = get_model(cfg)
        model.eval()
        print("✅ Model built successfully.")
    except Exception as e:
        print(f"❌ Failed to build model: {e}")
        traceback.print_exc()
        return

    # input
    x = make_dummy_input(model_name, patch_size=cfg.PATCH_SIZE)

    # forward
    try:
        y = model(x)
        print("✅ Forward pass successful.")
        print(f"   Input shape: {tuple(x.shape)}")
        print(f"   Output shape: {tuple(y.shape)}")
        assert y.shape == (x.shape[0], cfg.NUM_CLASSES), \
            f"Output shape mismatch! Expected {(x.shape[0], cfg.NUM_CLASSES)}, got {tuple(y.shape)}"
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        traceback.print_exc()
        if model_name == "biot":
            print("💡 Hint: BIOT 真实使用必须喂 BIOT-18 montage（18个双极导联、固定顺序）。")
        if model_name == "labram":
            print("💡 Hint: LaBraM 若仍报 pos_embed/token mismatch，说明 factory 里 num_ch/num_t 或 input_chans 没对齐。")
        return

    # freeze check
    check_freeze_linear_probing(model)


def main():
    for m in ["labram", "biot", "cbramod"]:
        test_model_flow(m)


if __name__ == "__main__":
    main()
