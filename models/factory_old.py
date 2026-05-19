# models/factory.py
import os
import torch
import json
import torch.nn as nn
from typing import Any, Dict, Iterable, Optional, Tuple, List
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import load_file
from timm.models import create_model as create_labram
import models.modeling_finetune
from models.cbramod import CBraMod
from models.biot import BIOTClassifier
from models.brainomni import BrainOmni
from models.FEMBA import FEMBA
from models.neurogpt import EEGConformer
import numpy as np
from models.eegmamba import EEGMamba
from models.neurolm import NeuroLM
from models.model_gpt import GPTConfig
from models.bendr import BENDRClassification
from models.brant import Brant
from models.EEGNet import EEGNet

from models.wrappers import (
    LaBraMFeatureExtractor,
    BIOTFeatureExtractor,
    CBraModFeatureExtractor,
    BrainOmniFeatureExtractor,
    FembaFeatureExtractor,
    NeuroGPTFeatureExtractor,
    # Add new model extractor here
    BrantPretrainFeatureExtractor,
    NeuroLMFeatureExtractor,
    EEGMambaFeatureExtractor,
    BENDRFeatureExtractor,
    ProbeHead,
    DownstreamModel,
    set_linear_probe,
    set_full_finetune,
    IdentityNorm,
    PerTrialZScore,
)

STANDARD_1020 = [
    'FP1', 'FPZ', 'FP2', 'AF9', 'AF7', 'AF5', 'AF3', 'AF1', 'AFZ', 'AF2', 'AF4', 'AF6', 'AF8', 'AF10',
    'F9', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'F10',
    'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10',
    'T9', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'T10',
    'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10',
    'P9', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'P10',
    'PO9', 'PO7', 'PO5', 'PO3', 'PO1', 'POZ', 'PO2', 'PO4', 'PO6', 'PO8', 'PO10',
    'O1', 'OZ', 'O2', 'O9', 'CB1', 'CB2',
    'IZ', 'O10', 'T3', 'T5', 'T4', 'T6', 'M1', 'M2', 'A1', 'A2',
    'CFC1', 'CFC2', 'CFC3', 'CFC4', 'CFC5', 'CFC6', 'CFC7', 'CFC8',
    'CCP1', 'CCP2', 'CCP3', 'CCP4', 'CCP5', 'CCP6', 'CCP7', 'CCP8',
    'T1', 'T2', 'FTT9h', 'TTP7h', 'TPP9h', 'FTT10h', 'TPP8h', 'TPP10h',
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP2-F8", "F8-T8", "T8-P8", "P8-O2", "FP1-F3", "F3-C3", "C3-P3", "P3-O1", "FP2-F4", "F4-C4", "C4-P4", "P4-O2"
]



def map_channels_to_indices(dataset_ch_names: List[str]) -> List[int]:
    upper_standard = [c.upper() for c in STANDARD_1020]
    input_chans = [0]
    missing = []
    ch_use = []
    for ch in dataset_ch_names:
        u = ch.upper()
        if u in upper_standard:
            input_chans.append(upper_standard.index(u) + 1)
            ch_use.append(True)
        else:
            missing.append(ch)
            ch_use.append(False)
    if missing:
        print(f"⚠️ Warning: {len(missing)} channels not found in STANDARD_1020: {missing[:5]}...")
    
    return input_chans,np.array(ch_use)


def _require(cfg: DictConfig, key: str) -> Any:
    v = OmegaConf.select(cfg, key)
    if v is None:
        raise ValueError(f"[factory] Missing required config key: `{key}`")
    return v


COORD_PATH = "data/standard_coords.pt"
COORD_CACHE = None
def _load_coords_if_needed():
    global COORD_CACHE
    if COORD_CACHE is None:
        if not os.path.exists(COORD_PATH):
            raise FileNotFoundError(f"Missing coords file: {COORD_PATH}. Please run generate_coords.py")
        print(f"Loading standard coords from {COORD_PATH}")
        data = torch.load(COORD_PATH, map_location="cpu")
        names = [n.upper() for n in data["ch_names"]]
        name_to_idx = {n: i for i, n in enumerate(names)}
        COORD_CACHE = {
            "pos": data["pos"],
            "sensor_type": data["sensor_type"],
            "map": name_to_idx
        }
    return COORD_CACHE

def map_channels_to_pos_and_type(chn_names):
    cache = _load_coords_if_needed()
    pos_list = []
    type_list = []
    
    for ch in chn_names:
        u = ch.upper()
        if u in cache["map"]:
            idx = cache["map"][u]
            pos_list.append(cache["pos"][idx])
            type_list.append(cache["sensor_type"][idx])
        else:
            print(f"⚠️ Warning: Channel {ch} not found in standard coords! Using zeros.")
            pos_list.append(torch.zeros(6))
            type_list.append(torch.tensor(0)) # Default to EEG
            
    return torch.stack(pos_list), torch.stack(type_list)


# ============================================================
# checkpoint parsing utilities
# ============================================================
def unwrap_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, nn.Module):
        return ckpt.state_dict()
    if not isinstance(ckpt, dict):
        raise ValueError("Checkpoint is not a dict or nn.Module; cannot unwrap state_dict.")
    for k in ["state_dict", "model", "module", "net", "params", "weights"]:
        v = ckpt.get(k, None)
        if isinstance(v, dict) and any(isinstance(x, torch.Tensor) for x in v.values()):
            return v
    if any(isinstance(x, torch.Tensor) for x in ckpt.values()):
        return ckpt
    raise ValueError(f"Cannot find state_dict in checkpoint keys: {list(ckpt.keys())[:20]} ...")


def strip_prefixes(sd: Dict[str, torch.Tensor], prefixes: Tuple[str, ...]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in sd.items():
        nk = k
        for p in prefixes:
            if nk.startswith(p):
                nk = nk[len(p):]
        out[nk] = v
    return out


def replace_prefix(sd: Dict[str, torch.Tensor], old: str, new: str = "") -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in sd.items():
        nk = k.replace(old, new, 1) if k.startswith(old) else k
        out[nk] = v
    return out


def drop_if_contains(sd: Dict[str, torch.Tensor], substrings: Iterable[str]) -> Dict[str, torch.Tensor]:
    subs = tuple(substrings)
    return {k: v for k, v in sd.items() if not any(s in k for s in subs)}


def filter_by_shape(sd: Dict[str, torch.Tensor], model: nn.Module, verbose: bool = True) -> Dict[str, torch.Tensor]:
    msd = model.state_dict()
    out: Dict[str, torch.Tensor] = {}
    skipped_shape = []
    skipped_missing = 0

    for k, v in sd.items():
        if k not in msd:
            skipped_missing += 1
            continue
        if tuple(v.shape) != tuple(msd[k].shape):
            skipped_shape.append((k, tuple(v.shape), tuple(msd[k].shape)))
            continue
        out[k] = v

    if verbose:
        print(f"  matched keys (name+shape): {len(out)}")
        print(f"  skipped (key not in model): {skipped_missing}")
        if skipped_shape:
            print(f"  skipped (shape mismatch): {len(skipped_shape)}")
            for k, a, b in skipped_shape[:6]:
                print(f"   - {k}: ckpt{a} vs model{b}")
            if len(skipped_shape) > 6:
                print("   ...")

    return out


def load_checkpoint_any(
    model: nn.Module,
    path: Optional[str],
    *,
    model_type: str,
    strict: bool = False,
    verbose: bool = True,
    min_loaded_ratio_warn: float = 0.30,
) -> None:
    if not path or str(path).lower() == "none":
        if verbose:
            print(f"[{model_type}] No checkpoint provided. Using random init.")
        return
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    if verbose:
        print(f"[{model_type}] Loading checkpoint: {path}")
    
    if str(path).endswith(".safetensors"):
        sd = load_file(path)
    else:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)

        # Special-case: DN3/BENDR checkpoints often split encoder/contextualizer.
        if (
            model_type == "bendr"
            and isinstance(ckpt, dict)
            and "encoder_state_dict" in ckpt
            and "contextualizer_state_dict" in ckpt
            and hasattr(model, "encoder")
            and hasattr(model, "contextualizer")
        ):
            enc_sd = ckpt["encoder_state_dict"]
            ctx_sd = ckpt["contextualizer_state_dict"]

            if not isinstance(enc_sd, dict) or not isinstance(ctx_sd, dict):
                raise ValueError("[bendr] encoder_state_dict/contextualizer_state_dict must be dicts")

            enc_sd = strip_prefixes(enc_sd, prefixes=("module.", "model.", "state_dict.", "net."))
            ctx_sd = strip_prefixes(ctx_sd, prefixes=("module.", "model.", "state_dict.", "net."))

            enc_sd = filter_by_shape(enc_sd, model.encoder, verbose=verbose)
            ctx_sd = filter_by_shape(ctx_sd, model.contextualizer, verbose=verbose)

            ratio_enc = len(enc_sd) / max(1, len(model.encoder.state_dict()))
            ratio_ctx = len(ctx_sd) / max(1, len(model.contextualizer.state_dict()))
            if verbose:
                print(f"   encoder loaded ratio: {ratio_enc:.1%}")
                print(f"   contextualizer loaded ratio: {ratio_ctx:.1%}")

            msg_enc = model.encoder.load_state_dict(enc_sd, strict=strict)
            msg_ctx = model.contextualizer.load_state_dict(ctx_sd, strict=strict)
            if verbose:
                print(f"[bendr] encoder load_state_dict(strict={strict}) => {msg_enc}")
                print(f"[bendr] contextualizer load_state_dict(strict={strict}) => {msg_ctx}")
            return

        sd = unwrap_state_dict(ckpt)

    sd = strip_prefixes(sd, prefixes=("module.", "model.", "state_dict.", "net.", "backbone."))
    sd = replace_prefix(sd, old="student.", new="")

    if model_type == "labram":
        sd = drop_if_contains(sd, ("head", "projection_head", "lm_head"))
        if "pos_embed" in sd and hasattr(model, "pos_embed"):
            if tuple(sd["pos_embed"].shape) != tuple(model.pos_embed.shape):
                if verbose:
                    print(f"  ⚠️ skip pos_embed mismatch: ckpt{tuple(sd['pos_embed'].shape)} vs model{tuple(model.pos_embed.shape)}")
                sd.pop("pos_embed", None)

    elif model_type == "biot":
        sd = drop_if_contains(sd, ("classifier", "fc", "head"))
        if not any(k.startswith("biot.") for k in sd.keys()):
            sd = {("biot." + k): v for k, v in sd.items()}

    sd = filter_by_shape(sd, model, verbose=verbose)

    ratio = len(sd) / max(1, len(model.state_dict()))
    if verbose:
        print(f"   loaded ratio: {ratio:.1%}")
    if ratio < min_loaded_ratio_warn and verbose:
        print("   WARNING: loaded ratio is low; checkpoint may not match model config.")

    msg = model.load_state_dict(sd, strict=strict)
    if verbose:
        print(f"[{model_type}] load_state_dict(strict={strict}) => {msg}")

def load_brant_pretrained(
        model: Brant,
        time_ckpt: str,
        channel_ckpt: Optional[str] = None,
        strict: bool = False,
    ):
        # ---- TimeEncoder ----
        time_sd = torch.load(time_ckpt, map_location="cpu")
        time_sd = unwrap_state_dict(time_sd)
        time_sd = strip_prefixes(
            time_sd,
            prefixes=("module.", "model.", "encoder.", "time.", "time_encoder.")
        )
        time_sd = filter_by_shape(time_sd, model.time, verbose=True)
        msg_t = model.time.load_state_dict(time_sd, strict=strict)
        print(f"[brant] time_encoder load => {msg_t}")

        # ---- ChannelEncoder ----
        if model.channel is not None and channel_ckpt is not None:
            ch_sd = torch.load(channel_ckpt, map_location="cpu")
            ch_sd = unwrap_state_dict(ch_sd)
            ch_sd = strip_prefixes(
                ch_sd,
                prefixes=("module.", "model.", "encoder.", "channel.", "channel_encoder.")
            )
            ch_sd = filter_by_shape(ch_sd, model.channel, verbose=True)
            msg_c = model.channel.load_state_dict(ch_sd, strict=strict)
            print(f"[brant] channel_encoder load => {msg_c}")

# ============================================================
# factory main (ALL lowercase)
# ============================================================
def get_model(cfg: DictConfig) -> nn.Module:
    """
    Expects cfg fields (lowercase):
      cfg.model.name         (single model, set by main loop)
      cfg.dataset.num_classes
      cfg.dataset.chn_names
      cfg.model.patch_size
      cfg.paths.labram / cfg.paths.biot / cfg.paths.cbramod (as needed)
      cfg.train.tuning_mode
      cfg.train.use_zscore
      cfg.model.head.*
    """
    model_name = str(_require(cfg, "model.name")).lower()
    
    # lzs 修改
    dataset_path = str(_require(cfg, "paths.dataset_name"))
    with open(os.path.join(dataset_path,'dataset_info.json'), "r", encoding="utf-8") as f:
        dataset_json = json.load(f)
    
    num_classes = dataset_json['dataset']['num_labels']
    chn_names = dataset_json['dataset']['channels']
    
    
    patch_size = int(_require(cfg, "model.patch_size"))
    print(f"patch_size: {patch_size}")
    time_points = dataset_json['processing']['window_sec'] * dataset_json['processing']['target_sampling_rate']
    use_zscore = bool(_require(cfg, "train.use_zscore"))
    norm = PerTrialZScore() if use_zscore else IdentityNorm()
    # ----- EEGNet -----
    if model_name == "eegnet":
        num_ch = len(chn_names)
        extractor = EEGNet(chans=num_ch,time_point=time_points,norm=norm)
       
    # ----- LaBraM -----
    elif model_name == "labram":
        ckpt_path = str(_require(cfg, "paths.labram"))
        input_chans,ch_use = map_channels_to_indices(chn_names)
        num_ch = len(chn_names)

        raw = create_labram(
            "labram_base_patch200_200",
            pretrained=False,
            num_classes=num_classes,
            init_values=0.1,
            num_ch=num_ch,
        )
        load_checkpoint_any(raw, ckpt_path, model_type="labram", strict=False)

        extractor = LaBraMFeatureExtractor(
            raw,
            pool="mean",
            patch_size=patch_size,
            norm=norm,
            input_chans=input_chans,
            ch_use=ch_use
        )

    # ----- BIOT -----
    elif model_name == "biot":
        ckpt_path = str(_require(cfg, "paths.biot"))
        in_ch = len(chn_names)
        hop = int(_require(cfg, "model.biot_hop"))

        raw = BIOTClassifier(
            n_classes=num_classes,
            n_channels=in_ch,
            n_fft=patch_size,
            hop_length=hop,
        )
        load_checkpoint_any(raw, ckpt_path, model_type="biot", strict=False)

        extractor = BIOTFeatureExtractor(
            raw,
            norm=norm,
            expected_channels=in_ch,
            strict_channels=False,
        )

    # ----- CBraMod -----
    elif model_name == "cbramod":
        ckpt_path = str(_require(cfg, "paths.cbramod"))
        raw = CBraMod()
        load_checkpoint_any(raw, ckpt_path, model_type="cbramod", strict=False)

        extractor = CBraModFeatureExtractor(
            raw,
            patch_size=patch_size,
            norm=norm,
        )

    # ----- Brianomni -----
    elif model_name == "brainomni":
        ckpt_path = str(_require(cfg, "paths.brainomni"))
        cfg_path = str(_require(cfg, "paths.brainomni_config"))
        static_pos, static_sensor = map_channels_to_pos_and_type(chn_names)

        def get_brainomni_model(ckpt_path,cfg_path):
            with open(cfg_path, 'r') as f:
                    cfg = json.load(f)
            model = BrainOmni(**cfg)
            pt_path = ckpt_path
            if os.path.exists(pt_path):
                model.load_state_dict(torch.load(pt_path, map_location="cpu", weights_only=True), strict=False)
            return model, cfg
        
        raw, brain_cfg = get_brainomni_model(ckpt_path, cfg_path)
        feature_dim = brain_cfg.get("lm_dim", 512)
        n_neuro = brain_cfg.get("n_neuro", None)
        extractor = BrainOmniFeatureExtractor(
            model=raw,
            static_pos=static_pos,     
            static_sensor_type=static_sensor, 
            feature_dim=feature_dim,
            output_num_tokens=n_neuro,
            norm=norm
        )

    # ----- FEMBA -----
    elif model_name == "femba":
        ckpt_path = str(_require(cfg, "paths.femba"))
        raw = FEMBA(seq_length=1280, num_channels=22, num_classes=num_classes, embed_dim=35, num_blocks=4)
        load_checkpoint_any(raw, ckpt_path, model_type="femba", strict=False)

        extractor = FembaFeatureExtractor(
            model=raw,
            input_freq=200, 
            input_chn_names = chn_names,
            norm=norm
        )
    
    # -----   NeuroGPT -----
    elif model_name == "neurogpt":
        ckpt_path = str(_require(cfg, "paths.neurogpt"))
        encoder = EEGConformer(
            n_outputs=num_classes, 
            n_chans=22, 
            n_times=200,
            n_filters_time=40,
            filter_time_length=25,
            pool_time_length=75,
            pool_time_stride=15,
            att_depth=6,
            att_heads=10,
            is_decoding_mode=False
        )
        full_sd = torch.load(ckpt_path, map_location="cpu")
        encoder_sd = {}
        for k, v in full_sd.items():
            if k.startswith("encoder."):
                new_key = k.replace("encoder.", "")
                encoder_sd[new_key] = v
        encoder.load_state_dict(encoder_sd, strict=False)

        extractor = NeuroGPTFeatureExtractor( encoder, input_chns=chn_names, embed_dim=40, chunk_len=200, stride=200, input_freq=200,norm=norm)
    

    # ----- NeuroLM -----
    elif model_name == "neurolm":
        ckpt_path = str(_require(cfg, "paths.neurolm"))
        ckpt_path_vq  = str(_require(cfg, "paths.neurolm_vq")) 

        # GPT config must match pretraining
        gpt_cfg = GPTConfig(
            block_size=1024,
            vocab_size=50304,
            n_layer=12,
            n_head=12,
            n_embd=768,
            dropout=0.0,
            bias=True,
        )

        raw = NeuroLM(
            GPT_config=gpt_cfg,
            init_from='scratch', # 'gpt2' / 'gpt2-medium' / 'scratch'
            tokenizer_ckpt_path=ckpt_path_vq,
        )

        load_checkpoint_any(raw, ckpt_path, model_type="neurolm", strict=False)

        # target_samples = int(
        #     OmegaConf.select(cfg, "model.neurolm.target_samples")
        #     or OmegaConf.select(cfg, "dataset.samples")
        # )

        extractor = NeuroLMFeatureExtractor(
            model=raw,
            # target_samples=target_samples,
            pool=str(OmegaConf.select(cfg, "model.neurolm.pool") or "mean"),
            norm=norm,
        )


    # ----- EEGMamba -----
    elif model_name == "eegmamba":
        ckpt_path = str(_require(cfg, "paths.eegmamba"))

        patch_size = int(_require(cfg, "model.patch_size"))
        target_samples = int(
            OmegaConf.select(cfg, "model.eegmamba.target_samples") or 6000
        )

        seq_len = target_samples // patch_size
        # if seq_len != 30:
        #     raise ValueError(
        #         "[factory] EEGMamba pretrained with seq_len=30. "
        #         f"Got target_samples={target_samples}."
        #     )

        raw = EEGMamba(
            in_dim=patch_size,
            out_dim=patch_size,
            d_model=patch_size,
            seq_len=seq_len,
        )

        load_checkpoint_any(raw, ckpt_path, model_type="eegmamba", strict=False)

        extractor = EEGMambaFeatureExtractor(
            model=raw,
            patch_size=patch_size,
            target_samples=target_samples,
            norm=norm,
            crop_mode=str(OmegaConf.select(cfg, "model.eegmamba.crop_mode") or "center_crop"),
            pad_mode=str(OmegaConf.select(cfg, "model.eegmamba.pad_mode") or "zero"),
        )


    # ----- BENDR -----
    elif model_name == "bendr":
        ckpt_path = str(_require(cfg, "paths.bendr"))

        # samples = OmegaConf.select(cfg, "dataset.samples")
        # if samples is None:
        #     raise ValueError("[factory] BENDR requires dataset.samples")
        expected_samples = 6000 # BENDR was trained at samples 200 * 30
        samples = int(expected_samples)

        channels = len(chn_names)

        # Use pretrained-default architecture
        raw = BENDRClassification(
            targets=num_classes,
            samples=samples,
            channels=channels,
            encoder_h=512,
            contextualizer_hidden=3076,
            projection_head=False,
            dropout=0.0,
            layer_drop=0.0,
        )

        # Load pytorch_model.bin
        load_checkpoint_any(raw, ckpt_path, model_type="bendr", strict=False)

        extractor = BENDRFeatureExtractor(
            model=raw,
            norm=norm,
        )

    # ----- Brant -----
    elif model_name == "brant":
        # Brant pretrained settings are encoded in the checkpoint tensor shapes.
        # To ensure alignment, infer them from the checkpoint instead of hardcoding.
        time_ckpt = str(_require(cfg, "paths.brant_time"))
        ch_ckpt = OmegaConf.select(cfg, "paths.brant_channel")

        def _infer_brant_time_spec(path: str):
            sd = torch.load(path, map_location="cpu", weights_only=False)
            # checkpoints are stored as an OrderedDict state_dict
            pos = sd["module.input_embedding.positional_encoding"]
            proj_w = sd["module.input_embedding.proj.0.weight"]
            band = sd["module.input_embedding.band_encoding"]
            ff_w = sd["module.trans_enc.layers.0.linear1.weight"]
            # count encoder layers
            layer_ids = set()
            for k in sd.keys():
                if "module.trans_enc.layers." in k:
                    try:
                        layer_ids.add(int(k.split("module.trans_enc.layers.", 1)[1].split(".", 1)[0]))
                    except Exception:
                        pass
            n_layer = (max(layer_ids) + 1) if layer_ids else 0

            return {
                "seq_len": int(pos.shape[0]),
                "d_model": int(pos.shape[1]),
                "in_dim": int(proj_w.shape[1]),
                "band_num": int(band.shape[0]),
                "dim_feedforward": int(ff_w.shape[0]),
                "n_layer_time": int(n_layer),
            }

        def _infer_brant_channel_spec(path: str):
            sd = torch.load(path, map_location="cpu", weights_only=False)
            # count encoder layers
            layer_ids = set()
            for k in sd.keys():
                if "module.trans_enc.layers." in k:
                    try:
                        layer_ids.add(int(k.split("module.trans_enc.layers.", 1)[1].split(".", 1)[0]))
                    except Exception:
                        pass
            n_layer = (max(layer_ids) + 1) if layer_ids else 0
            out_w = sd.get("module.proj_out.0.weight", None)
            out_dim = int(out_w.shape[0]) if isinstance(out_w, torch.Tensor) and out_w.ndim == 2 else None
            return {"n_layer_ch": int(n_layer), "ch_out_dim": out_dim}

        spec_t = _infer_brant_time_spec(time_ckpt)
        spec_c = _infer_brant_channel_spec(str(ch_ckpt)) if ch_ckpt is not None else {"n_layer_ch": 0, "ch_out_dim": None}

        # Heads aren't inferable from state_dict shapes; keep config default (must divide d_model).
        nhead_time = int(OmegaConf.select(cfg, "model.brant_nhead_time") or 8)
        nhead_ch = int(OmegaConf.select(cfg, "model.brant_nhead_ch") or 8)

        # Informative warning if the user-provided patch_size/seq_len differ from the pretrained spec.
        cfg_patch = int(_require(cfg, "model.patch_size"))
        if cfg_patch != spec_t["in_dim"]:
            print(
                f"[brant] WARNING: cfg.model.patch_size={cfg_patch} but Brant checkpoint expects in_dim={spec_t['in_dim']}. "
                "Using checkpoint in_dim for Brant."
            )
        cfg_seq = OmegaConf.select(cfg, "model.brant_seq_len")
        if cfg_seq is not None and int(cfg_seq) != spec_t["seq_len"]:
            print(
                f"[brant] WARNING: cfg.model.brant_seq_len={int(cfg_seq)} but checkpoint expects seq_len={spec_t['seq_len']}. "
                "Using checkpoint seq_len for Brant."
            )

        # Brant expects input windows of length T = seq_len * in_dim.
        # The feature extractor will crop/pad to this length; if your dataset windows are much shorter,
        # this may heavily pad with zeros and degrade performance.
        target_samples = int(spec_t["seq_len"] * spec_t["in_dim"])
        print(f"[brant] pretrained spec: seq_len={spec_t['seq_len']} in_dim={spec_t['in_dim']} => target_samples={target_samples}")

        raw = Brant(
            in_dim=spec_t["in_dim"],
            seq_len=spec_t["seq_len"],
            d_model=spec_t["d_model"],
            dim_feedforward=spec_t["dim_feedforward"],
            n_layer_time=spec_t["n_layer_time"],
            nhead_time=nhead_time,
            band_num=spec_t["band_num"],
            project_mode="linear",
            learnable_mask=False,
            use_channel_encoder=True,
            n_layer_ch=int(spec_c["n_layer_ch"] or 0),
            nhead_ch=nhead_ch,
            ch_out_dim=int(spec_c["ch_out_dim"] or spec_t["in_dim"]),
        )

        load_brant_pretrained(
            model=raw,
            time_ckpt=time_ckpt,
            channel_ckpt=ch_ckpt,
            strict=False,
        )

        extractor = BrantPretrainFeatureExtractor(
            model=raw,
            seg_len=spec_t["in_dim"],
            seq_len=spec_t["seq_len"],
            band_num=spec_t["band_num"],
            use_power=bool(OmegaConf.select(cfg, "model.brant_use_power") or False),
            norm=norm,
        )

    
    else:
        raise ValueError(f"Unknown model.name: {model_name}")
    
    

    # ----- head -----
    head_type = str(_require(cfg, "model.head.type"))
    hidden = OmegaConf.select(cfg, "model.head.hidden")
    dropout = float(_require(cfg, "model.head.dropout"))
    act = str(_require(cfg, "model.head.act"))

    head = ProbeHead(
        in_dim=int(extractor.embed_dim),
        out_dim=num_classes,
        head_type=head_type,
        hidden_dim=(int(hidden) if hidden is not None else None),
        dropout=dropout,
        act=act,
    )

    model = DownstreamModel(extractor, head)
    return model
