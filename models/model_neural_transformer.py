"""
by Wei-Bang Jiang
https://github.com/935963004/NeuroLM
"""

import math
import torch.nn as nn
from models.model_gpt import Block
from einops import rearrange
from dataclasses import dataclass


class TemporalConv(nn.Module):
    """ EEG to Patch Embedding
    """
    def __init__(self, in_chans=1, out_chans=8):
        '''
        in_chans: in_chans of nn.Conv2d()
        out_chans: out_chans of nn.Conv2d(), determing the output dimension
        '''
        super().__init__()
        self.conv1 = nn.Conv2d(in_chans, out_chans, kernel_size=(1, 15), stride=(1, 8), padding=(0, 7))
        self.gelu1 = nn.GELU()
        self.norm1 = nn.GroupNorm(4, out_chans)
        self.conv2 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 3), padding=(0, 1))
        self.gelu2 = nn.GELU()
        self.norm2 = nn.GroupNorm(4, out_chans)
        self.conv3 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 3), padding=(0, 1))
        self.norm3 = nn.GroupNorm(4, out_chans)
        self.gelu3 = nn.GELU()
        self.l = nn.Sequential(
            nn.Linear(400, 768),
            nn.GELU()
        )

    def forward(self, x, **kwargs):
        B, NA, T = x.shape
        x = x.unsqueeze(1)
        x = self.gelu1(self.norm1(self.conv1(x)))
        x = self.gelu2(self.norm2(self.conv2(x)))
        x = self.gelu3(self.norm3(self.conv3(x)))
        x = rearrange(x, 'B C NA T -> B NA (T C)')
        x = self.l(x)
        return x


@dataclass
class NTConfig:
    block_size: int = 1024
    patch_size: int = 200
    num_classes: int = 0
    in_chans: int = 1
    out_chans: int = 16
    use_mean_pooling: bool = True
    init_scale: float = 0.001
    n_layer: int = 12
    n_head: int = 10
    n_embd: int = 400
    dropout: float = 0.0
    bias: bool = False # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster


class NeuralTransformer(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.num_classes = config.num_classes

        # To identify whether it is neural tokenizer or neural decoder. 
        # For the neural decoder, use linear projection (PatchEmbed) to project codebook dimension to hidden dimension.
        # Otherwise, use TemporalConv to extract temporal features from EEG signals.
        self.patch_embed = TemporalConv(out_chans=config.out_chans) if config.in_chans == 1 else nn.Linear(config.in_chans, config.n_embd)
        self.patch_size = config.patch_size

        self.pos_embed = nn.Embedding(256, config.n_embd)
        self.time_embed = nn.Embedding(64, config.n_embd)

        self.rel_pos_bias = None

        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.norm = nn.Identity() if config.use_mean_pooling else nn.LayerNorm(config.n_embd, eps=1e-6)
        self.fc_norm = nn.LayerNorm(config.n_embd, eps=1e-6) if config.use_mean_pooling else None
        self.head = nn.Linear(config.n_embd, self.num_classes) if self.num_classes > 0 else nn.Identity()

        self.pos_drop = nn.Dropout(p=config.dropout)

        if isinstance(self.head, nn.Linear):
            nn.init.trunc_normal_(self.head.weight, std=.02)
        self.apply(self._init_weights)
        self.fix_init_weight()

        if isinstance(self.head, nn.Linear):
            self.head.weight.data.mul_(config.init_scale)
            self.head.bias.data.mul_(config.init_scale)

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.c_proj.weight.data, layer_id + 1)
            rescale(layer.mlp.c_proj.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x, input_chans=None, input_times=None, mask=None, return_all_tokens=False, **kwargs):
        batch_size, n, t = x.shape
        x = self.patch_embed(x)

        # add position and temporal embeddings
        pos_embed_used = self.pos_embed(input_chans)
        x = x + pos_embed_used
        time_embed = self.time_embed(input_times)
        x = x + time_embed

        x = self.pos_drop(x)
        
        for blk in self.blocks:
            x = blk(x, mask)
        
        x = self.norm(x)
        if self.fc_norm is not None:
            if return_all_tokens:
                return self.fc_norm(x)
            else:
                return self.fc_norm(x.mean(1))
        else:
            return x

    def forward(self, x, input_chans=None, input_times=None, mask=None, return_all_tokens=False, **kwargs):
        '''
        x: [batch size, sequence length, patch size]
        '''
        x = self.forward_features(x, input_chans, input_times, mask, return_all_tokens=return_all_tokens, **kwargs)
        x = self.head(x)
        return x
