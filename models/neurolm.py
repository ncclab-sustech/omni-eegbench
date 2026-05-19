"""
by Wei-Bang Jiang
https://github.com/935963004/NeuroLM
"""

import inspect
import torch
import torch.nn as nn
from torch.nn import functional as F
from models.model_gpt import *
from torch.autograd import Function
from models.model_neural_transformer import NTConfig
from models.model_neural_transformer import NeuralTransformer
from collections import OrderedDict


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class NeuroLM(nn.Module):

    def __init__(self,
                 GPT_config,
                 tokenizer_ckpt_path=None,
                 init_from='gpt2',
                 n_embd=768,
                 eeg_vocab_size=8192,
                 ):
        super().__init__()

        if init_from == 'scratch':
            self.GPT2 = GPT(GPT_config)
        elif init_from.startswith('gpt2'):
            override_args = dict(dropout=0.0)
            self.GPT2 = GPT.from_pretrained(init_from, override_args)
        self.GPT2.enlarge_wte(50304)
        self.GPT2.enlarge_lm_head(self.GPT2.config.vocab_size + eeg_vocab_size)

        if tokenizer_ckpt_path is not None:
            print('loading weight from VQ_align')
            encoder_args = dict(n_layer=12, n_head=10, n_embd=400, block_size=1024,
                                bias=False, dropout=0., num_classes=0, in_chans=1, out_chans=16)
            tokenizer_checkpoint = torch.load(tokenizer_ckpt_path)
            tokenizer_checkpoint_model_args = tokenizer_checkpoint['encoder_args']
            # force these config attributes to be equal otherwise we can't even resume training
            # the rest of the attributes (e.g. dropout) can stay as desired from command line
            for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias']:
                encoder_args[k] = tokenizer_checkpoint_model_args[k]
            tokenizer_checkpoint_model_args = tokenizer_checkpoint['decoder_args']
            # create the model
            encoder_conf = NTConfig(**encoder_args)
            self.tokenizer = NeuralTransformer(encoder_conf)
            tokenizer_state_dict = tokenizer_checkpoint['model']
            # fix the keys of the state dictionary :(
            # honestly no idea how checkpoints sometimes get this prefix, have to debug more
            unwanted_prefix = '_orig_mod.'
            for k,v in list(tokenizer_state_dict.items()):
                if k.startswith(unwanted_prefix):
                    tokenizer_state_dict[k[len(unwanted_prefix):]] = tokenizer_state_dict.pop(k)
            
            all_keys = list(tokenizer_state_dict.keys())
            new_dict = OrderedDict()
            for key in all_keys:
                if key.startswith('VQ.encoder.'):
                    new_dict[key[11:]] = tokenizer_state_dict[key]
            self.tokenizer.load_state_dict(new_dict)
        else:
            encoder_args = dict(n_layer=12, n_head=12, n_embd=768, block_size=1024,
                                bias=False, dropout=0., num_classes=0, in_chans=1, out_chans=16)
            encoder_conf = NTConfig(**encoder_args)
            self.tokenizer = NeuralTransformer(encoder_conf)
        
        for p in self.tokenizer.parameters():
            p.requires_grad = False

        self.pos_embed = nn.Embedding(256, self.GPT2.config.n_embd)

        # task layer
        self.encode_transform_layer = nn.Sequential(
            nn.Linear(n_embd, self.GPT2.config.n_embd),
            nn.GELU(),
        ) if n_embd != self.GPT2.config.n_embd else nn.Identity()

        self.encode_transform_layer.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x_eeg=None, y_eeg=None, x_text=None, y_text=None, input_chans=None, input_time=None, input_mask=None, eeg_mask=None, eeg_text_mask=None):
        """
        x_eeg: shape [B, N1, T]
        x_text: shape [B, N2]
        """
        if x_eeg is not None:
            input_mask = input_mask.unsqueeze(1).repeat(1, x_eeg.size(1), 1).unsqueeze(1)
            x_eeg = self.tokenizer(x_eeg, input_chans, input_time, input_mask, return_all_tokens=True)
            x_eeg = self.encode_transform_layer(x_eeg)
            x_eeg += self.pos_embed(input_chans)   

        logits, loss, accuracy = self.GPT2(x_eeg, y_eeg, x_text, y_text, input_time, eeg_mask, eeg_text_mask)

        log = {}
        split="train" if self.training else "val"
        if loss is not None:
            log[f'{split}/loss'] = loss.item()
        if accuracy is not None:
            log[f'{split}/accuracy'] = accuracy.item()

        return loss, log, logits
    
    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.GPT2.transformer.wpe.weight.numel()
        return n_params

    @torch.no_grad()
    def generate(self, x_eeg, x_text, input_chans, input_time, input_mask, eeg_mask=None, eeg_text_mask=None, max_new_tokens=10, temperature=1.0, top_k=1):
        if x_eeg is not None:
            input_mask = input_mask.unsqueeze(1).repeat(1, x_eeg.size(1), 1).unsqueeze(1)
            x_eeg = self.tokenizer(x_eeg, input_chans, input_time, input_mask, return_all_tokens=True)
            x_eeg = self.encode_transform_layer(x_eeg)
            x_eeg += self.pos_embed(input_chans)
            #input_time = torch.zeros((x_eeg.size(0), x_eeg.size(1)), device=x_eeg.device).int()
        
        for _ in range(max_new_tokens):
            logits, _, _ = self.GPT2(x_eeg=x_eeg, x_text=x_text, eeg_time_idx=input_time, eeg_mask=eeg_mask, eeg_text_mask=eeg_text_mask)
            logits = logits[:, -1, :50257] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            #_, idx_next = logits.max(-1)

            x_text = torch.cat((x_text, idx_next), dim=1)
            if eeg_text_mask is not None:
                eeg_text_mask = torch.cat((eeg_text_mask, torch.zeros((eeg_text_mask.size(0), eeg_text_mask.size(1), eeg_text_mask.size(2), 1), device=eeg_text_mask.device)), dim=-1)
                eeg_text_mask = torch.cat((eeg_text_mask, torch.ones((eeg_text_mask.size(0), eeg_text_mask.size(1), 1, eeg_text_mask.size(3)), device=eeg_text_mask.device)), dim=-2)
        
        return x_text
    
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer
