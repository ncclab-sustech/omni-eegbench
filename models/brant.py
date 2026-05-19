import torch
import torch.nn as nn

from models.pre_model_encoder import TimeEncoder, ChannelEncoder

class Brant(nn.Module):
    """
    Backbone = TimeEncoder (+ optional ChannelEncoder)
    Input:
      data:  (B, C, S, L)  where commonly S=15, L=1500
      power: (B, C, S, band_num)  OR you can pass zeros and set use_power=False
    """
    def __init__(
        self,
        in_dim: int,          # seg_len (e.g., 1500 if project_mode='linear')
        seq_len: int,         # S (e.g., 15)
        d_model: int,
        dim_feedforward: int,
        n_layer_time: int,
        nhead_time: int,
        band_num: int,
        project_mode: str = "linear",
        learnable_mask: bool = False,
        use_channel_encoder: bool = True,
        n_layer_ch: int = 2,
        nhead_ch: int = 8,
        ch_out_dim: int | None = None,  # only needed if you care about ChannelEncoder "rec"
    ):
        super().__init__()

        self.time = TimeEncoder(
            in_dim=in_dim,
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            seq_len=seq_len,
            n_layer=n_layer_time,
            nhead=nhead_time,
            band_num=band_num,
            project_mode=project_mode,
            learnable_mask=learnable_mask,
        )

        self.use_channel_encoder = use_channel_encoder
        if use_channel_encoder:
            if ch_out_dim is None:
                ch_out_dim = in_dim  # or whatever matches their reconstruction target
            self.channel = ChannelEncoder(
                out_dim=ch_out_dim,
                d_model=d_model,
                dim_feedforward=dim_feedforward,
                n_layer=n_layer_ch,
                nhead=nhead_ch,
            )
        else:
            self.channel = None

    def forward(self, data, power=None, use_power=True):
        B, C, S, L = data.shape

        # ---- make dummy mask (ignored when need_mask=False) ----
        dummy_mask = torch.zeros((1,), dtype=torch.long, device=data.device)

        if power is None:
            # if you don't use power, pass zeros and set use_power=False
            power = torch.zeros((B, C, S, 1), device=data.device)
            use_power = False

        # ---- TimeEncoder ----
        # returns: (B*C, S, D)
        t = self.time(
            mask=dummy_mask,
            data=data,
            power=power,
            need_mask=False,
            mask_by_ch=False,
            rand_mask=False,
            mask_len=None,
            use_power=use_power,
        )

        # reshape to (B, C, S, D)
        D = t.shape[-1]
        t = t.view(B, C, S, D)

        if not self.use_channel_encoder:
            # simple pooled feature: average over channels and time
            feat = t.mean(dim=(1, 2))  # (B, D)
            return feat

        # ---- ChannelEncoder expects: (B*S, C, D) ----
        x = t.permute(0, 2, 1, 3).contiguous().view(B * S, C, D)  # (B*S, C, D)
        ch_z, _rec = self.channel(x)  # ch_z: (B*S, C, D)

        # pool channels then pool time
        ch_z = ch_z.mean(dim=1)              # (B*S, D)
        ch_z = ch_z.view(B, S, D).mean(dim=1)  # (B, D)
        return ch_z