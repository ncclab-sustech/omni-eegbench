import torch
import torch.nn as nn


def _weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class InputEmbedding(nn.Module):
    def __init__(self, in_dim, c_dim, seq_len, d_model, band_num, project_mode, learnable_mask):
        super(InputEmbedding, self).__init__()

        self.mode = project_mode
        self.band_num = band_num
        self.band_encoding = nn.Parameter(torch.randn(band_num, d_model), requires_grad=True)  # learnable band encoding
        self.positional_encoding = nn.Parameter(torch.randn(seq_len, d_model), requires_grad=True)  # learnable positional encoding
        if learnable_mask:
            self.mask_encoding = nn.Parameter(torch.randn(in_dim), requires_grad=True)
            self.power_mask_encoding = nn.Parameter(torch.randn(d_model), requires_grad=True)
        else:
            self.mask_encoding = nn.Parameter(torch.zeros(in_dim), requires_grad=False)
            self.power_mask_encoding = nn.Parameter(torch.zeros(d_model), requires_grad=False)

        self.softmax = nn.Softmax(dim=-1)
        if project_mode == 'cnn':
            self.cnn = nn.Sequential(
                nn.Conv1d(1, c_dim, 150, 10),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.MaxPool1d(4, 2),
                nn.Conv1d(c_dim, c_dim*2, 10, 5),
                nn.ReLU(inplace=True),  
                nn.Dropout(0.5),
                nn.MaxPool1d(4, 2),
            )

            self.cnn_proj = nn.Sequential(
                nn.Linear(c_dim*2, d_model),
            )
        elif project_mode == 'linear':
            self.proj = nn.Sequential(
                nn.Linear(in_dim, d_model),
            )
        else:
            raise NotImplementedError

        self.apply(_weights_init)

    def forward(self, mask, data, power, need_mask, mask_by_ch, rand_mask, mask_len, use_power):
        bat_size, ch_num, seq_len, seg_len = data.shape
        mask_pow = False
        if use_power:
            power = self.softmax(power)
        power_emb = torch.einsum('hijk, kl->hijl', power, self.band_encoding)

        if need_mask:
            masked_x = data.clone()
            if rand_mask:
                if mask_by_ch:
                    masked_x[:, mask[:, 0], mask[:, 1], :] = self.mask_encoding
                    if use_power and mask_pow:
                        power_emb[:, mask[:, 0], mask[:, 1], :] = self.power_mask_encoding
                else:
                    masked_x = masked_x.reshape(bat_size, ch_num*seq_len, seg_len)
                    masked_x[:, mask, :] = self.mask_encoding
                    if use_power and mask_pow:
                        power_emb = power_emb.reshape(bat_size, ch_num * seq_len, -1)
                        power_emb[:, mask, :] = self.power_mask_encoding
            else:
                masked_x[:, :, -mask_len:, :] = self.mask_encoding
                if use_power and mask_pow:
                    power_emb[:, :, -mask_len:, :] = self.power_mask_encoding
        else:
            masked_x = data
        masked_x = masked_x.view(bat_size * ch_num, seq_len, seg_len)
        # projection
        if self.mode == 'cnn':
            masked_x = masked_x.view(bat_size*ch_num*seq_len, 1, seg_len)
            input_emb = torch.mean(self.cnn(masked_x), dim=-1)
            input_emb = self.cnn_proj(input_emb)
            input_emb = torch.transpose(input_emb, 1, 2)
        elif self.mode == 'linear':
            input_emb = self.proj(masked_x)

        # add encodings
        power_emb = power_emb.view(bat_size * ch_num, seq_len, -1)
        input_emb += power_emb
        input_emb += self.positional_encoding

        return input_emb


class TimeEncoder(nn.Module):
    def __init__(self, in_dim, d_model, dim_feedforward, seq_len, n_layer, nhead, band_num, project_mode, learnable_mask):
        super(TimeEncoder, self).__init__()

        self.input_embedding = InputEmbedding(in_dim=in_dim, c_dim=64, seq_len=seq_len, d_model=d_model, band_num=band_num, project_mode=project_mode, learnable_mask=learnable_mask)

        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True)
        self.trans_enc = nn.TransformerEncoder(enc_layer, num_layers=n_layer)

        self.apply(_weights_init)

    def forward(self, mask, data, power, need_mask=True, mask_by_ch=False, rand_mask=True, mask_len=None, use_power=True):
        input_emb = self.input_embedding(mask, data, power, need_mask, mask_by_ch, rand_mask, mask_len, use_power)
        trans_out = self.trans_enc(input_emb)

        return trans_out


class ChannelEncoder(nn.Module):
    def __init__(self, out_dim, d_model, dim_feedforward, n_layer, nhead):
        super(ChannelEncoder, self).__init__()

        self.proj_out = nn.Sequential(
            nn.Linear(d_model, out_dim),
        )

        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True)
        self.trans_enc = nn.TransformerEncoder(enc_layer, num_layers=n_layer)

        self.apply(_weights_init)

    def forward(self, time_z):
        # time_z.shape: bat_size*seq_len, ch_num, d_model
        ch_z = self.trans_enc(time_z)
        rec = self.proj_out(ch_z)

        return ch_z, rec
