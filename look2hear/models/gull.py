import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .base_model import BaseModel

class RMVN(nn.Module):
    def __init__(self, in_channels):
        super(RMVN, self).__init__()
        self.mean = nn.Parameter(torch.zeros(in_channels))
        self.var = nn.Parameter(torch.ones(in_channels))
        self.eps = torch.finfo(torch.float32).eps

    def forward(self, x):
        return (x - x.mean(dim=1, keepdim=True)) / (x.var(dim=1, keepdim=True) + self.eps).sqrt() * self.var.view(1, -1, 1) + self.mean.view(1, -1, 1)

class Encoder_layer(nn.Module):
    def __init__(self, in_channels, out_channels, bidirectional=False):
        super(Encoder_layer, self).__init__()
        # Process Time
        self.process_time = nn.ModuleList([
            RMVN(in_channels),
            nn.LSTM(in_channels, out_channels, 1, batch_first=True, bidirectional=bidirectional),
            nn.Linear(out_channels * 2 if bidirectional else out_channels, in_channels)
        ])

        # Process Frequency
        self.process_frequency = nn.ModuleList([
            RMVN(in_channels),
            nn.LSTM(in_channels, out_channels, 1, batch_first=True, bidirectional=bidirectional),
            nn.Linear(out_channels * 2 if bidirectional else out_channels, in_channels)
        ])

    def forward(self, x):
        # x: [batch, n_band, n_chan, time]
        batch, n_band, n_chan, time = x.shape
        # Process Time
        t_1 = self.process_time[0](x.permute(0, 1, 2, 3).contiguous().view(batch * n_band, n_chan, time))
        t_2, _ = self.process_time[1](t_1.transpose(1, 2).contiguous()) # [batch * n_band, time, out_channels]
        t_3 = self.process_time[2](t_2).view(batch, n_band, n_chan, time) # [batch, n_band, n_chan, time]
        x = x + t_3

        # Process Frequency
        f_1 = self.process_frequency[0](x.permute(0, 3, 2, 1).contiguous().view(batch * time, n_chan, n_band))
        f_2, _ = self.process_frequency[1](f_1.transpose(1, 2).contiguous()) # [batch * time, n_band, out_channels]
        f_3 = self.process_frequency[2](f_2).view(batch, time, n_chan, n_band).permute(0, 3, 2, 1).contiguous() # [batch, n_band, n_chan, time]
        x = x + f_3
        
        return x
    
class Quantization_layer(nn.Module):
    def __init__(self, nband, num_code, code_channels, decay=0.99, layer_idx = 0):
        super(Quantization_layer, self).__init__()
        self.nband = nband
        self.num_code = num_code
        self.code_channels = code_channels
        self.decay = decay
        self.eps = torch.finfo(torch.float32).eps
        self.stale_tolerance = 100
        self.layer_idx = layer_idx

        if self.layer_idx == 0:
            codebook = torch.empty(nband, num_code, code_channels).normal_()
            codebook = codebook / (codebook.pow(2).sum(-1) + self.eps).sqrt().unsqueeze(-1)
        else:
            codebook = torch.empty(nband, num_code, code_channels).normal_() / code_channels
            codebook[:, 0] = codebook[:, 0] * 0
        
        self.register_buffer('codebook', codebook)
        self.register_buffer("ema_weight", self.codebook.clone())
        self.register_buffer("ema_count", torch.zeros(self.nband, self.num_code))
        self.register_buffer("stale_counter", torch.ones(self.nband, self.num_code)*self.stale_tolerance)
    
    def forward(self, x):
        # x: [batch, n_band, n_chan, time]
        batch, n_band, n_chan, time = x.shape

        x_detach = x.detach().permute(0, 3, 1, 2).contiguous().view(batch * time, n_band, n_chan)
        eq_distance = (x_detach.pow(2).sum(-1, keepdim=True) + self.codebook.pow(2).sum(-1) - 2 * torch.stack([torch.matmul(x_detach[:, i], self.codebook[i].t()) for i in range(n_band)], dim=1))
        
        # Find the closest code
        min_distance, min_idx = eq_distance.min(-1)
        quantized = []
        for i in range(n_band):
            quantized.append(torch.gather(self.codebook[i], 0, min_idx[:, i].unsqueeze(-1).expand(-1, self.code_channels)))
        quantized = torch.stack(quantized, dim=1).view(batch, time, n_band, n_chan).permute(0, 2, 3, 1).contiguous() # [batch, n_band, n_chan, time]

        # Update codebook
        encodings = F.one_hot(min_idx, self.num_code).float() # [batch * time, n_band, num_code]

        if self.training:
            self.ema_count = self.ema_count * self.decay + (1 - self.decay) * encodings.sum(0)
            
            direction = encodings.permute(1, 2, 0).contiguous().bmm(x_detach.permute(1, 0, 2).contiguous()) # [n_band, num_code, n_chan]
            self.ema_weight = self.ema_weight * self.decay + (1 - self.decay) * direction

            n = torch.sum(self.ema_count, dim=-1, keepdim=True)
            self.ema_count = (self.ema_count + self.eps) / (n + self.num_code * self.eps) * n

            self.codebook = self.ema_weight / self.ema_count.unsqueeze(-1)

            # Stale counter
            stale_codes = (encodings.sum(0) == 0).float()
            self.stale_counter = self.stale_counter - stale_codes

            # Reset stale codes
            replace_code = (self.stale_counter == 0).float() # [n_band, num_code]
            if replace_code.sum(-1).max() > 0:
                random_input_idx = torch.randperm(x_detach.shape[0])
                random_input = x_detach[random_input_idx].view(x_detach.shape)
                if random_input.shape[0] < self.num_code:
                    random_input = torch.cat([random_input]*(self.num_code // random_input.shape[0] + 1), dim=0)
                random_input = random_input[:self.num_code].contiguous().transpose(0, 1).contiguous() # [n_band, num_code, n_chan]
                self.codebook = self.codebook * (1 - replace_code.unsqueeze(-1)) + random_input * replace_code.unsqueeze(-1)
                self.ema_weight = self.ema_weight * (1 - replace_code.unsqueeze(-1)) + random_input * replace_code.unsqueeze(-1)
                self.ema_count = self.ema_count * (1 - replace_code)
                self.stale_counter = self.stale_tolerance * replace_code + self.stale_counter

            # Code constrains
            if self.layer_idx == 0:
                self.codebook = self.codebook / (self.codebook.pow(2).sum(-1) + self.eps).sqrt().unsqueeze(-1)
            else:
                self.codebook[:, 0] = self.codebook[:, 0] * 0
                self.ema_weight[:,0] = self.ema_weight[:,0] * 0
        
        return quantized


class Quantization(nn.Module):
    def __init__(self, nband, num_code, code_channels, decay=0.99, num_RVQ=5):
        super(Quantization, self).__init__()
        self.num_code = num_code
        self.code_channels = code_channels
        self.decay = decay
        self.num_RVQ = num_RVQ
        self.nband = nband
        self.eps = torch.finfo(torch.float32).eps

        self.quant_layers = nn.ModuleList([])
        for i in range(self.num_RVQ):
            self.quant_layers.append(Quantization_layer(nband, num_code, code_channels, decay, i))

    def forward(self, x):
        # x: [batch, n_band, n_chan, time]
        batch, n_band, n_chan, time = x.shape
        quantized = []
        residual = x.clone()
        for i in range(self.num_RVQ):
            quantized_tmp = self.quant_layers[i](residual)
            residual = residual - quantized_tmp
            if i == 0:
                quantized.append(quantized_tmp)
            else:
                quantized.append(quantized[-1] + quantized_tmp)

        quantized = torch.stack(quantized, dim=-1) # [batch, n_band, n_chan, time, num_RVQ]

        commit_loss = 0
        for i in range(self.num_RVQ):
            commit_loss = commit_loss + F.mse_loss(x, quantized.detach()[..., i])

        quantized = quantized / (quantized.pow(2).sum(2, keepdim=True) + self.eps).sqrt()

        return quantized, commit_loss

class Gull(BaseModel):
    def __init__(
            self,
            window_size=20,
            hop_size=10,
            sample_rate=16000,
            in_channels=128,
            num_layers=4,
            total_bps = 100000,
            num_RVQ = 5
    ):
        super().__init__()
        self.eps = torch.finfo(torch.float32).eps
        self.window_size = int(window_size * sample_rate / 1000)
        self.hop_size = int(hop_size * sample_rate / 1000)
        self.n_fft = self.window_size // 2 + 1
        self.in_channels = in_channels
        self.num_layers = num_layers
        self.total_bps = total_bps
        self.num_RVQ = num_RVQ

        self.bpf = total_bps // (2000 // self.window_size)

        # Band split
        # 400 Hz * 10, 1 kHz * 4, 2 kHz * 4, 4 kHz * 1
        bandwidth_400 = int(np.floor(400 / (sample_rate / 2.) * self.n_fft))
        bandwidth_1k = int(np.floor(1000 / (sample_rate / 2.) * self.n_fft))
        bandwidth_2k = int(np.floor(2000 / (sample_rate / 2.) * self.n_fft))
        bandwidth_4k = int(np.floor(4000 / (sample_rate / 2.) * self.n_fft))
        self.band_width = [bandwidth_400 * 10, bandwidth_1k * 4, bandwidth_2k * 4, bandwidth_4k]
        self.band_width.append(self.n_fft - sum(self.band_width))
        self.nband = len(self.band_width)
        self.bpb = self.bpf // self.nband
        
        self.pre_band = nn.ModuleList([])
        for i in range(self.nband):
            self.pre_band.append(nn.Sequential(
                RMVN(self.band_width[i] * 2 + 1),
                nn.Conv1d(self.band_width[i] * 2 + 1, in_channels, 1),
            ))
        
        # Encoder
        self.encoder = []
        for i in range(num_layers):
            self.encoder.append(Encoder_layer(in_channels, in_channels*2, bidirectional=False))
        self.encoder = nn.Sequential(*self.encoder)

        # Quantization
        self.codebook = Quantization(self.nband, self.bpb, in_channels, num_RVQ=self.num_RVQ)

        # Decoder
        self.decoder = []
        for i in range(num_layers):
            self.decoder.append(Encoder_layer(in_channels, in_channels, bidirectional=False))
        self.decoder = nn.Sequential(*self.decoder)

        # Band Restoration
        self.restoration = nn.ModuleList([])
        for i in range(self.nband):
            self.restoration.append(nn.Conv1d(in_channels, self.band_width[i] * 4, 1))
    
    def gain_shape(self, x):
        B, nch, nsample = x.shape

        spec = torch.stft(x.view(B*nch, nsample), n_fft=self.window_size, hop_length=self.hop_size, 
                          window=torch.hann_window(self.window_size).to(x.device), return_complex=True)

        subband_spec = []
        subband_spec_norm = []
        subband_power = []
        band_idx = 0
        for i in range(self.nband):
            this_spec = spec[:,band_idx:band_idx+self.band_width[i]]
            subband_spec.append(this_spec)  # B, BW, T
            subband_power.append((this_spec.abs().pow(2).sum(1) + self.eps).sqrt().unsqueeze(1))  # B, 1, T
            subband_spec_norm.append(torch.complex(this_spec.real / subband_power[-1], this_spec.imag / subband_power[-1]))  # B, BW, T
            band_idx += self.band_width[i]
        subband_power = torch.cat(subband_power, 1)  # B, nband, T

        return subband_spec, subband_spec_norm, subband_power
    
    def feature_extraction(self, x):
        subband_spec, subband_spec_norm, subband_power = self.gain_shape(x)

        subband_feature = []
        for i in range(self.nband):
            concat_spec = torch.cat([subband_spec_norm[i].real, subband_spec_norm[i].imag, torch.log(subband_power[:,i].unsqueeze(1))], 1)
            subband_feature.append(self.pre_band[i](concat_spec))
        subband_feature = torch.stack(subband_feature, 1)  # B, nband, N, T

        return subband_feature

    def forward(self, x, vq_depth=100):
        B, nch, nsample = x.shape

        subband_feature = self.feature_extraction(x) # B, nband, N, T

        # Encoder
        enc_output = self.encoder(subband_feature)
        enc_output = enc_output / (enc_output.pow(2).sum(2) + self.eps).sqrt().unsqueeze(2)  # unit norm
        
        # Quantization
        quantized_emb, latent_loss = self.codebook(enc_output)
        selected_quantized_emb = []
        if self.training:
            select_len = np.random.choice(self.num_RVQ, B*nch)
            for i in range(B*nch):
                selected_quantized_emb.append(quantized_emb[i,:,:,:,select_len[i]])
            selected_quantized_emb = torch.stack(selected_quantized_emb, 0)
        else:
            vq_depth = min(max(vq_depth-1, 0), self.num_RVQ - 1)
            selected_quantized_emb = quantized_emb[:,:,:,:,vq_depth]
        
        quantized_emb = enc_output + (selected_quantized_emb - enc_output).detach()

        # Decoder
        output = self.decoder(quantized_emb)
        est_spec = []
        for i in range(self.nband):
            this_RI = self.restoration[i](output[:,i]).view(B*nch, 2, 2, self.band_width[i], -1)
            this_RI = this_RI[:,0] * torch.sigmoid(this_RI[:,1])
            est_spec.append(torch.complex(this_RI[:,0], this_RI[:,1]))
        est_spec = torch.cat(est_spec, 1)
        output = torch.istft(est_spec, n_fft=self.window_size, hop_length=self.hop_size, 
                                window=torch.hann_window(self.window_size).to(x.device), length=nsample).view(B, nch, -1)

        if self.training:
            return output, latent_loss
        else:
            return output