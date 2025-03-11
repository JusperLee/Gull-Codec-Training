import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

def freq_MAE(output, target, win=20, sr=44100):
    win = int (sr / 1000 * win)
    stride = win // 2
    eps = torch.finfo(torch.float32).eps

    est_spec = torch.stft(output.view(-1, output.shape[-1]), n_fft=win, hop_length=stride, 
                          window=torch.hann_window(win).to(output.device).float(),
                          return_complex=True)
    target_spec = torch.stft(target.view(-1, target.shape[-1]), n_fft=win, hop_length=stride, 
                             window=torch.hann_window(win).to(target.device).float(),
                             return_complex=True)
    
    return ((est_spec.real - target_spec.real).abs().mean() + (est_spec.imag - target_spec.imag).abs().mean()) / (target_spec.abs().mean() + eps)

class DiscriminatorLoss(_Loss):
    def __init__(self, win=20, sr=44100):
        super(DiscriminatorLoss, self).__init__()
        self.win = win
        self.sr = sr

    def forward(self, est, target):
        loss_real = 0
        loss_fake = 0
        for i in range(len(est)):
            loss_real = loss_real + (target[i] - 1).pow(2).mean() / len(target)
            loss_fake = loss_fake + est[i].pow(2).mean() / len(est)
        return loss_real + loss_fake
    
class GeneratorLoss(_Loss):
    def __init__(self, win=20, sr=44100):
        super(GeneratorLoss, self).__init__()
        self.win = win
        self.sr = sr
        self.eps = torch.finfo(torch.float32).eps

    def forward(self, raw_data, ests_outputs, commit_loss, d_ests_outputs, d_ests_feature_maps, targets_feature_maps):
        feature_matching = 0
        loss_fake = 0
        for i in range(len(d_ests_outputs)):
            loss_fake = loss_fake + (d_ests_outputs[i] - 1).pow(2).mean() / len(d_ests_outputs)
            for j in range(len(d_ests_feature_maps[i])):
                feature_matching = feature_matching + (d_ests_feature_maps[i][j] - targets_feature_maps[i][j].detach()).abs().mean() / (targets_feature_maps[i][j].detach().abs().mean() + self.eps)
        feature_matching = feature_matching / (len(d_ests_outputs) * len(d_ests_feature_maps[0]))
        reconstruction = freq_MAE(ests_outputs, raw_data)
        return feature_matching + loss_fake + reconstruction + 0.2 * commit_loss