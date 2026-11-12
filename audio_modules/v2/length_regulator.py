from typing import Tuple
import torch
import torch.nn as nn
from torch.nn import functional as F
from audio_modules.commons import sequence_mask
import numpy as np

f0_max = 1100.0
f0_min = 50.0
f0_mel_min = 1127 * np.log(1 + f0_min / 700)
f0_mel_max = 1127 * np.log(1 + f0_max / 700)


def f0_to_coarse(f0, f0_bin):
    f0_mel = 1127 * (1 + f0 / 700).log()
    a = (f0_bin - 2) / (f0_mel_max - f0_mel_min)
    f0_mel = torch.where(f0_mel > 0, f0_mel * a - (f0_mel_min * a - 1.0), f0_mel)
    f0_coarse = torch.round(f0_mel).long()
    f0_coarse = f0_coarse * (f0_coarse > 0)
    f0_coarse = f0_coarse + ((f0_coarse < 1) * 1)
    f0_coarse = f0_coarse * (f0_coarse < f0_bin)
    return f0_coarse + ((f0_coarse >= f0_bin) * (f0_bin - 1))


class InterpolateRegulator(nn.Module):
    def __init__(
        self,
        channels: int,
        sampling_ratios: Tuple,
        is_discrete: bool = False,
        in_channels: int = None,  # only applies to continuous input
        codebook_size: int = 1024,  # for discrete only
        out_channels: int = None,
        groups: int = 1,
        f0_condition: bool = False,
        n_f0_bins: int = 512,
    ):
        super().__init__()
        self.sampling_ratios = sampling_ratios
        out_channels = out_channels or channels
        model = nn.ModuleList([])
        if len(sampling_ratios) > 0:
            self.interpolate = True
            for _ in sampling_ratios:
                module = nn.Conv1d(channels, channels, 3, 1, 1)
                model.extend([module, nn.GroupNorm(groups, channels), nn.Mish()])
        else:
            self.interpolate = False
        model.append(
            nn.Conv1d(channels, out_channels, 1, 1)
            if channels != out_channels
            else nn.Identity()
        )
        self.model = nn.Sequential(*model)
        self.embedding = nn.Embedding(codebook_size, channels)
        self.is_discrete = is_discrete

        self.mask_token = nn.Parameter(torch.zeros(1, channels))

        if f0_condition:
            self.f0_embedding = nn.Embedding(n_f0_bins, channels)
            self.f0_condition = f0_condition
            self.n_f0_bins = n_f0_bins
            self.f0_bins = torch.arange(2, 1024, 1024 // n_f0_bins)
            self.f0_mask = nn.Parameter(torch.zeros(1, channels))
        else:
            self.f0_condition = False

        if not is_discrete:
            self.content_in_proj = nn.Linear(in_channels, channels)

    def forward(self, x, ylens=None, f0=None):
        if self.is_discrete:
            if len(x.size()) == 2:
                x = self.embedding(x)
            else:
                x = self.embedding(x[:, 0])
        else:
            x = self.content_in_proj(x)
        # x in (B, T, D)

        if self.interpolate:
            mask = sequence_mask(ylens).unsqueeze(-1)
            x = F.interpolate(
                x.transpose(1, 2).contiguous(), size=ylens.max(), mode="nearest"
            )
        else:
            x = x.transpose(1, 2).contiguous()
            mask = None
        if self.f0_condition:
            if f0 is None:
                x = x + self.f0_mask.unsqueeze(-1)
            else:
                quantized_f0 = f0_to_coarse(f0, self.n_f0_bins)
                quantized_f0 = quantized_f0.clamp(0, self.n_f0_bins - 1).long()
                f0_emb = self.f0_embedding(quantized_f0)
                f0_emb = F.interpolate(
                    f0_emb.transpose(1, 2).contiguous(),
                    size=ylens.max(),
                    mode="nearest",
                )
                x = x + f0_emb
        out = self.model(x).transpose(1, 2).contiguous()
        return out * mask if mask is not None else out, ylens
