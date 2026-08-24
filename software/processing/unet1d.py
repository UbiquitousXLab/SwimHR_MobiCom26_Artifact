"""1D U-Net for per-sample R-peak logits.

Channels-first input and output:
    input  : (B, num_leads=3, win_size=500)
    output : (B, win_size)
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


def conv_level(in_channels: int, out_channels: int, time_kernel: int) -> nn.Sequential:
    """Two Conv-BN-ReLU blocks that preserve temporal length."""
    padding = time_kernel // 2
    return nn.Sequential(
        nn.Conv1d(in_channels,  out_channels, kernel_size=time_kernel, padding=padding),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv1d(out_channels, out_channels, kernel_size=time_kernel, padding=padding),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(inplace=True),
    )


def upsample_time(in_channels: int, out_channels: int,
                  fix_odd_width: bool = False) -> nn.ConvTranspose1d:
    """Double temporal length, optionally restoring 62 to 125 instead of 124."""
    return nn.ConvTranspose1d(
        in_channels, out_channels,
        kernel_size=2, stride=2,
        output_padding=1 if fix_odd_width else 0,
    )


class UNet1D(nn.Module):
    """Four-level U-Net with widths (f, f, 2f, 2f) and kernels (7, 5, 3, 3)."""

    def __init__(self, in_channels: int = 3, base_filters: int = 16,
                 win_size: int = 500):
        super().__init__()
        self.hparams: Dict[str, int] = {
            'in_channels':  in_channels,
            'base_filters': base_filters,
            'win_size':     win_size,
        }
        f = base_filters
        w1, w2, w3, wb = f, f, 2 * f, 2 * f

        self.pool = nn.MaxPool1d(kernel_size=2)

        self.enc1 = conv_level(in_channels, w1, time_kernel=7)
        self.enc2 = conv_level(w1,          w2, time_kernel=5)
        self.enc3 = conv_level(w2,          w3, time_kernel=3)
        self.bottleneck = conv_level(w3,    wb, time_kernel=3)

        self.up1  = upsample_time(wb, w3, fix_odd_width=True)
        self.dec1 = conv_level(w3 + w3, w3, time_kernel=3)
        self.up2  = upsample_time(w3, w2)
        self.dec2 = conv_level(w2 + w2, w2, time_kernel=5)
        self.up3  = upsample_time(w2, w1)
        self.dec3 = conv_level(w1 + w1, w1, time_kernel=7)

        self.head = nn.Conv1d(w1, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b  = self.bottleneck(self.pool(e3))

        d1 = self.dec1(torch.cat([self.up1(b),  e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d1), e2], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d2), e1], dim=1))

        out = self.head(d3)
        return out.squeeze(1)
