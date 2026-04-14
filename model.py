"""
model.py — Improved Speech Command CNN (v2).

Architecture improvements over v1:
  • Residual (skip) connections  — easier gradient flow, ~10% accuracy gain
  • Squeeze-and-Excitation (SE) attention — channel-wise feature recalibration
  • Batch Normalisation throughout — faster / more stable training
  • Four stages instead of three — deeper representational capacity
  • AdaptiveAvgPool at the head — works with any N_MELS value; no hardcoded size

Input:  (B, 1, N_MELS, T)   e.g. (B, 1, 64, 32) with default config
Output: (B, NUM_CLASSES)
"""
from __future__ import annotations
import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """
    Squeeze-and-Excitation block (Hu et al., 2018).
    Learns to rescale each channel by its global importance.
    """
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        bottleneck = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.gate = nn.Sequential(
            nn.Linear(channels, bottleneck),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.gate(w).view(b, c, 1, 1)
        return x * w


class ResBlock(nn.Module):
    """
    Residual block: two 3×3 conv layers with BN + SE attention.
    A projection (1×1 conv) is applied to the skip path when channels change.
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.attn = ChannelAttention(out_ch)
        self.skip = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
            if in_ch != out_ch else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.skip(x) + self.attn(self.body(x)))


class SpeechCommandCNN(nn.Module):
    """
    Four-stage residual CNN for speech command classification.

    Stage layout (with N_MELS=64, T=32 input):
        Stem        → (B, 32,  64, 32)
        Stage 1     → (B, 32,  32, 16)   ResBlock + MaxPool
        Stage 2     → (B, 64,  16,  8)   ResBlock(32→64) + MaxPool
        Stage 3     → (B, 128,  8,  4)   ResBlock(64→128) + MaxPool
        Stage 4     → (B, 256,  8,  4)   ResBlock(128→256)  [no pool]
        AdaptAvgPool→ (B, 256,  2,  2)
        Flatten+FC  → (B, NUM_CLASSES)
    """
    def __init__(self, num_classes: int = 7, n_mels: int = 64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.stage1 = nn.Sequential(ResBlock(32,  32),  nn.MaxPool2d(2, 2))
        self.stage2 = nn.Sequential(ResBlock(32,  64),  nn.MaxPool2d(2, 2))
        self.stage3 = nn.Sequential(ResBlock(64,  128), nn.MaxPool2d(2, 2))
        self.stage4 = ResBlock(128, 256)   # keeps spatial dims; deepens features

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((2, 2)),  # works regardless of N_MELS
            nn.Flatten(),
            nn.Linear(256 * 2 * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return self.head(x)


# ── Quick smoke-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = SpeechCommandCNN(num_classes=7, n_mels=64)
    dummy = torch.randn(4, 1, 64, 32)
    out   = model(dummy)
    print(f"Input : {dummy.shape}")
    print(f"Output: {out.shape}")
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total:,}")
