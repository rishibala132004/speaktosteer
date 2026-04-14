"""
augmentation.py — Audio and spectrogram augmentation for training.

Never apply these during inference.

Techniques:
  Waveform-level  : Gaussian noise, volume perturbation, time shift
  Spectrogram-level: SpecAugment (frequency + time masking)
  Batch-level     : Mixup (label-smoothing via interpolation)
"""
from __future__ import annotations
import random
import numpy as np
import torch
import torch.nn.functional as F


# ── Waveform augmentation ──────────────────────────────────────────────────────

def add_gaussian_noise(waveform: np.ndarray, snr_db: float | None = None) -> np.ndarray:
    """
    Add white Gaussian noise at a random SNR between 10 and 30 dB.
    Simulates distant microphones and reverberant rooms.
    """
    if snr_db is None:
        snr_db = random.uniform(10.0, 30.0)
    signal_power = float(np.mean(waveform ** 2)) + 1e-9
    noise_power  = signal_power / (10 ** (snr_db / 10.0))
    noise = np.random.randn(*waveform.shape).astype(np.float32) * np.sqrt(noise_power)
    return np.clip(waveform + noise, -1.0, 1.0).astype(np.float32)


def volume_perturbation(
    waveform: np.ndarray,
    min_gain: float = 0.5,
    max_gain: float = 1.5,
) -> np.ndarray:
    """
    Randomly scale amplitude.
    Teaches the model that the same word can be loud or quiet.
    """
    gain = random.uniform(min_gain, max_gain)
    return np.clip(waveform * gain, -1.0, 1.0).astype(np.float32)


def time_shift(waveform: np.ndarray, max_shift_ms: int = 100, sample_rate: int = 16000) -> np.ndarray:
    """
    Cyclically shift the waveform left or right by up to max_shift_ms ms.
    Simulates slightly delayed speech onset.
    """
    max_shift = int(sample_rate * max_shift_ms / 1000)
    shift = random.randint(-max_shift, max_shift)
    return np.roll(waveform, shift).astype(np.float32)


def augment_waveform(waveform: np.ndarray, p: float = 0.5) -> np.ndarray:
    """
    Apply one random waveform-level augmentation with probability p.
    Call this once per sample during dataset preprocessing.
    """
    if random.random() > p:
        return waveform
    choice = random.random()
    if choice < 0.40:
        return add_gaussian_noise(waveform)
    elif choice < 0.70:
        return volume_perturbation(waveform)
    else:
        return time_shift(waveform)


# ── Spectrogram augmentation ───────────────────────────────────────────────────

def spec_augment(
    mel: torch.Tensor,
    freq_mask_max: int = 10,
    time_mask_max: int  = 6,
    num_freq_masks: int = 2,
    num_time_masks: int = 2,
) -> torch.Tensor:
    """
    SpecAugment (Park et al., 2019).
    Randomly zeros out horizontal (frequency) and vertical (time) bands.

    Args:
        mel           : shape (1, N_MELS, T)
        freq_mask_max : maximum width of each frequency mask
        time_mask_max : maximum width of each time mask
        num_freq_masks: how many frequency masks to apply
        num_time_masks: how many time masks to apply
    """
    mel = mel.clone()
    _, n_mels, n_frames = mel.shape

    for _ in range(num_freq_masks):
        f  = random.randint(1, freq_mask_max)
        f0 = random.randint(0, max(n_mels - f, 1))
        mel[:, f0:f0 + f, :] = 0.0

    for _ in range(num_time_masks):
        t  = random.randint(1, time_mask_max)
        t0 = random.randint(0, max(n_frames - t, 1))
        mel[:, :, t0:t0 + t] = 0.0

    return mel


# ── Batch augmentation ─────────────────────────────────────────────────────────

def mixup_batch(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    alpha: float = 0.3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Mixup (Zhang et al., 2018).
    Blends two random samples and their labels via a Beta(alpha, alpha) weight.
    Returns mixed inputs and soft one-hot targets for use with soft_cross_entropy.

    Why it helps: the model learns smoother decision boundaries and becomes less
    susceptible to overconfident predictions on ambiguous inputs.
    """
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(inputs.size(0), device=inputs.device)

    mixed_inputs = lam * inputs + (1.0 - lam) * inputs[idx]

    a = F.one_hot(targets,       num_classes).float()
    b = F.one_hot(targets[idx],  num_classes).float()
    mixed_targets = lam * a + (1.0 - lam) * b

    return mixed_inputs, mixed_targets
