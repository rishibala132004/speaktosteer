"""
process_audio.py — Convert raw Speech Commands WAVs → training tensors.

Run once after downloading the dataset:
    python process_audio.py

Changes from v1:
  • N_MELS raised 40 → 64 (richer frequency resolution)
  • Output stored channel-first (N, 1, N_MELS, T) — no transpose needed in training
  • Resampling support for non-16 kHz files
  • Waveform augmentation applied during processing (doubles usable data)
  • Consistent silence processing (same transform pipeline as commands)
"""
from __future__ import annotations
import glob
import os
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
import torchaudio.functional as AF

from augmentation import augment_waveform
from config import (
    COMMAND_CLASSES, DATA_DIR, HOP_LENGTH, N_FFT, N_MELS,
    PROCESSED_DIR, SAMPLE_RATE, SAMPLES_PER_CLASS,
)

os.makedirs(PROCESSED_DIR, exist_ok=True)

# ── Feature extractor (built once, reused for every file) ─────────────────────
_mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
)
_db_transform = torchaudio.transforms.AmplitudeToDB()


def audio_to_melspec(file_path: str, augment: bool = False) -> np.ndarray | None:
    """
    Load a WAV file and return a log-Mel spectrogram.

    Returns:
        numpy array of shape (1, N_MELS, T) — channel first, ready for PyTorch
        None if the file is unreadable
    """
    try:
        audio, sr = sf.read(file_path, dtype="float32")
    except Exception as exc:
        print(f"  ⚠️  Could not read {file_path}: {exc}")
        return None

    # Stereo → mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Optional waveform-level augmentation (before resampling / padding)
    if augment:
        audio = augment_waveform(audio, p=0.8)

    # Resample if the file's native sample rate differs from target
    if sr != SAMPLE_RATE:
        wav_t = torch.tensor(audio).unsqueeze(0)
        wav_t = AF.resample(wav_t, sr, SAMPLE_RATE)
        audio = wav_t.squeeze(0).numpy()

    # Pad or truncate to exactly 1 second
    target = SAMPLE_RATE
    if len(audio) < target:
        audio = np.pad(audio, (0, target - len(audio)))
    else:
        audio = audio[:target]

    wav_t = torch.tensor(audio).unsqueeze(0)              # (1, 16000)
    mel   = _db_transform(_mel_transform(wav_t))          # (1, N_MELS, T)
    return mel.numpy()                                     # (1, N_MELS, T)


# ── Dataset builder ────────────────────────────────────────────────────────────

def process_dataset() -> None:
    TARGET_COMMANDS = [c for c in COMMAND_CLASSES if not c.startswith("_")]
    label_map       = {word: i for i, word in enumerate(COMMAND_CLASSES)}

    X: list[np.ndarray] = []
    y: list[int]         = []

    # ── 1. Target commands ────────────────────────────────────────────────────
    print("\n── Target commands ─────────────────────────────────")
    for command in TARGET_COMMANDS:
        files = glob.glob(str(DATA_DIR / command / "*.wav"))
        if not files:
            print(f"  ⚠️  No files found for '{command}' — check DATA_DIR in config.py")
            continue
        random.shuffle(files)

        count = 0
        for fpath in files:
            if count >= SAMPLES_PER_CLASS:
                break
            # Original
            spec = audio_to_melspec(fpath, augment=False)
            if spec is not None:
                X.append(spec); y.append(label_map[command]); count += 1
            # Augmented copy (counts toward the cap)
            if count < SAMPLES_PER_CLASS:
                spec_aug = audio_to_melspec(fpath, augment=True)
                if spec_aug is not None:
                    X.append(spec_aug); y.append(label_map[command]); count += 1

        print(f"  {command:12s}: {count:4d} samples")

    # ── 2. Unknown words ──────────────────────────────────────────────────────
    print("\n── Unknown words ───────────────────────────────────")
    excluded = set(TARGET_COMMANDS) | {"_background_noise_"}
    unknown_files: list[str] = []
    for folder in DATA_DIR.iterdir():
        if folder.is_dir() and folder.name not in excluded:
            unknown_files.extend(glob.glob(str(folder / "*.wav")))
    random.shuffle(unknown_files)

    count = 0
    for fpath in unknown_files:
        if count >= SAMPLES_PER_CLASS:
            break
        spec = audio_to_melspec(fpath)
        if spec is not None:
            X.append(spec); y.append(label_map["_unknown_"]); count += 1
    print(f"  _unknown_   : {count:4d} samples")

    # ── 3. Background / silence ───────────────────────────────────────────────
    print("\n── Background noise (silence class) ────────────────")
    noise_files = glob.glob(str(DATA_DIR / "_background_noise_" / "*.wav"))
    count = 0

    for nf in noise_files:
        if count >= SAMPLES_PER_CLASS:
            break
        try:
            audio, sr = sf.read(nf, dtype="float32")
        except Exception:
            continue
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != SAMPLE_RATE:
            t = torch.tensor(audio).unsqueeze(0)
            audio = AF.resample(t, sr, SAMPLE_RATE).squeeze(0).numpy()

        num_chunks = len(audio) // SAMPLE_RATE
        for i in range(num_chunks):
            if count >= SAMPLES_PER_CLASS:
                break
            chunk = torch.tensor(audio[i * SAMPLE_RATE:(i + 1) * SAMPLE_RATE]).unsqueeze(0)
            spec  = _db_transform(_mel_transform(chunk)).numpy()   # (1, N_MELS, T)
            X.append(spec); y.append(label_map["_silence_"]); count += 1
    print(f"  _silence_   : {count:4d} samples")

    # ── Serialise ─────────────────────────────────────────────────────────────
    X_arr = np.array(X, dtype=np.float32)   # (N, 1, N_MELS, T)
    y_arr = np.array(y, dtype=np.int64)

    total = len(y_arr)
    print(f"\n── Summary ─────────────────────────────────────────")
    print(f"  Total samples : {total:,}")
    print(f"  Feature shape : {X_arr.shape}")

    np.save(PROCESSED_DIR / "X_features.npy", X_arr)
    np.save(PROCESSED_DIR / "y_labels.npy",   y_arr)
    print(f"\n  ✅ Saved to {PROCESSED_DIR}\n")


if __name__ == "__main__":
    process_dataset()
