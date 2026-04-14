# 🤖 Speak-to-Steer — Voice-Controlled Robot System

A real-time, speaker-verified speech recognition system that lets authorised users control a robot via voice commands. Built with PyTorch, SpeechBrain (ECAPA-TDNN), OpenAI Whisper, and a custom CNN trained on Google Speech Commands v2.

---

## Table of Contents

- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Architecture Overview](#architecture-overview)
- [Quick Start](#quick-start)
- [File-by-File Reference](#file-by-file-reference)
- [Configuration Guide](#configuration-guide)
- [Threshold Tuning](#threshold-tuning)
- [Enrollment and Calibration Flow](#enrollment-and-calibration-flow)
- [Robot Serial Protocol](#robot-serial-protocol)
- [Raspberry Pi Deployment](#raspberry-pi-deployment)
- [Troubleshooting](#troubleshooting)
- [Known Limitations](#known-limitations)
- [Dependencies](#dependencies)

---

## How It Works

The system has three layers of decision-making on every utterance:

```
Microphone → VAD → [Is this speech?]
                        │
                        ▼
              [Is this "My name is X"?]  ──Yes──▶  Enroll + Calibrate
                        │No
                        ▼
              [Is this an enrolled speaker?]  ──No──▶  REJECT
                  (ECAPA-TDNN cosine sim)
                        │Yes
                        ▼
              [What command was spoken?]  ──low conf──▶  IGNORE
                  (CNN + softmax)
                        │high conf
                        ▼
              Send byte to robot via serial Bluetooth
```

**Enrollment** — Say `"My name is Vaishnav"`. Whisper transcribes it, extracts the name, and stores an ECAPA-TDNN voice embedding.

**Calibration** — Immediately after enrollment the system asks you to say 5 command words (one at a time). These short-clip embeddings are stored alongside the enrollment embedding so that word-to-word comparison is possible at runtime.

**Command recognition** — The CNN classifies 1-second log-Mel spectrograms into 7 classes: `forward`, `backward`, `left`, `right`, `stop`, `_unknown_`, `_silence_`.

**Speaker verification** — Every command utterance is compared against all stored embeddings for each enrolled speaker using mean cosine similarity. Only matches above `SPEAKER_MATCH_THRESHOLD` proceed.

---

## Project Structure

```
speak-to-steer/
│
├── config.py            ← All tunable constants live here. Edit this first.
├── model.py             ← CNN architecture (ResBlock + SE attention)
├── augmentation.py      ← Waveform, SpecAugment, and Mixup helpers
│
├── download_data.py     ← Step 1: Download Speech Commands v2 (~2.8 GB)
├── process_audio.py     ← Step 2: Convert WAVs to log-Mel spectrogram .npy files
├── train.py             ← Step 3: Train the CNN, save best checkpoint
│
├── live_system.py       ← Step 4 (daily): Real-time streaming pipeline
│
├── requirements.txt     ← Python dependencies
└── README.md            ← This file
```

---

## Architecture Overview

### CNN — `model.py`

```
Input (B, 1, 64, 32)  ← log-Mel spectrogram, 64 mel bins, ~32 time frames
       │
    Stem Conv 3×3 → 32 ch
       │
  Stage 1: ResBlock(32→32)  + MaxPool2d   → (B, 32, 32, 16)
  Stage 2: ResBlock(32→64)  + MaxPool2d   → (B, 64, 16,  8)
  Stage 3: ResBlock(64→128) + MaxPool2d   → (B, 128, 8,  4)
  Stage 4: ResBlock(128→256)              → (B, 256, 8,  4)
       │
  AdaptiveAvgPool(2×2) → Flatten
       │
  FC(1024→256) → ReLU → Dropout(0.4)
       │
  FC(256→7)    ← one logit per class
```

Each ResBlock contains: Conv→BN→ReLU→Conv→BN + Squeeze-and-Excitation channel attention + skip connection.

### Speaker Verification — SpeechBrain ECAPA-TDNN

Pre-trained on VoxCeleb. Produces a 192-dimensional speaker embedding. No fine-tuning required. Downloaded automatically on first run (~80 MB).

### Enrollment Transcription — Whisper tiny.en

Used **only** during enrollment to extract a name from `"My name is X"`. Never runs during command recognition. Downloaded automatically on first run (~39 MB).

---

## Quick Start

### Requirements

- Python 3.9 or later
- 4 GB RAM minimum (8 GB recommended for training)
- Microphone
- Bluetooth serial module on robot (HC-05 or similar)

### Installation

```bash
git clone <your-repo-url>
cd speak-to-steer
pip install -r requirements.txt
```

### Run Order

Steps 1–3 are one-time setup. Step 4 is run every session.

```bash
# Step 1 — Download dataset (~2.8 GB, takes 5–15 min)
python download_data.py

# Step 2 — Process audio into spectrograms (~10–20 min)
python process_audio.py

# Step 3 — Train the model (~30–60 min CPU, ~5–10 min GPU)
python train.py

# Step 4 — Run the live system
python live_system.py
```

Before Step 4, set your serial port in `config.py`:

```python
ROBOT_PORT = '/dev/cu.HC-05'   # macOS
ROBOT_PORT = 'COM3'             # Windows
ROBOT_PORT = '/dev/rfcomm0'    # Linux / Raspberry Pi
```

---

## File-by-File Reference

### `config.py`

Single source of truth. Every other module imports constants from here. The only file you should need to edit for deployment.

Key sections: audio parameters, dataset sizes, training hyperparameters, inference thresholds, VAD settings, robot serial port.

### `model.py`

Defines `SpeechCommandCNN`. Can be imported and instantiated anywhere:

```python
from model import SpeechCommandCNN
model = SpeechCommandCNN(num_classes=7, n_mels=64)
```

Run standalone to verify the architecture and parameter count:

```bash
python model.py
# Output: Parameters: 1,234,567
```

### `augmentation.py`

Three augmentation layers used during training. **Do not call these during inference.**

| Function | What it does | When applied |
|---|---|---|
| `augment_waveform()` | Noise / volume / time-shift | `process_audio.py` |
| `spec_augment()` | Frequency + time masking | `train.py` (per batch) |
| `mixup_batch()` | Interpolates two samples | `train.py` (50% of batches) |

### `download_data.py`

Downloads Google Speech Commands v2 via `torchaudio`. Run once. The dataset contains ~105,000 one-second WAV files across 35 word classes.

### `process_audio.py`

Reads WAV files from `DATA_DIR`, converts them to log-Mel spectrograms, applies waveform augmentation (doubles usable samples), and saves `X_features.npy` and `y_labels.npy` to `PROCESSED_DIR`.

Output shape: `(N, 1, N_MELS, T)` — channel-first, ready for PyTorch.

### `train.py`

Full training pipeline. Key improvements over a basic training loop:

- **Focal Loss** (gamma=2.0): downweights easy examples, focuses gradient on hard ones
- **Label smoothing** (0.1): prevents overconfident predictions
- **SpecAugment**: applied per sample during training
- **Mixup**: blends 50% of batches for smoother decision boundaries
- **OneCycleLR**: warm-up + annealing schedule for faster convergence
- **Early stopping**: halts when validation loss stops improving (patience=6)
- **Best checkpoint**: saves the epoch that generalises best, not the last one
- **70/15/15 split**: honest train/val/test evaluation

The trained model is saved to `MODEL_PATH` (default `./models/speak_to_steer_cnn.pth`).

### `live_system.py`

The runtime system. Key components:

**`VAD`** — Energy-based voice activity detector. Accumulates 30ms frames in a state machine (IDLE → SPEAKING → IDLE). Only emits complete utterances to the model pipeline. Prevents silence and background noise from ever reaching inference.

**`AudioProcessor`** — Converts raw numpy arrays to `(1, 1, N_MELS, T)` CNN input tensors. Includes amplitude normalisation and a sliding-window energy search to extract the most informative 1-second chunk from longer clips.

**`SpeakerRegistry`** — Thread-safe dict mapping speaker names to a *list* of ECAPA-TDNN embeddings (1 enrollment sentence + up to 29 calibration/command samples). Similarity is the **mean** across all stored embeddings for stability.

**`Calibration`** — State tracker for the post-enrollment calibration phase. Collects 5 short-clip embeddings from the new speaker so that word-to-word comparison is possible at runtime.

**`SpeakToSteerSystem`** — Top-level orchestrator. Manages the `sounddevice` input stream, a thread-safe audio queue, and a VAD processing loop. Each detected utterance is dispatched to `handle_utterance()` in its own daemon thread.

---

## Configuration Guide

```python
# config.py — most commonly changed values

# Set this before running live_system.py
ROBOT_PORT = '/dev/rfcomm0'         # Your Bluetooth serial port

# Raise if background noise triggers the VAD in your environment
VAD_ENERGY_THRESHOLD = 0.008        # Try 0.015 in noisy rooms

# Speaker verification sensitivity — see Threshold Tuning below
SPEAKER_MATCH_THRESHOLD = 0.30

# Command recognition minimum confidence
COMMAND_CONFIDENCE_THRESHOLD = 0.75

# How many command words to collect during calibration
# (set in live_system.py, not config.py)
CALIBRATION_TARGET = 5              # Increase to 8–10 for higher accuracy
```

---

## Threshold Tuning

The system prints similarity scores and confidence for every utterance. Use them to calibrate.

### `SPEAKER_MATCH_THRESHOLD`

After calibration, check your own scores across several commands:

```
👤  [Vaishnav @ 52%]  →  "forward"  (91%)
👤  [Vaishnav @ 47%]  →  "left"     (88%)
👤  [Vaishnav @ 61%]  →  "stop"     (94%)
```

Then ask someone else to speak and note their scores:

```
🚫  [Vaishnav @ 11%]  →  "forward"  (85%)
```

Set the threshold halfway between the two groups. If your scores cluster at 45–60% and strangers at 8–15%, set it to `0.30`–`0.35`.

| Symptom | Fix |
|---|---|
| Your own commands rejected | Lower threshold (e.g. 0.30 → 0.25) |
| Strangers' commands accepted | Raise threshold (e.g. 0.30 → 0.40) |
| Scores all very low (<30%) | Did calibration complete? Check logs |

### `COMMAND_CONFIDENCE_THRESHOLD`

| Symptom | Fix |
|---|---|
| Valid commands ignored | Lower to 0.65 |
| Wrong commands sent occasionally | Raise to 0.80 |

---

## Enrollment and Calibration Flow

```
You say:  "My name is Vaishnav"
              │
              ▼
         Whisper transcribes → "my name is vaishnav"
         Regex extracts name → "Vaishnav"
         ECAPA-TDNN stores sentence embedding #0
              │
              ▼
         System prints:
         "Say 5 command words, one at a time:
          FORWARD  BACKWARD  LEFT  RIGHT  STOP"
              │
   [Say each word, pause between them]
              │
              ▼
         5 short-clip embeddings stored (#1–#5)
              │
              ▼
         "Calibration complete! Vaishnav is fully authorised."
              │
              ▼
         Commands now matched against mean of all 6 embeddings
```

**Re-enrollment** — Saying `"My name is Vaishnav"` again at any time starts a new calibration cycle and refreshes the voice print. Useful if voice quality changes (cold, different microphone distance).

**Multiple users** — Each person says their own name. The registry grows independently for each speaker. All enrolled speakers are authorised simultaneously.

---

## Robot Serial Protocol

The system sends single ASCII bytes over serial Bluetooth (HC-05 default: 9600 baud):

| Command | Byte sent |
|---|---|
| forward | `F` (0x46) |
| backward | `B` (0x42) |
| left | `L` (0x4C) |
| right | `R` (0x52) |
| stop | `S` (0x53) |

Your robot firmware should read one byte at a time and act on it immediately. Example Arduino sketch:

```cpp
char cmd;
void loop() {
  if (Serial.available()) {
    cmd = Serial.read();
    if      (cmd == 'F') moveForward();
    else if (cmd == 'B') moveBackward();
    else if (cmd == 'L') turnLeft();
    else if (cmd == 'R') turnRight();
    else if (cmd == 'S') stopMotors();
  }
}
```

---

## Raspberry Pi Deployment

See the section below for the full list of changes required.

### Hardware tested on

- Raspberry Pi 4 (4 GB RAM) — recommended minimum
- Raspberry Pi 5 — ideal
- Raspberry Pi 3B+ — works but Whisper enrollment is slow (~8s per name)

### Do NOT run training on the Pi

Training requires `process_audio.py` and `train.py`. These are computationally expensive. Run them on a laptop/desktop, copy the trained `.pth` file to the Pi, and run only `live_system.py` there.

### What to copy to the Pi

```
speak-to-steer/
├── config.py
├── model.py
├── live_system.py
├── requirements_pi.txt      ← use this instead of requirements.txt
└── models/
    └── speak_to_steer_cnn.pth
```

### Pi-specific `requirements_pi.txt`

```
torch==2.1.0               # CPU-only wheel — see install note below
torchaudio==2.1.0
speechbrain>=1.0.0
openai-whisper>=20231117
sounddevice>=0.4.6
soundfile>=0.12.1
numpy>=1.24.0
pyserial>=3.5
```

PyTorch CPU-only wheel for Pi (aarch64):

```bash
pip install torch==2.1.0 torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cpu
```

### System packages needed on Pi OS

```bash
sudo apt update
sudo apt install -y \
    python3-pip \
    portaudio19-dev \          # required by sounddevice
    libsndfile1 \              # required by soundfile
    ffmpeg \                   # required by Whisper
    libblas-dev liblapack-dev  # speeds up numpy on ARM
```

### Changes required in `config.py` for Pi

```python
# 1. Set your serial port (HC-05 over Bluetooth on Pi)
ROBOT_PORT = '/dev/rfcomm0'

# 2. Raise VAD threshold — Pi USB mics often have more background noise
VAD_ENERGY_THRESHOLD = 0.012

# 3. Whisper is slow on Pi CPU — tiny.en is already the smallest model
#    No change needed here; just expect ~6-8 seconds for enrollment
```

### Changes required in `live_system.py` for Pi

In `_load_models()`, change the Whisper device to always use CPU on Pi:

```python
# Replace this line:
self._whisper = whisper.load_model("tiny.en", device=str(self._device))

# With this:
self._whisper = whisper.load_model("tiny.en", device="cpu")
```

Whisper on Pi ARM has MPS/CUDA unavailable; forcing CPU avoids a silent fallback error.

### Pairing HC-05 Bluetooth on Pi

```bash
# Pair the HC-05 module
bluetoothctl
  scan on
  pair XX:XX:XX:XX:XX:XX     # your HC-05 MAC address
  trust XX:XX:XX:XX:XX:XX
  exit

# Bind to a serial port
sudo rfcomm bind /dev/rfcomm0 XX:XX:XX:XX:XX:XX

# Add your user to the dialout group (needed for serial access)
sudo usermod -aG dialout $USER
# Log out and back in for this to take effect
```

### Run on Pi boot (optional)

Create `/etc/systemd/system/speak-to-steer.service`:

```ini
[Unit]
Description=Speak-to-Steer Voice Robot
After=network.target sound.target

[Service]
User=pi
WorkingDirectory=/home/pi/speak-to-steer
ExecStart=/usr/bin/python3 live_system.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Enable it:

```bash
sudo systemctl daemon-reload
sudo systemctl enable speak-to-steer
sudo systemctl start speak-to-steer
```

View logs:

```bash
journalctl -u speak-to-steer -f
```

---

## Troubleshooting

### All commands rejected ("unauthorised speaker")

1. Check that calibration completed — you should have seen `"Calibration complete!"` in the logs.
2. Look at the similarity score in brackets, e.g. `[Vaishnav @ 23%]`. If it is consistently below 30%, lower `SPEAKER_MATCH_THRESHOLD` to 0.20 and test again.
3. Re-enrol. Say `"My name is X"` again — this restarts calibration with fresh embeddings.

### VAD triggers on background noise

Raise `VAD_ENERGY_THRESHOLD` in `config.py` from `0.008` to `0.015` or `0.020`. Print RMS values by adding `print(f"RMS: {rms:.4f}")` inside `VAD.process()` to find the right value for your room.

### Wrong command words recognised

This is a CNN issue, not a speaker verification issue. The model may need retraining with more `SAMPLES_PER_CLASS`. Also check the microphone distance — the CNN was trained on close-mic recordings.

### Whisper misreads the name during enrollment

Speak clearly and include the full phrase `"My name is X"`. Common misreadings: "Vaishnav" → "Wishnove", "Wachnau". This does not affect command recognition at all — only the display name in logs. The voice print is stored correctly regardless.

### `sounddevice` error on Raspberry Pi

```bash
sudo apt install portaudio19-dev
pip install sounddevice --force-reinstall
```

If using a USB microphone, list devices with:

```python
import sounddevice; print(sounddevice.query_devices())
```

Then set the device index in `live_system.py`:

```python
with sd.InputStream(..., device=1):   # set to your USB mic index
```

### Serial port permission denied

```bash
sudo usermod -aG dialout $USER
# Then log out and back in
```

---

## Known Limitations

- **Accent sensitivity** — ECAPA-TDNN handles accents well, but the CNN was trained on English-accented Speech Commands data. Non-native accents may reduce command confidence scores.
- **Simultaneous speakers** — If two people speak at once, VAD captures the blend. The speaker verifier will match neither; the command is rejected. This is correct behaviour.
- **"Stop" vs. background words** — Short words with similar spectrograms (e.g. "top", "drop") may occasionally be misclassified as "stop". Raising `COMMAND_CONFIDENCE_THRESHOLD` to 0.80 mitigates this.
- **Whisper enrollment latency** — On CPU (especially Pi 3/4), enrollment takes 5–10 seconds. This is a one-time cost per session and does not affect command latency.

---

## Dependencies

| Package | Purpose | Version |
|---|---|---|
| `torch` | CNN training and inference | ≥ 2.1.0 |
| `torchaudio` | Audio transforms, dataset download | ≥ 2.1.0 |
| `speechbrain` | ECAPA-TDNN speaker verification | ≥ 1.0.0 |
| `openai-whisper` | Enrollment name transcription | ≥ 20231117 |
| `sounddevice` | Real-time microphone stream | ≥ 0.4.6 |
| `soundfile` | WAV file I/O (bypasses torchaudio loader bugs) | ≥ 0.12.1 |
| `numpy` | Array operations | ≥ 1.24.0 |
| `scikit-learn` | Train/val/test split, classification report | ≥ 1.3.0 |
| `pyserial` | Bluetooth serial communication to robot | ≥ 3.5 |

---

## Licence

MIT — free to use, modify, and distribute with attribution.
