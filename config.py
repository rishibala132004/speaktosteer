"""
config.py — single source of truth for every tunable value.
"""
from __future__ import annotations
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR      = Path("./data/SpeechCommands/speech_commands_v0.02")
PROCESSED_DIR = Path("./data/processed")
MODEL_DIR     = Path("./models")
MODEL_PATH    = MODEL_DIR / "speak_to_steer_cnn.pth"

# ── Audio ──────────────────────────────────────────────────────────────────────
SAMPLE_RATE = 16000
N_MELS      = 64
N_FFT       = 1024
HOP_LENGTH  = 512

# ── Dataset ────────────────────────────────────────────────────────────────────
COMMAND_CLASSES   = ["forward", "backward", "left", "right", "stop",
                     "_unknown_", "_silence_"]
NUM_CLASSES       = len(COMMAND_CLASSES)
SAMPLES_PER_CLASS = 2000

# ── Training ───────────────────────────────────────────────────────────────────
BATCH_SIZE           = 64
EPOCHS               = 30
LEARNING_RATE        = 1e-3
WEIGHT_DECAY         = 1e-4
EARLY_STOP_PATIENCE  = 6
MIXUP_ALPHA          = 0.3
FOCAL_GAMMA          = 2.0
LABEL_SMOOTHING      = 0.1

# ── Inference Thresholds ───────────────────────────────────────────────────────
#
#  SPEAKER_MATCH_THRESHOLD
#  ────────────────────────
#  After calibration, same-speaker mean similarity with short command clips
#  is typically 0.40–0.65.  A different person scores 0.05–0.20.
#  0.30 sits safely in the gap and is the recommended starting value.
#
#  How to tune for your microphone:
#    • If you see your own commands being REJECTED at 0.30 → lower to 0.25
#    • If strangers' commands are ACCEPTED at 0.30 → raise to 0.35
#
SPEAKER_MATCH_THRESHOLD     = 0.30

#  COMMAND_CONFIDENCE_THRESHOLD
#  ─────────────────────────────
#  CNN softmax probability.  0.75 is a good balance — catches valid commands
#  while still rejecting most noise and unknown words.
#
COMMAND_CONFIDENCE_THRESHOLD = 0.75

# ── Voice Activity Detection ───────────────────────────────────────────────────
VAD_FRAME_MS             = 30
VAD_ENERGY_THRESHOLD     = 0.008   # Raise to 0.015 in noisy rooms
VAD_SPEECH_TRIGGER_FRAMES = 5
VAD_SILENCE_END_FRAMES   = 20
VAD_PRE_SPEECH_FRAMES    = 5
VAD_MIN_SPEECH_DURATION  = 0.3
VAD_MAX_SPEECH_DURATION  = 5.0

# ── Robot ──────────────────────────────────────────────────────────────────────
ROBOT_PORT = '/dev/cu.YOUR_BLUETOOTH_PORT' 
try:
    robot_serial = serial.Serial(ROBOT_PORT, 9600, timeout=1)
    time.sleep(2)
except Exception:
    robot_serial = None
ROBOT_BAUD = 9600