"""
live_system.py — Real-Time Speech Robot Controller (v2).

HOW IT WORKS
────────────
Enrollment  : Say "My name is <name>" out loud.
              Whisper transcribes → name extracted → voice print stored.
              Anyone can enroll; all enrolled speakers are authorised.

Commands    : Say FORWARD / BACKWARD / LEFT / RIGHT / STOP.
              System checks:
                1. Is this an enrolled speaker?  (ECAPA-TDNN cosine similarity)
                2. What command did they say?     (CNN + softmax confidence)
              Only authorised speakers' commands are forwarded to the robot.

Re-enrolment: Saying "My name is X" again updates X's voice print.

Unknown voice: Commands from unrecognised speakers are rejected with a log.

Edge cases handled
──────────────────
• Silence / background noise → VAD prevents them from reaching the models
• Multiple speakers           → registry stores all enrolled speakers; best match wins
• Low microphone volume       → amplitude normalisation before every inference
• Accent variation            → ECAPA-TDNN is accent-agnostic; no user-specific training needed
• Noisy environment           → raise VAD_ENERGY_THRESHOLD in config.py
• Whisper slow on CPU         → enrollment is one-shot; commands never touch Whisper

Usage
──────
    python live_system.py

Set ROBOT_PORT in config.py before running.
"""
from __future__ import annotations

import re
import threading
import time
import queue
from collections import deque
from enum import Enum, auto

import numpy as np
import torch
import torchaudio
import sounddevice as sd
import serial
import whisper

if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]
from speechbrain.inference.speaker import EncoderClassifier

from config import (
    COMMAND_CLASSES, COMMAND_CONFIDENCE_THRESHOLD, HOP_LENGTH,
    MODEL_PATH, N_FFT, N_MELS, NUM_CLASSES, ROBOT_BAUD, ROBOT_PORT,
    SAMPLE_RATE, SPEAKER_MATCH_THRESHOLD,
    VAD_ENERGY_THRESHOLD, VAD_FRAME_MS, VAD_MAX_SPEECH_DURATION,
    VAD_MIN_SPEECH_DURATION, VAD_PRE_SPEECH_FRAMES,
    VAD_SILENCE_END_FRAMES, VAD_SPEECH_TRIGGER_FRAMES,
)
from model import SpeechCommandCNN

# How many command words to collect during calibration
CALIBRATION_TARGET = 5


# ─────────────────────────────────────────────────────────────────────────────
#  System States
# ─────────────────────────────────────────────────────────────────────────────

class State(Enum):
    WAITING_ENROLL  = auto()   # No one enrolled yet
    CALIBRATING     = auto()   # Enrolled, collecting command-clip embeddings
    ACTIVE          = auto()   # Fully calibrated, accepting commands


# ─────────────────────────────────────────────────────────────────────────────
#  Voice Activity Detector
# ─────────────────────────────────────────────────────────────────────────────

class VAD:
    def __init__(self) -> None:
        self._frame_sz    = int(SAMPLE_RATE * VAD_FRAME_MS / 1000)
        self._pre_buf     = deque(maxlen=VAD_PRE_SPEECH_FRAMES)
        self._speech_buf : list[np.ndarray] = []
        self._speech_cnt  = 0
        self._silence_cnt = 0
        self._speaking    = False
        self._max_samples = int(SAMPLE_RATE * VAD_MAX_SPEECH_DURATION)

    @property
    def frame_size(self) -> int:
        return self._frame_sz

    def process(self, frame: np.ndarray) -> np.ndarray | None:
        rms       = float(np.sqrt(np.mean(frame ** 2) + 1e-9))
        is_speech = rms > VAD_ENERGY_THRESHOLD

        if not self._speaking:
            self._pre_buf.append(frame.copy())
            if is_speech:
                self._speech_cnt += 1
                if self._speech_cnt >= VAD_SPEECH_TRIGGER_FRAMES:
                    self._speaking    = True
                    self._speech_buf  = list(self._pre_buf)
                    self._silence_cnt = 0
            else:
                self._speech_cnt = max(0, self._speech_cnt - 1)
        else:
            self._speech_buf.append(frame.copy())
            self._silence_cnt = 0 if is_speech else self._silence_cnt + 1
            total = sum(f.shape[0] for f in self._speech_buf)
            if self._silence_cnt >= VAD_SILENCE_END_FRAMES or total >= self._max_samples:
                utterance = np.concatenate(self._speech_buf).flatten()
                self._reset()
                if len(utterance) >= int(SAMPLE_RATE * VAD_MIN_SPEECH_DURATION):
                    return utterance
        return None

    def _reset(self) -> None:
        self._speech_buf  = []
        self._speech_cnt  = 0
        self._silence_cnt = 0
        self._speaking    = False


# ─────────────────────────────────────────────────────────────────────────────
#  Audio Feature Processor  (for CNN)
# ─────────────────────────────────────────────────────────────────────────────

class AudioProcessor:
    def __init__(self) -> None:
        self._mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH,
        )
        self._db = torchaudio.transforms.AmplitudeToDB()

    def to_tensor(self, audio: np.ndarray) -> torch.Tensor:
        wv  = torch.tensor(audio.flatten(), dtype=torch.float32).unsqueeze(0)
        wv  = self._normalise(wv)
        wv  = self._pad_or_trim(wv)
        mel = self._db(self._mel(wv))
        return mel.unsqueeze(0)             # (1, 1, N_MELS, T)

    @staticmethod
    def _normalise(wv: torch.Tensor) -> torch.Tensor:
        return wv / (wv.abs().max() + 1e-8)

    def _pad_or_trim(self, wv: torch.Tensor) -> torch.Tensor:
        n = wv.shape[1]
        if n < SAMPLE_RATE:
            return torch.nn.functional.pad(wv, (0, SAMPLE_RATE - n))
        if n > SAMPLE_RATE:
            return self._best_chunk(wv)
        return wv

    @staticmethod
    def _best_chunk(wv: torch.Tensor) -> torch.Tensor:
        step, win = 512, SAMPLE_RATE
        energies  = [
            torch.sum(wv[:, i:i + win] ** 2).item()
            for i in range(0, wv.shape[1] - win, step)
        ]
        start = int(np.argmax(energies)) * step
        return wv[:, start:start + win]


# ─────────────────────────────────────────────────────────────────────────────
#  Speaker Registry  — stores a LIST of embeddings per speaker
# ─────────────────────────────────────────────────────────────────────────────

class SpeakerRegistry:
    """
    Each speaker is represented by a LIST of embeddings:
        [0]        → the enrollment sentence embedding
        [1 … N]    → short command-word embeddings from calibration

    Similarity is the MEAN cosine score across all stored embeddings.
    This is far more stable than using a single embedding when the
    enrollment and verification clips have different durations.
    """

    MAX_EMBEDDINGS_PER_SPEAKER = 30

    def __init__(self) -> None:
        self._db  : dict[str, list[torch.Tensor]] = {}
        self._lock = threading.Lock()

    # ── write ops ─────────────────────────────────────────────────────────────

    def enroll(self, name: str, embedding: torch.Tensor) -> None:
        """Store the initial enrollment embedding (sentence-length)."""
        with self._lock:
            self._db[name] = [embedding.detach().clone()]

    def add_sample(self, name: str, embedding: torch.Tensor) -> int:
        """
        Add a calibration / update embedding for an existing speaker.
        Returns the new total number of stored embeddings for that speaker.
        """
        with self._lock:
            if name not in self._db:
                return 0
            self._db[name].append(embedding.detach().clone())
            # Cap size: keep first (sentence) + most recent short-clips
            if len(self._db[name]) > self.MAX_EMBEDDINGS_PER_SPEAKER:
                self._db[name] = (
                    self._db[name][:1] +
                    self._db[name][-(self.MAX_EMBEDDINGS_PER_SPEAKER - 1):]
                )
            return len(self._db[name])

    # ── read ops ──────────────────────────────────────────────────────────────

    def match(self, embedding: torch.Tensor) -> tuple[str | None, float]:
        """
        Return (best_speaker_name, mean_cosine_similarity).
        Uses mean across ALL stored embeddings to smooth out duration mismatch noise.
        """
        with self._lock:
            if not self._db:
                return None, 0.0
            best_name, best_score = None, -1.0
            for name, refs in self._db.items():
                scores = [
                    float(
                        torch.nn.functional.cosine_similarity(
                            embedding, ref, dim=2
                        ).item()
                    )
                    for ref in refs
                ]
                # Mean of all stored embeddings — much more stable than max
                score = float(np.mean(scores))
                if score > best_score:
                    best_score = score
                    best_name  = name
            return best_name, best_score

    @property
    def is_empty(self) -> bool:
        with self._lock:
            return not self._db

    @property
    def names(self) -> list[str]:
        with self._lock:
            return list(self._db.keys())

    def embedding_count(self, name: str) -> int:
        with self._lock:
            return len(self._db.get(name, []))


# ─────────────────────────────────────────────────────────────────────────────
#  Robot Interface
# ─────────────────────────────────────────────────────────────────────────────

_COMMAND_BYTES: dict[str, bytes] = {
    "forward":  b"F",
    "backward": b"B",
    "left":     b"L",
    "right":    b"R",
    "stop":     b"S",
}


class RobotInterface:
    def __init__(self) -> None:
        self._serial: serial.Serial | None = None
        if ROBOT_PORT:
            try:
                self._serial = serial.Serial(ROBOT_PORT, ROBOT_BAUD, timeout=1)
                time.sleep(2)
                print(f"   Robot connected → {ROBOT_PORT}")
            except serial.SerialException as exc:
                print(f"   ⚠️  Robot serial failed ({exc}) — dry-run mode.")

    def send(self, command: str) -> None:
        b = _COMMAND_BYTES.get(command)
        if not b:
            return
        if self._serial:
            try:
                self._serial.write(b)
            except serial.SerialException as exc:
                print(f"   ⚠️  Serial write error: {exc}")
        else:
            print(f"   [dry-run] Would send: {b}")


# ─────────────────────────────────────────────────────────────────────────────
#  Calibration tracker
# ─────────────────────────────────────────────────────────────────────────────

class Calibration:
    """Tracks calibration progress for a just-enrolled speaker."""

    def __init__(self) -> None:
        self.speaker : str | None = None
        self.count   : int        = 0
        self.active  : bool       = False

    def start(self, name: str) -> None:
        self.speaker = name
        self.count   = 0
        self.active  = True
        print(f"\n{'─'*55}")
        print(f"  🎯  CALIBRATION  for  {name}")
        print(f"      Say {CALIBRATION_TARGET} command words, one at a time:")
        print(f"      FORWARD  BACKWARD  LEFT  RIGHT  STOP")
        print(f"{'─'*55}\n")

    def record(self) -> bool:
        """Tick one calibration sample. Returns True when calibration is done."""
        self.count += 1
        remaining = CALIBRATION_TARGET - self.count
        if remaining > 0:
            print(f"   ✓  {self.count}/{CALIBRATION_TARGET} recorded  "
                  f"({remaining} more needed)")
            return False
        else:
            print(f"   ✓  {self.count}/{CALIBRATION_TARGET} recorded")
            print(f"\n   ✅  Calibration complete!  "
                  f"{self.speaker} is fully authorised.\n")
            self.active  = False
            self.speaker = None
            self.count   = 0
            return True


# ─────────────────────────────────────────────────────────────────────────────
#  Main System
# ─────────────────────────────────────────────────────────────────────────────

class SpeakToSteerSystem:

    _ENROLL_RE = re.compile(r"my\s+name\s+is\s+(\w+)", re.IGNORECASE)

    def __init__(self) -> None:
        if torch.cuda.is_available():
            self._device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self._device = torch.device("mps")
        else:
            self._device = torch.device("cpu")

        print(f"\n{'='*58}")
        print(f"  SPEAK-TO-STEER  v3  (device: {self._device})")
        print(f"{'='*58}\n")

        self._proc        = AudioProcessor()
        self._registry    = SpeakerRegistry()
        self._robot       = RobotInterface()
        self._vad         = VAD()
        self._calibration = Calibration()
        self._state       = State.WAITING_ENROLL
        self._queue : queue.Queue[np.ndarray] = queue.Queue(maxsize=200)
        self._running     = False
        self._model_lock  = threading.Lock()

        self._load_models()

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_models(self) -> None:
        print("Loading models …")

        print("  [1/3] Command CNN")
        self._cnn = SpeechCommandCNN(num_classes=NUM_CLASSES, n_mels=N_MELS).to(self._device)
        state = torch.load(MODEL_PATH, map_location=self._device, weights_only=True)
        self._cnn.load_state_dict(state)
        self._cnn.eval()

        print("  [2/3] Speaker verification (ECAPA-TDNN)")
        self._spk = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="tmp_model",
            run_opts={"device": str(self._device)},
        )

        print("  [3/3] Whisper tiny.en")
        self._whisper = whisper.load_model("tiny.en", device="cpu")

        print("  All models ready.\n")

    # ── Inference helpers ─────────────────────────────────────────────────────

    def _get_embedding(self, audio: np.ndarray) -> torch.Tensor:
        """
        Produce an ECAPA-TDNN embedding.

        Key fix: pad short clips with REFLECTION padding instead of tiling.
        Reflection sounds acoustically natural (no sudden jumps at boundaries)
        and avoids the periodicity artefacts that tiling introduces.
        Always extend to at least 1.5 s so the stats-pooling layer has
        enough frames to compute reliable mean/std.
        """
        flat   = audio.flatten().astype(np.float32)
        target = int(SAMPLE_RATE * 1.5)          # 1.5 s minimum

        if len(flat) < target:
            # reflect padding: time-reverses the signal at each boundary
            pad_needed = target - len(flat)
            flat = np.pad(flat, (0, pad_needed), mode="reflect")

        wv = torch.tensor(flat, dtype=torch.float32).unsqueeze(0)
        with self._model_lock, torch.no_grad():
            return self._spk.encode_batch(wv)    # (1, 1, 192)

    def _transcribe(self, audio: np.ndarray) -> str:
        fp = audio.flatten().astype(np.float32)
        with self._model_lock:
            result = self._whisper.transcribe(
                fp,
                fp16=(self._device.type == "cuda"),
                language="en",
            )
        return result["text"].strip().lower()

    def _classify_command(self, audio: np.ndarray) -> tuple[str, float]:
        feat = self._proc.to_tensor(audio).to(self._device)
        with self._model_lock, torch.no_grad():
            logits = self._cnn(feat)
            probs  = torch.softmax(logits, dim=1)[0]
            conf, idx = torch.max(probs, 0)
        return COMMAND_CLASSES[idx.item()], float(conf.item())

    # ── Core decision logic ────────────────────────────────────────────────────

    def handle_utterance(self, audio: np.ndarray) -> None:
        t0        = time.perf_counter()
        duration  = len(audio) / SAMPLE_RATE
        embedding = self._get_embedding(audio)

        # ── STATE: WAITING FOR ENROLLMENT ─────────────────────────────────────
        if self._state == State.WAITING_ENROLL:
            # Always transcribe — we need a name
            text = self._transcribe(audio)
            hit  = self._ENROLL_RE.search(text)
            if hit:
                name = hit.group(1).capitalize()
                self._registry.enroll(name, embedding)
                self._calibration.start(name)
                self._state = State.CALIBRATING
                elapsed = (time.perf_counter() - t0) * 1000
                print(f'✅  ENROLLED  "{name}"  (from: "{text}")  [{elapsed:.0f} ms]')
            else:
                print(f'⏳  Say "My name is <your name>" to enrol.  (heard: "{text}")')
            return

        # ── STATE: CALIBRATING ────────────────────────────────────────────────
        if self._state == State.CALIBRATING:
            name = self._calibration.speaker

            # During calibration accept any utterance as a voice sample.
            # Run the CNN anyway so the user sees their words being recognised.
            command, confidence = self._classify_command(audio)
            elapsed = (time.perf_counter() - t0) * 1000
            print(f"   [{elapsed:.0f} ms]  heard: \"{command}\" ({confidence:.0%})", end="  ")

            self._registry.add_sample(name, embedding)
            done = self._calibration.record()

            if done:
                self._state = State.ACTIVE
                # Show a brief score check so user can see calibration worked
                _, score = self._registry.match(embedding)
                print(f"   🔍  Post-calibration self-match: {score:.0%}  "
                      f"(threshold is {SPEAKER_MATCH_THRESHOLD:.0%})")
            return

        # ── STATE: ACTIVE — process commands ──────────────────────────────────
        matched, score = self._registry.match(embedding)
        is_authorised  = score >= SPEAKER_MATCH_THRESHOLD

        # Still watch for new enrollments (long utterances only)
        if duration >= 1.5:
            text = self._transcribe(audio)
            hit  = self._ENROLL_RE.search(text)
            if hit:
                name = hit.group(1).capitalize()
                self._registry.enroll(name, embedding)
                self._calibration.start(name)
                self._state = State.CALIBRATING
                elapsed = (time.perf_counter() - t0) * 1000
                print(f'✅  ENROLLED  "{name}"  [{elapsed:.0f} ms]')
                return

        command, confidence = self._classify_command(audio)
        elapsed = (time.perf_counter() - t0) * 1000

        auth_icon = "👤" if is_authorised else "🚫"
        n_samples = self._registry.embedding_count(matched) if matched else 0
        print(
            f"{auth_icon}  [{matched or '?'} @ {score:.0%} ({n_samples} samples)]"
            f"  →  \"{command}\"  ({confidence:.0%})"
            f"  [{elapsed:.0f} ms]"
        )

        if not is_authorised:
            print("     REJECTED — unauthorised speaker\n")
            return

        if command in ("_unknown_", "_silence_"):
            print("     IGNORED — noise or unknown word\n")
            return

        if confidence >= COMMAND_CONFIDENCE_THRESHOLD:
            print(f"     ✅  SENDING  {command.upper()}\n")
            self._robot.send(command)
        else:
            print(f"     LOW CONFIDENCE ({confidence:.0%}) — ignored\n")

    # ── Audio streaming ────────────────────────────────────────────────────────

    def _audio_callback(
        self, indata: np.ndarray, frames: int, time_info, status
    ) -> None:
        if status:
            print(f"⚠️  Audio status: {status}")
        try:
            self._queue.put_nowait(indata.copy())
        except queue.Full:
            pass

    def _processing_loop(self) -> None:
        remainder = np.array([], dtype=np.float32)
        fs        = self._vad.frame_size

        while self._running:
            try:
                chunk = self._queue.get(timeout=0.3)
            except queue.Empty:
                continue

            data = np.append(remainder, chunk.flatten())

            while len(data) >= fs:
                utterance = self._vad.process(data[:fs])
                data      = data[fs:]
                if utterance is not None:
                    threading.Thread(
                        target=self.handle_utterance,
                        args=(utterance,),
                        daemon=True,
                    ).start()

            remainder = data

    # ── Entry point ────────────────────────────────────────────────────────────

    def run(self) -> None:
        print('📣  Say "My name is <your name>" to enrol.')
        print("    You will then be asked to say 5 command words for calibration.")
        print("    After that, commands are live.  Ctrl-C to quit.\n")

        self._running = True
        proc_thread = threading.Thread(target=self._processing_loop, daemon=True)
        proc_thread.start()

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=self._vad.frame_size,
            callback=self._audio_callback,
        ):
            try:
                while True:
                    time.sleep(0.05)
            except KeyboardInterrupt:
                print("\n\n👋  System stopped.")
                self._running = False


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    system = SpeakToSteerSystem()
    system.run()
