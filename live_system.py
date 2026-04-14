import torch
import torch.nn as nn
import torchaudio
import numpy as np
import serial
import time
from pathlib import Path

# --- MONKEY PATCH FOR SPEECHBRAIN ---
if not hasattr(torchaudio, 'list_audio_backends'):
    torchaudio.list_audio_backends = lambda: ['soundfile']
from speechbrain.inference.speaker import EncoderClassifier

# --- Configuration ---
MODEL_PATH = Path("./models/speak_to_steer_cnn.pth")
ENROLLMENT_FILE = Path("./my_authorized_voice.wav")
TARGET_COMMANDS = ["forward", "backward", "left", "right", "stop", "_unknown_", "_silence_"]
SAMPLE_RATE = 16000

# Thresholds
SPEAKER_MATCH_THRESHOLD = 0.70
COMMAND_CONFIDENCE_THRESHOLD = 0.80

# --- Serial Connection to SyncGear Robot ---
# Mac typically looks like '/dev/cu.usbserial-110' or '/dev/cu.HC-05' for Bluetooth
ROBOT_PORT = '/dev/cu.YOUR_BLUETOOTH_PORT' 
try:
    print(f"Connecting to Robot on {ROBOT_PORT}...")
    robot_serial = serial.Serial(ROBOT_PORT, 9600, timeout=1)
    time.sleep(2) # Give Arduino time to reset upon connection
    print("Robot Connected Successfully!")
except Exception as e:
    print(f"WARNING: Could not connect to robot. Running in dry-run mode. Error: {e}")
    robot_serial = None

# --- Rebuild CNN ---
class SpeechCommandCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(SpeechCommandCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 5 * 4, 128), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        x = self.conv_layers(x)
        return self.fc_layers(x)

# --- Global Model Initialization ---
print("\nLoading ML Models into Memory...")
cnn_model = SpeechCommandCNN(num_classes=7)
cnn_model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
cnn_model.eval()

speaker_model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": "cpu"})

# Generate Authorized Voice Print once at startup
import soundfile as sf
audio_array, _ = sf.read(ENROLLMENT_FILE)
auth_waveform = torch.tensor(audio_array, dtype=torch.float32).unsqueeze(0)
if auth_waveform.shape[0] > 1: auth_waveform = torch.mean(auth_waveform, dim=0, keepdim=True)
AUTH_PRINT = speaker_model.encode_batch(auth_waveform)
print("Models Loaded and Voice Print Secured. System Ready.\n")


# --- THE REAL-TIME INFERENCE FUNCTION ---
#  BLE team members will call this function and pass the live audio array
def process_live_audio(live_audio_np):
    waveform = torch.tensor(live_audio_np, dtype=torch.float32).unsqueeze(0)
    
    # 1. BIOMETRICS (Full Clip)
    incoming_print = speaker_model.encode_batch(waveform)
    match_score = torch.nn.functional.cosine_similarity(AUTH_PRINT, incoming_print, dim=2).item()

    # 2. PRECISION SLIDING WINDOW (Find the loudest 1.0s)
    window_len = 16000 
    if waveform.shape[1] > window_len:
        energies = [torch.sum(waveform[:, i:i+window_len]**2) for i in range(0, waveform.shape[1] - window_len, 4000)]
        best_start = np.argmax(energies) * 4000
        cnn_waveform = waveform[:, best_start:best_start+window_len]
    else:
        cnn_waveform = torch.nn.functional.pad(waveform, (0, window_len - waveform.shape[1]))

    # 3. SPECTROGRAM (Must be exactly 40x32)
    # Note: hop_length=512 on 16000 samples gives exactly 32 time steps
    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, 
        n_mels=40, 
        n_fft=1024, 
        hop_length=512
    )(cnn_waveform)
    
    db_spec = torchaudio.transforms.AmplitudeToDB()(mel_spec).unsqueeze(0)
    
    # Check shape: should be [1, 1, 40, 32]
    # If it's [1, 1, 40, 31], we pad the last pixel
    if db_spec.shape[3] < 32:
        db_spec = torch.nn.functional.pad(db_spec, (0, 32 - db_spec.shape[3]))
    elif db_spec.shape[3] > 32:
        db_spec = db_spec[:, :, :, :32]

    with torch.no_grad():
        output = cnn_model(db_spec)
        conf, idx = torch.max(torch.nn.functional.softmax(output, dim=1)[0], 0)
        cmd = TARGET_COMMANDS[idx.item()]

    print(f"[{int(match_score*100)}% Match] Heard: '{cmd}' ({conf:.1%})")

    if match_score >= 0.40 and conf >= 0.75:
        print(f"✅ SUCCESS -> Sending {cmd}")
    else:
        print("❌ REJECTED")