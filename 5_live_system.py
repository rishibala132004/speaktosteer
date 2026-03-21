import torch
import torch.nn as nn
import torchaudio
import numpy as np
import serial
import time
from pathlib import Path

if not hasattr(torchaudio, 'list_audio_backends'):
    torchaudio.list_audio_backends = lambda: ['soundfile']
from speechbrain.inference.speaker import EncoderClassifier

# --- Configuration ---
MODEL_PATH = Path("./models/speak_to_steer_cnn.pth")
ENROLLMENT_FILE = Path("./my_authorized_voice.wav")
TARGET_COMMANDS = ["forward", "backward", "left", "right", "stop", "_unknown_", "_silence_"]
SAMPLE_RATE = 16000

# Back to Strict Security!
SPEAKER_MATCH_THRESHOLD = 0.70
COMMAND_CONFIDENCE_THRESHOLD = 0.80

ROBOT_PORT = '/dev/cu.YOUR_BLUETOOTH_PORT' 
try:
    robot_serial = serial.Serial(ROBOT_PORT, 9600, timeout=1)
    time.sleep(2)
except Exception:
    robot_serial = None

class SpeechCommandCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(SpeechCommandCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 5 * 4, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.fc_layers(self.conv_layers(x))

print("\nLoading ML Models into Memory...")
cnn_model = SpeechCommandCNN(num_classes=7)
cnn_model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
cnn_model.eval()

speaker_model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="tmp_model")

import soundfile as sf
audio_array, _ = sf.read(ENROLLMENT_FILE)
auth_waveform = torch.tensor(audio_array, dtype=torch.float32).unsqueeze(0)
if auth_waveform.shape[0] > 1: auth_waveform = torch.mean(auth_waveform, dim=0, keepdim=True)
AUTHORIZED_VOICE_PRINT = speaker_model.encode_batch(auth_waveform)
print("Models Loaded and Voice Print Secured. System Ready.\n")


def process_live_audio(live_audio_numpy_array):
    start_time = time.time()
    waveform = torch.tensor(live_audio_numpy_array, dtype=torch.float32).unsqueeze(0)
    
    # 1. BIOMETRICS: SpeechBrain gets the full 3 seconds
    incoming_voice_print = speaker_model.encode_batch(waveform)
    similarity_score = torch.nn.functional.cosine_similarity(AUTHORIZED_VOICE_PRINT, incoming_voice_print, dim=2).item()

    # 2. SLIDING WINDOW: Find the loudest 1-second chunk
    if waveform.shape[1] > SAMPLE_RATE:
        # We look for the 1-second block with the most 'energy' (volume)
        window_size = SAMPLE_RATE
        energies = [torch.sum(waveform[:, i:i+window_size]**2) for i in range(0, waveform.shape[1] - window_size, 1000)]
        best_start = np.argmax(energies) * 1000
        cnn_waveform = waveform[:, best_start:best_start+window_size]
    else:
        cnn_waveform = waveform
    # 3. COMMAND RECOGNITION: CNN gets exactly 1 second
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE, n_mels=40, n_fft=1024, hop_length=512)
    amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
    cnn_features = amplitude_to_db(mel_spectrogram(cnn_waveform)).unsqueeze(0)
    
    with torch.no_grad():
        cnn_outputs = cnn_model(cnn_features)
        probabilities = torch.nn.functional.softmax(cnn_outputs, dim=1)[0]
        confidence, predicted_idx = torch.max(probabilities, 0)
        predicted_command = TARGET_COMMANDS[predicted_idx.item()]

    latency_ms = (time.time() - start_time) * 1000
    print(f"[{latency_ms:.0f}ms] Heard: '{predicted_command}' ({confidence:.1%}) | Match: {similarity_score:.1%}")
    
    if similarity_score >= SPEAKER_MATCH_THRESHOLD and confidence.item() >= COMMAND_CONFIDENCE_THRESHOLD:
        command_map = {"forward": b'F', "backward": b'B', "left": b'L', "right": b'R', "stop": b'S'}
        if predicted_command in command_map:
            print(f"SUCCESS -> Sending {command_map[predicted_command]} to Robot.")
            if robot_serial: robot_serial.write(command_map[predicted_command])
        else:
            print("IGNORED -> Noise Detected.")
    else:
        print("REJECTED -> Unauthorized or Unclear.")