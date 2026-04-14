import torch
import torch.nn as nn
import torchaudio
import numpy as np
import soundfile as sf

# --- MONKEY PATCH FOR SPEECHBRAIN ---
# PyTorch removed this function, but SpeechBrain still looks for it.
if not hasattr(torchaudio, 'list_audio_backends'):
    torchaudio.list_audio_backends = lambda: ['soundfile']

# Now it is safe to import SpeechBrain!
from speechbrain.inference.speaker import EncoderClassifier
from pathlib import Path

# --- Configuration ---
MODEL_PATH = Path("./models/speak_to_steer_cnn.pth")
TARGET_COMMANDS = ["forward", "backward", "left", "right", "stop", "_unknown_", "_silence_"]
SAMPLE_RATE = 16000

# Thresholds for the robot to actually move
SPEAKER_MATCH_THRESHOLD = 0.70  # 70% voice similarity required
COMMAND_CONFIDENCE_THRESHOLD = 0.80 # 80% confident it heard the right word

# --- 1. Rebuild the CNN Architecture ---
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

# --- 2. Audio Processing Helper Functions ---
def load_audio_safe(file_path):
    """Safely loads audio using soundfile to bypass torchaudio bugs."""
    audio_array, sr = sf.read(file_path)
    waveform = torch.tensor(audio_array, dtype=torch.float32).unsqueeze(0)
    if waveform.shape[0] > 1: # Convert stereo to mono
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    return waveform

def extract_cnn_features(waveform):
    """Converts waveform to Mel Spectrogram for the CNN."""
    if waveform.shape[1] < SAMPLE_RATE:
        waveform = torch.nn.functional.pad(waveform, (0, SAMPLE_RATE - waveform.shape[1]))
    elif waveform.shape[1] > SAMPLE_RATE:
        waveform = waveform[:, :SAMPLE_RATE]
        
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE, n_mels=40, n_fft=1024, hop_length=512)
    amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
    
    mel_spec = amplitude_to_db(mel_spectrogram(waveform))
    return mel_spec.unsqueeze(0) # Shape: (1, 1, 40, 32)

# --- 3. Main Inference Function ---
def run_system(enrollment_file, incoming_command_file):
    print("\n--- Initializing Speak-to-Steer System ---")
    
    # Load CNN
    print("1. Loading Command Recognition CNN...")
    cnn_model = SpeechCommandCNN(num_classes=7)
    cnn_model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    cnn_model.eval()

    # Load SpeechBrain (Downloads automatically the first time)
    print("2. Loading Speaker Verification Model (SpeechBrain)...")
    speaker_model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": "cpu"})

    print("\n--- Processing Audio ---")
    # Step A: Create the "Voice Print" of the authorized user
    enrollment_wav = load_audio_safe(enrollment_file)
    authorized_voice_print = speaker_model.encode_batch(enrollment_wav)

    # Step B: Process the incoming live command
    test_wav = load_audio_safe(incoming_command_file)
    incoming_voice_print = speaker_model.encode_batch(test_wav)
    
    # Step C: Ask the CNN what word was spoken
    cnn_features = extract_cnn_features(test_wav)
    with torch.no_grad():
        cnn_outputs = cnn_model(cnn_features)
        probabilities = torch.nn.functional.softmax(cnn_outputs, dim=1)[0]
        confidence, predicted_idx = torch.max(probabilities, 0)
        predicted_command = TARGET_COMMANDS[predicted_idx.item()]

    # Step D: Ask SpeechBrain if the voices match
    similarity_score = torch.nn.functional.cosine_similarity(authorized_voice_print, incoming_voice_print, dim=2).item()

    # --- 4. Final Decision Logic ---
    print("\n" + "="*40)
    print("SYSTEM RESULTS:")
    print("="*40)
    print(f"Heard Command  : '{predicted_command.upper()}' (Confidence: {confidence.item():.1%})")
    print(f"Voice Match    : {similarity_score:.1%} Similar to Authorized User")
    print("-" * 40)
    
    if similarity_score >= SPEAKER_MATCH_THRESHOLD and confidence.item() >= COMMAND_CONFIDENCE_THRESHOLD:
        if predicted_command in ["_unknown_", "_silence_"]:
            print("ACTION: IGNORED (Noise or Unknown Word Detected)")
        else:
            print(f"ACTION: SUCCESS! Sending '{predicted_command.upper()}' to Robot.")
    elif similarity_score < SPEAKER_MATCH_THRESHOLD:
        print("ACTION: REJECTED (Unauthorized Speaker Detected)")
    else:
        print("ACTION: REJECTED (Command Not Understood Clearly)")
    print("="*40 + "\n")

if __name__ == "__main__":
    base_dir = Path("./data/SpeechCommands/speech_commands_v0.02")
    
    # File 1: The "Authorized" Voice 
    enrollment_audio = base_dir / "forward" / "0a2b400e_nohash_0.wav" 
    
    # File 2: The "Live" Command (Same exact file to test a 100% match)
    test_audio = base_dir / "forward" / "0a2b400e_nohash_0.wav" 
    
    if not enrollment_audio.exists():
        print("Error: Could not find the test audio files. Make sure the paths are correct.")
    else:
        run_system(enrollment_audio, test_audio)