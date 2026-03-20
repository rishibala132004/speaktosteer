import os
import glob
import random
import torch
import torchaudio
import numpy as np
import soundfile as sf  # <--- Bypassing torchaudio's broken loader
from pathlib import Path

# --- Configuration ---
DATA_DIR = Path("./data/SpeechCommands/speech_commands_v0.02")
PROCESSED_DIR = Path("./data/processed")
TARGET_COMMANDS = ["forward", "backward", "left", "right", "stop"]
SAMPLES_PER_CLASS = 1000 # Number of audio files to grab per word
SAMPLE_RATE = 16000      # 16 kHz

# Create the folder to save our spectrograms
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Define the audio-to-image converter (Log-Mel Spectrogram)
mel_spectrogram = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_mels=40,            # 40 frequency bands
    n_fft=1024,           # Window size
    hop_length=512        # Shift size
)
amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

def audio_to_melspec(file_path):
    """Loads a .wav file, ensures it's exactly 1 second, and converts to Mel Spec."""
    # Read directly with soundfile, then convert to PyTorch tensor
    audio_array, sr = sf.read(file_path)
    waveform = torch.tensor(audio_array, dtype=torch.float32).unsqueeze(0)
    
    # If audio is stereo, convert to mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        
    # Pad or truncate to exactly 1 second (16000 samples)
    if waveform.shape[1] < SAMPLE_RATE:
        padding = SAMPLE_RATE - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, padding))
    elif waveform.shape[1] > SAMPLE_RATE:
        waveform = waveform[:, :SAMPLE_RATE]
        
    # Convert to Mel Spectrogram
    mel_spec = mel_spectrogram(waveform)
    log_mel_spec = amplitude_to_db(mel_spec)
    
    return log_mel_spec.squeeze(0).numpy()

def process_dataset():
    print("Starting audio processing...")
    X = [] # Features (Spectrograms)
    y = [] # Labels (0 to 6)
    
    label_map = {word: i for i, word in enumerate(TARGET_COMMANDS)}
    label_map["_unknown_"] = 5
    label_map["_silence_"] = 6
    
    # 1. Process Target Commands
    for command in TARGET_COMMANDS:
        print(f"Processing '{command}'...")
        files = glob.glob(str(DATA_DIR / command / "*.wav"))
        random.shuffle(files)
        
        for file in files[:SAMPLES_PER_CLASS]:
            spec = audio_to_melspec(file)
            X.append(spec)
            y.append(label_map[command])

    # 2. Process 'Unknown' (Words we want to ignore)
    print("Processing '_unknown_' (random words)...")
    unknown_files = []
    all_folders = [f.name for f in DATA_DIR.iterdir() if f.is_dir()]
    for folder in all_folders:
        if folder not in TARGET_COMMANDS and folder != "_background_noise_":
            unknown_files.extend(glob.glob(str(DATA_DIR / folder / "*.wav")))
            
    random.shuffle(unknown_files)
    for file in unknown_files[:SAMPLES_PER_CLASS]:
        spec = audio_to_melspec(file)
        X.append(spec)
        y.append(label_map["_unknown_"])

    # 3. Process 'Silence' (Background Noise)
    print("Processing '_silence_' (background noise)...")
    noise_files = glob.glob(str(DATA_DIR / "_background_noise_" / "*.wav"))
    silence_count = 0
    
    for file in noise_files:
        if silence_count >= SAMPLES_PER_CLASS:
            break
            
        # Read directly with soundfile
        audio_array, sr = sf.read(file)
        waveform = torch.tensor(audio_array, dtype=torch.float32).unsqueeze(0)
        
        # Noise files are long, so chop them into 1-second chunks
        num_chunks = waveform.shape[1] // SAMPLE_RATE
        for i in range(num_chunks):
            if silence_count >= SAMPLES_PER_CLASS:
                break
            chunk = waveform[:, i*SAMPLE_RATE : (i+1)*SAMPLE_RATE]
            spec = amplitude_to_db(mel_spectrogram(chunk))
            X.append(spec.squeeze(0).numpy())
            y.append(label_map["_silence_"])
            silence_count += 1

    # Convert to massive NumPy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Add a "channel" dimension for the CNN (Num_Samples, Height, Width, Channels)
    X = np.expand_dims(X, axis=-1)
    
    print(f"\nFinal Dataset Shape: {X.shape}")
    print(f"Total Labels: {y.shape}")
    
    print("Saving processed tensors to disk...")
    np.save(PROCESSED_DIR / "X_features.npy", X)
    np.save(PROCESSED_DIR / "y_labels.npy", y)
    print("Done! Ready for training.")

if __name__ == "__main__":
    process_dataset()