import torchaudio
import os

def setup_dataset():
    # Create a folder called 'data' in your project directory
    data_dir = "./data"
    os.makedirs(data_dir, exist_ok=True)

    print("Connecting to Google servers...")
    print("Downloading the Speech Commands v2 Dataset (This is about 2.8 GB, it may take a while depending on your Wi-Fi)...")
    
    # Download the dataset
    dataset = torchaudio.datasets.SPEECHCOMMANDS(
        root=data_dir,
        url='speech_commands_v0.02', # This is the v2 dataset
        folder_in_archive='SpeechCommands',
        download=True
    )
    
    print(f"\nSuccess! Download complete.")
    print(f"The dataset contains a total of {len(dataset)} audio files.")

if __name__ == "__main__":
    setup_dataset()