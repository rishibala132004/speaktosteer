"""
download_data.py — Download the Google Speech Commands v2 dataset (~2.8 GB).

Run this first, before process_audio.py.
"""
from __future__ import annotations
import sys
import torchaudio
from pathlib import Path


def download() -> None:
    data_dir = Path("./data")
    data_dir.mkdir(exist_ok=True)

    print("Connecting to Google servers …")
    print("Downloading Speech Commands v2 (~2.8 GB). This may take a few minutes.\n")

    try:
        dataset = torchaudio.datasets.SPEECHCOMMANDS(
            root=str(data_dir),
            url="speech_commands_v0.02",
            folder_in_archive="SpeechCommands",
            download=True,
        )
    except Exception as exc:
        print(f"❌  Download failed: {exc}")
        sys.exit(1)

    print(f"\n✅  Download complete — {len(dataset):,} audio files.")
    print(f"    Saved to: {data_dir / 'SpeechCommands' / 'speech_commands_v0.02'}")
    print("\nNext step: python process_audio.py")


if __name__ == "__main__":
    download()
