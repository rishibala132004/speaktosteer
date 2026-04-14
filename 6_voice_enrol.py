import sounddevice as sd
import soundfile as sf
import time

print("\n--- NEW MASTER KEY (1.5 Seconds) ---")
print("Wait for the red dot, then say 'FORWARD' clearly.")
time.sleep(2)

print("🔴 RECORDING...")
# 1.5 seconds is the sweet spot for both models
recording = sd.rec(int(16000 * 1.5), samplerate=16000, channels=1, dtype='float32')
sd.wait()


sf.write('my_authorized_voice.wav', recording, 16000)
print("✅ NEW MASTER KEY SAVED!")