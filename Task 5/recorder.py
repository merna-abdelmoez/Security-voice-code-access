import sounddevice as sd
import wavio as wv

def record_audio(duration=3, filename="recording.wav"):
  """
  Records audio for `duration` seconds and saves it to `filename`.
  """
  fs = 44100
  recording = sd.rec(duration * fs, samplerate=fs, channels=2)
  sd.wait()
  wv.write(filename, recording, fs, sampwidth=2)

if __name__ == "__main__":
  record_audio(duration=3, filename="recording.wav")
