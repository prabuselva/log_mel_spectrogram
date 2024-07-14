import sys
import platform

if platform.system() == "Windows":
    sys.path.append( '../../build/Release' )
else:
    sys.path.append( '../../build' )
    
import whisper
import torch
# Import our custom log_mel_spectrogram module
import py_log_mel_spectrogram

model = whisper.load_model("tiny")

# load audio and pad/trim it to fit 30 seconds
audio = whisper.load_audio("../../assets/jfk.wav")
audio = whisper.pad_or_trim(audio)

# make log-Mel spectrogram and move to the same device as the model
#o_mel = whisper.log_mel_spectrogram(audio).to(model.device)
log_mel_spectrogram_instance = py_log_mel_spectrogram.LogMelSpectrogram("../../assets/mel_80.bin")
lmel = log_mel_spectrogram_instance.compute(audio)
mel = lmel[:(80 * 3000)].reshape(80, 3000)
mel = torch.from_numpy(mel)

# detect the spoken language
_, probs = model.detect_language(mel)
print(f"Detected language: {max(probs, key=probs.get)}")

# decode the audio
options = whisper.DecodingOptions(fp16=True)
result = whisper.decode(model, mel, options)

# print the recognized text
print(result.text)

