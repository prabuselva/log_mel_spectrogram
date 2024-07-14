import sys
import platform

if platform.system() == "Windows":
    sys.path.append( '../../build/Release' )
else:
    sys.path.append( '../../build' )
    
import whisper
import torch
import py_log_mel_spectrogram

model = whisper.load_model("tiny")

# load audio and pad/trim it to fit 30 seconds
audio = whisper.load_audio("../../assets/jfk.wav")
print("1=>", audio.shape)
audio = whisper.pad_or_trim(audio)
print("2=>", audio.shape)

# make log-Mel spectrogram and move to the same device as the model
o_mel = whisper.log_mel_spectrogram(audio).to(model.device)
print(o_mel[0][:10])

log_mel_spectrogram_instance = py_log_mel_spectrogram.LogMelSpectrogram("../../assets/mel_80.bin")
lmel = log_mel_spectrogram_instance.compute(audio)
mel = lmel[:(80 * 3000)].reshape(80, 3000)
print(mel[0][:10])
mel = torch.from_numpy(mel)


print(o_mel.shape, mel.shape)
print(type(o_mel), type(mel), o_mel.dtype, mel.dtype)
t = torch.isclose(o_mel, mel)
print(torch.all(t))
t2 = t == False
print(t2.nonzero())
print(t[0][23])
print(o_mel[0][23], mel[0][23])


"""
for i in range(80):
    for j in range(3000):
        if mel[i][j] != o_mel[i][j]:
            print("NE: (", i, j, ") ", lmel[i * 3000 + j], o_mel[i][j])

# detect the spoken language
_, probs = model.detect_language(o_mel)
print(f"Detected language: {max(probs, key=probs.get)}")

# decode the audio
options = whisper.DecodingOptions(fp16=True)
result = whisper.decode(model, mel, options)

# print the recognized text
print(result.text)

"""