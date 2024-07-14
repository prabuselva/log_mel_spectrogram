import sys
import platform

if platform.system() == "Windows":
    sys.path.append( '../../build/Release' )
else:
    sys.path.append( '../../build' )

import numpy as np
# Import our custom log_mel_spectrogram module
import py_log_mel_spectrogram

WHISPER_SAMPLE_RATE = 16000
log_mel_spectrogram_instance = py_log_mel_spectrogram.LogMelSpectrogram("../../assets/mel_80.bin")
audio_data = np.zeros(WHISPER_SAMPLE_RATE * 3)
print(audio_data.shape)
output_data = log_mel_spectrogram_instance.compute(audio_data)
print(output_data.shape)
print(output_data[:10])
