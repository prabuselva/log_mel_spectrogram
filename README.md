# Log Mel Spectrogram C++

- This project is heavily inspired from whisper.cpp and most of the code is from https://github.com/ggerganov/whisper.cpp
- Thanks @ggerganov for this excellent contributions to the OSS AI community.
- This is a quick portable C/C++ version of computing log-mel-spectrograms that can be used in conjunction with whisper Models

# Supported Bindings
- Python
- C#

# Instructions

- Build the Python binding module or c++ dynamic library module using CMakeLists
```shell
$ mkdir build
$ cd build
$ cmake ..
# Ends up with two types of files
# For Linux - liblog_mel_spectrogram_cpp.so  py_log_mel_spectrogram.cpython-310-x86_64-linux-gnu.so
# For Windows - log_mel_spectrogram.dll py_log_mel_spectrogram.cp310-win_amd64.pyd cs_log_mel_spectrogram.dll
```

# License

- MIT License Project - You can freely distribute or use the code for your own projects.
