# Log Mel Spectrogram C++

- This project is heavily inspired from whisper.cpp and most of the code is from https://github.com/ggerganov/whisper.cpp
- Thanks @ggerganov for this excellent contributions to the OSS AI community.
- This is a quick portable C/C++ version of computing log-mel-spectrograms that can be computed to be used in conjunction with whisper Models

# Instructions

- Build the Python binding module or c++ dynamic library module using CMakeLists
```shell
$ mkdir build
$ cd build
$ cmake ..
# Ends up with two types of files
# liblog_mel_spectrogram_cpp.so  log_mel_spectrogram.cpython-310-x86_64-linux-gnu.so
```

# License

- MIT License Project - You can freely distribute or use the code for your own projects.
