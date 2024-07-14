cmake_minimum_required(VERSION 3.10)
project(LogMelSpectrogram)

set(CMAKE_CXX_STANDARD 17)

include_directories(inc)

add_subdirectory(extern/pybind11)
pybind11_add_module(py_log_mel_spectrogram src/log_mel_spectrogram.cpp src/pybind11_binding.cpp)


add_library(log_mel_spectrogram MODULE src/log_mel_spectrogram.cpp src/pybind11_binding.cpp)
target_link_libraries(log_mel_spectrogram PRIVATE pybind11::module)
