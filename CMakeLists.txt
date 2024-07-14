cmake_minimum_required(VERSION 3.10)
project(LogMelSpectrogram)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(inc)

add_subdirectory(extern/pybind11)
pybind11_add_module(py_log_mel_spectrogram src/log_mel_spectrogram.cpp src/pybind11_binding.cpp)


add_library(log_mel_spectrogram MODULE src/log_mel_spectrogram.cpp src/pybind11_binding.cpp)
target_link_libraries(log_mel_spectrogram PRIVATE pybind11::module)

if(MSVC)
    add_library(cs_log_mel_spectrogram SHARED
        src/log_mel_spectrogram.cpp
        src/csharp_binding.cpp
    )
    set_target_properties(cs_log_mel_spectrogram PROPERTIES COMMON_LANGUAGE_RUNTIME "")

    target_include_directories(cs_log_mel_spectrogram PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} inc)

    target_compile_options(cs_log_mel_spectrogram PRIVATE /EHsc)
    #target_link_libraries(cs_log_mel_spectrogram PRIVATE log_mel_spectrogram)
endif()