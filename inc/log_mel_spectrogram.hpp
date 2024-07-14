#ifndef LOG_MEL_SPECTROGRAM_
#define LOG_MEL_SPECTROGRAM_

#include <memory>
#include <string>
#include <vector>

namespace mel_spectrogram {

class mel_calc_cpu;

class LogMelSpectrogram {

public:
    LogMelSpectrogram(std::string mel_filter_binfile);
    ~LogMelSpectrogram();

    std::vector<float> compute(const std::vector<float>& audio_data);

private:
    std::shared_ptr<mel_calc_cpu> mel_calculator_sptr;
    int n_threads;
};

}

#endif //LOG_MEL_SPECTROGRAM_