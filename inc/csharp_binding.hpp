#pragma once

#include "log_mel_spectrogram.hpp"
#include <vector>
#include <vcclr.h>


using namespace System;
using namespace System::Collections::Generic;

namespace LogMelSpectrogramCS {

    public ref class LogMelSpectrogramCSBinding {
    private:
        mel_spectrogram::LogMelSpectrogram* melSpectrogram;

        void MarshalNetToStdString(System::String^ s, std::string& os)
        {
            using System::IntPtr;
            using System::Runtime::InteropServices::Marshal;

            const char* chars = (const char*)(Marshal::StringToHGlobalAnsi(s)).ToPointer( );
            os = chars;
            Marshal::FreeHGlobal(IntPtr((void*)chars));
        }

    public:
        LogMelSpectrogramCSBinding(System::String^ filename) {
            std::string filename_cpp = "";
            MarshalNetToStdString(filename, filename_cpp);
            melSpectrogram = new mel_spectrogram::LogMelSpectrogram(filename_cpp);
        }

        ~LogMelSpectrogramCSBinding() {
            this->!LogMelSpectrogramCSBinding();
        }

        !LogMelSpectrogramCSBinding() {
            delete melSpectrogram;
        }

        List<float>^ compute(List<float>^ input) {
            List<float>^ outputList = gcnew List<float>();
            
            std::vector<float> inputVector;          
            for each (float value in input) {
                inputVector.push_back(value);
            }
            
            std::vector<float> outputVector = melSpectrogram->compute(inputVector);
             
            for (float value : outputVector) {
                outputList->Add(value);
            }

            return outputList;
        }
    };

}
