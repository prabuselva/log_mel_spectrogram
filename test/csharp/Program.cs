using System;
using System.IO;
using System.Collections.Generic;
using LogMelSpectrogramCS;

class Program
{
    static void Main()
    {
        String mel_bin_file = "../../../../../assets/mel_80.bin";
        if (File.Exists(mel_bin_file) == false)
        {
            // Check current directory
            if (File.Exists("mel_80.bin") == true)
            {
                mel_bin_file = "mel_80.bin";
            }
        }
        LogMelSpectrogramCSBinding log_mel_spectrogram = new LogMelSpectrogramCSBinding(mel_bin_file);

        int SAMPLE_SIZE = 16000  * 3;
        List<float> input = new List<float>(); // 3 Secs Empty Audio Sample
        for (int i=0; i<SAMPLE_SIZE; i++)
        {
            input.Add(0.0f);
        }
        Console.WriteLine("len:" + input.Count);
        List<float> output = log_mel_spectrogram.compute(input);

        Console.WriteLine("Output:");
        if (output.Count > 10)
        {
            for (int i=0; i<10; i++)
            {
                Console.WriteLine(output[i]);
            }
        }
    }
}