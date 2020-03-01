from ThinkX import thinkdsp
import pandas as pd
import numpy as np
import fft_eng


def write_csv():
    # create time domain waveform
    wave = thinkdsp.SinSignal(freq=10, amp=1, offset=0).make_wave(duration=1, start=0, framerate=8192)

    # create wave spectrum
    spectrum = fft_eng.get_spectrum(wave)

    # create dictionaries
    wave_dic = {'tdw_ts': wave.ts, 'tdw_ys': wave.ys}
    spec_dic = {'fft_fs': spectrum.fs, 'fft_amps': spectrum.amps}

    # convert dictionaries to pandas dataframe
    wave_df = pd.DataFrame(wave_dic)  # , index=None
    spec_df = pd.DataFrame(spec_dic)  # , index=None

    # concatenate both dataframes into one dataframe
    pdf = pd.concat([wave_df, spec_df], axis=1)  # ignore_index=True,

    # create csv file from dataframe
    pdf.to_csv(path_or_buf='C://Users//cmolina//Desktop//tdw_fft.csv', index=None)

def write_csv2():
    # create time domain waveform
    wave = thinkdsp.SinSignal(freq=10, amp=1, offset=0).make_wave(duration=1, start=0, framerate=8192)

    # create wave spectrum
    spectrum = fft_eng.get_spectrum(wave)

    # create dictionaries
    wave_dic = fft_eng.wave_to_dict(wave)
    spec_dic = fft_eng.spectrum_to_dict(spectrum)

    # convert dictionaries to pandas dataframe
    wave_df = pd.DataFrame(wave_dic)  # , index=None
    spec_df = pd.DataFrame(spec_dic)  # , index=None

    # concatenate both dataframes into one dataframe
    pdf = pd.concat([wave_df, spec_df], axis=1)  # ignore_index=True,

    # create csv file from dataframe
    pdf.to_csv(path_or_buf='C://Users//cmolina//Desktop//tdw_fft2.csv', index=None)


if __name__ == "__main__":
    '''execute only if run as a main script'''
    print('testing ran as main!')

    # write_csv()
    # write_csv2()

