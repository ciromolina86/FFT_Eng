'''import useful modules'''
from ThinkX import thinkdsp
from ThinkX import thinkplot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import hilbert
import json



def get_wave(sde_tag='vrms', n_tdw=8192, fs=20000):
    # read time domain waveform from SDE
    # tdw = n_tdw values from sde_tag

    # create an array of evenly spaced sample times
    t = np.linspace(start=0, stop=n_tdw/fs, num=n_tdw, endpoint=False)

    # reproduce time domain waveform for testing
    tdw = 1.0 * np.cos(2 * np.pi * 20 * t) + 0.05 * np.cos(2 * np.pi * 2500 * t) + 0.04 * np.cos(2 * np.pi * 3000 * t) + 0.03 * np.cos(2 * np.pi * 3500 * t)

    # create the result wave
    wave = thinkdsp.Wave(ys=tdw, ts=t, framerate=fs)

    return wave

def get_spectrum(wave, window='hanning', normalize=False, unbias=False, beta=6):
    # unbiases the signal
    if unbias == True:
        wave.unbias()

    # normalizes the signal
    if normalize == True:
        wave.normalize()

    # apply user defined window to the time domain waveform
    if window == 'hanning':
        '''The Hanning window is a taper formed by using a weighted cosine'''

        wave.window(np.hanning(len(wave.ys)))

    elif window == 'blackman':
        '''The Blackman window is a taper formed by using the first three terms of a summation of cosines. 
        It was designed to have close to the minimal leakage possible. 
        It is close to optimal, only slightly worse than a Kaiser window'''

        wave.window(np.blackman(len(wave.ys)))

    elif window == 'bartlett':
        '''The Bartlett window is very similar to a triangular window, 
        except that the end points are at zero. It is often used in signal processing for tapering a signal, 
        without generating too much ripple in the frequency domain.'''

        wave.window(np.bartlett(len(wave.ys)))

    elif window == 'kaiser':
        '''The Kaiser window is a taper formed by using a Bessel function.
        beta    Window shape
        0	    Rectangular
        5	    Similar to a Hamming
        6	    Similar to a Hanning
        8.6	    Similar to a Blackman '''

        wave.window(np.kaiser(len(wave.ys), beta=beta))

    else:
        '''The Hanning window is a taper formed by using a weighted cosine'''

        wave.window(np.hanning(len(wave.ys)))

    # obtain the spectrum from a wave
    result = wave.make_spectrum(full=False)

    return result

def demodulate_steps(wave):
    '''
    show the 8 plots of the demodulation process

    :param wave: thinkdsp.Wave object
    :return:
    '''

    # create a copy of the original wave
    raw_wave = wave.copy()

    # create a figure
    fig = plt.figure()

    # create a subplot
    ax = fig.add_subplot(421)
    # plot a segment of the raw_wave
    raw_wave_segment = raw_wave.segment(0,0.1)
    raw_wave_segment.plot(label='raw wave', color='b')
    # show legend label
    ax.legend()

    # make a spectrum from test wave
    raw_spectrum = raw_wave.make_spectrum(full=False)

    # create another subplot
    ax = fig.add_subplot(422)
    # plot the raw spectrum
    raw_spectrum.plot(label='raw spectrum', color='g')
    # show legend label
    ax.legend()

    # apply a high pass filter at 2kHz to the raw spectrum
    raw_spectrum_filtered = raw_spectrum.copy()
    raw_spectrum_filtered.high_pass(2000)

    # create another subplot
    ax = fig.add_subplot(424)
    # plot the raw spectrum
    raw_spectrum_filtered.plot(label='high pass', color='r')
    # show legend label
    ax.legend()


    # make a time waveform from the filtered spectrum
    raw_wave_filtered = raw_spectrum_filtered.make_wave()
    # apply hanning windowing to the result waveform
    raw_wave_filtered.window(np.hanning(len(raw_wave_filtered)))

    # create another subplot
    ax = fig.add_subplot(423)
    # plot the raw spectrum
    raw_wave_filtered_segment = raw_wave_filtered.segment(0, 0.01)
    raw_wave_filtered.plot(label='high pass', color='c')
    # show legend label
    ax.legend()

    # obtain the envelop of the result waveform
    raw_wave_filtered_envelop = thinkdsp.Wave(ys=np.abs(hilbert(raw_wave_filtered.ys)),
                                              ts=raw_wave_filtered.ts,
                                              framerate=raw_wave_filtered.framerate)

    # create another subplot
    ax = fig.add_subplot(425)
    # plot the raw spectrum
    raw_wave_filtered_envelop_segment = raw_wave_filtered_envelop.segment(0, 0.01)
    raw_wave_filtered_envelop.plot(label='envelop', color='m')
    # show legend label
    ax.legend()

    # obtain the spectrum from the envelop
    raw_spectrum_filtered_envelop = raw_wave_filtered_envelop.make_spectrum(full=False)

    # create another subplot
    ax = fig.add_subplot(426)
    # plot the raw spectrum
    raw_spectrum_filtered_envelop.plot(label='envelop', color='y')
    # show legend label
    ax.legend()

    # color option = one of these {'b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'}
    # show plot
    plt.show()

def demodulation_wave(wave, fc=400):
    # create a copy of the original wave
    raw_wave = wave.copy()

    # make a spectrum from  wave
    raw_spectrum = raw_wave.make_spectrum(full=False)

    # apply a high pass filter at fc1 to the raw spectrum
    raw_spectrum.high_pass(fc)

    # make a time waveform from the filtered spectrum
    raw_wave_filtered = raw_spectrum.make_wave()
    # apply hanning windowing to the result waveform
    raw_wave_filtered.window(np.hanning(len(raw_wave_filtered)))

    # obtain the envelop of the result waveform
    raw_wave_filtered_envelop = thinkdsp.Wave(ys=np.abs(hilbert(raw_wave_filtered.ys)),
                                              ts=raw_wave_filtered.ts,
                                              framerate=raw_wave_filtered.framerate)

    # obtain the spectrum from the envelop
    raw_spectrum_filtered_envelop = raw_wave_filtered_envelop.make_spectrum(full=False)

    # create a result dictionary
    result_dict = {}
    result_dict.update({'amps': raw_spectrum_filtered_envelop.amps})
    result_dict.update({'power': raw_spectrum_filtered_envelop.power})
    result_dict.update({'freqs': raw_spectrum_filtered_envelop.fs})

    return result_dict

def demodulation_spectrum(spectrum, fc=400):
    # make a spectrum copy of the original spectrum
    raw_spectrum = spectrum.copy()

    # apply a high pass filter at fc1 to the raw spectrum
    raw_spectrum.high_pass(fc)

    # make a time waveform from the filtered spectrum
    raw_wave_filtered = raw_spectrum.make_wave()
    # apply hanning windowing to the result waveform
    raw_wave_filtered.window(np.hanning(len(raw_wave_filtered)))

    # obtain the envelop of the result waveform
    raw_wave_filtered_envelop = thinkdsp.Wave(ys=np.abs(hilbert(raw_wave_filtered.ys)),
                                              ts=raw_wave_filtered.ts,
                                              framerate=raw_wave_filtered.framerate)

    # obtain the spectrum from the envelop
    raw_spectrum_filtered_envelop = raw_wave_filtered_envelop.make_spectrum(full=False)

    # create a result dictionary
    result_dict = {}
    result_dict.update({'amps': raw_spectrum_filtered_envelop.amps})
    result_dict.update({'power': raw_spectrum_filtered_envelop.power})
    result_dict.update({'freqs': raw_spectrum_filtered_envelop.fs})

    return result_dict

def main():
    #get time domain waveform
    wave = get_wave(sde_tag='test', n_tdw=8192, fs=20000)

    # get the spectrum. default windowing = 'hanning'
    spectrum = get_spectrum(wave=wave)

    # testing functions
    demodulate_steps(wave=wave)
    envelope = demodulation_wave(wave=wave, fc=2000)
    # envelope = demodulation_spectrum(spectrum=spectrum, fc=2000)

    # plot the result envelope spectrum
    plt.plot(envelope['freqs'], envelope['amps'])
    plt.show()

if __name__ == "__main__":
    # execute only if run as a script
    main()

