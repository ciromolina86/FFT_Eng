'''import useful modules'''
from ThinkX import thinkdsp
from ThinkX import thinkplot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import hilbert



def read_wave():
    # not sure how to read from SDE

    # test signal for proof of concept
    sig1 = thinkdsp.CosSignal(freq=20, amp=1.0)
    sig2 = thinkdsp.CosSignal(freq=2500, amp=0.05)
    sig3 = thinkdsp.CosSignal(freq=3000, amp=0.04)
    sig4 = thinkdsp.CosSignal(freq=3500, amp=0.03)
    sig = sig1 + sig2 + sig3 + sig4

    # set sampling frequency
    framerate = 20000

    # make a wave object from signal object
    wave = sig.make_wave(duration=1, start=0, framerate=framerate)

    return wave

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


def main():
    # read test wave
    wave = read_wave()
    demodulate_steps(wave=wave)

if __name__ == "__main__":
    # execute only if run as a script
    main()

