'''import useful modules'''
from ThinkX import thinkdsp
from ThinkX import thinkplot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import hilbert
import json

def read_sde_tag(query='0'):
    ''' read latest value of sde tag

    :param query: query string
    :return:
    '''

    # TODO: implement this
    tag = np.random.random_sample()

    return tag

def read_sde_tdw(tag='vrms', start=0, n=8192):
    ''' Read an array of values store on the SDE as a time series

    :param tag: SDE tag
    :param start: starting timestamp
    :param stop: ending timestamp
    :return: np.array object
    '''

    # this is just an idea of how to read
    # the time domain waveform stored on the SDE
    tdw = np.array()

    # read a series of values from SDE
    for i in range(stop=n):
        # create query
        query = 'select {} ' \
                'from table_1 ' \
                'where ts={}'.format(tag, start + i)

        # insert value in numpy array
        # tdw[i] = read_sde_tag(query=query)
        tdw[i] = np.random.random_sample()

def read_wave(sde_tag='vrms',n_tdw=8192, fs=20000):
    # read time domain waveform from SDE
    # tdw = n_tdw values from sde_tag
    # n_tdw = len(tdw)
    n_tdw = 8192

    # create an array of evenly spaced sample times
    t = np.linspace(start=0, stop=(n_tdw/fs), num=n_tdw, endpoint=False)

    # reproduce time domain waveform for testing
    tdw = 1.0 * np.cos(2 * np.pi * 20 * t) + \
          0.05 * np.cos(2 * np.pi * 2500 * t) + \
          0.04 * np.cos(2 * np.pi * 3000 * t) + \
          0.03 * np.cos(2 * np.pi * 3500 * t)

    # returns the result wave
    return thinkdsp.Wave(ys=tdw, ts=t, framerate=fs)

def read_wave2(sde_tag='vrms', n_tdw=8192, fs=20000):
    # read time domain waveform from SDE
    # tdw = n_tdw values from sde_tag

    # create an array of evenly spaced sample times
    t = np.linspace(start=0, stop=(n_tdw/fs), num=n_tdw, endpoint=False)

    # reproduce time domain waveform for testing
    tdw = 1.0 * np.cos(2 * np.pi * 20 * t) + \
          0.15 * np.cos(2 * np.pi * 2500 * t) + \
          0.14 * np.cos(2 * np.pi * 3000 * t) + \
          0.13 * np.cos(2 * np.pi * 3500 * t)

    # returns the result wave
    return thinkdsp.Wave(ys=tdw, ts=t, framerate=fs)

def get_wave(spectrum):
    '''Reconstructing wave back from spectrum object

    :param s: spectrum
    :return: reconstructed wave
    '''

    # removing the attenuation due to the FFT algorithm
    N = len(spectrum.hs)
    spectrum.hs /= 2/N

    # returns the reconstructed wave
    return spectrum.make_wave()

def get_spectrum(wave, window='hanning', beta=6):
    ''' Get a spectrum from a given wave

    :param wave: thinkdsp.Wave object
    :param window: windowing string, default is 'hanning'. e.g. 'hanning', 'blackman', 'bartlett', 'kaiser'
    :return: thinkdsp.Spectrum object
    '''

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

    # removing the attenuation of the FFT algorithm
    N = len(result.hs)
    result.hs *= 2/N

    # returns a thinkdsp.Spectrum object
    return result

def get_spectrum2(wave, window='hanning', normalize=False, amp=1.0, unbias=False, beta=6):
    # unbiases the signal
    if unbias == True:
        wave.unbias()

    # normalizes the signal
    if normalize == True:
        wave.normalize(amp=amp)

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

    # removing the attenuation due to the FFT algorithm
    N = len(result.hs)
    result.hs *= 2/N

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

def demodulation_wave(wave, fc1=400, fc2=800):
    ''' computes the envelope of a wave (fc1<freq<fc2)

    :param wave: thinkdsp.Wave object
    :param fc1: first cutoff frequency
    :param fc2: second cutoff frequency
    :return: thinkdsp.Spectrum object
    '''

    # create a copy of the original wave
    raw_wave = wave.copy()

    # make a spectrum from  wave
    raw_spectrum = raw_wave.make_spectrum(full=False)

    # apply a high pass filter at fc1 to the raw spectrum
    raw_spectrum.high_pass(fc1)
    raw_spectrum.low_pass(fc2)

    # make a time waveform from the filtered spectrum
    raw_wave_filtered = raw_spectrum.make_wave()
    # apply hanning windowing to the result waveform
    raw_wave_filtered.window(np.hanning(len(raw_wave_filtered)))

    # obtain the envelop of the result waveform
    raw_wave_filtered_envelop = thinkdsp.Wave(ys=np.abs(hilbert(raw_wave_filtered.ys)),
                                              ts=raw_wave_filtered.ts,
                                              framerate=raw_wave_filtered.framerate)

    # obtain the spectrum from the envelop
    result = raw_wave_filtered_envelop.make_spectrum(full=False)

    # returns the result
    return result

def demodulation_spectrum(spectrum, fc1=400, fc2=800):
    ''' computes the envelope of a spectrum (fc1<freq<fc2)

    :param spectrum: thinkdsp.Spectrum object
    :param fc1: first cutoff frequency
    :param fc2: second cutoff frequency
    :return: tinkdsp.Spectrum object
    '''

    # make a spectrum copy of the original spectrum
    raw_spectrum = spectrum.copy()

    # apply a high pass filter at fc1 to the raw spectrum
    raw_spectrum.high_pass(fc1)
    raw_spectrum.low_pass(fc2)

    # make a time waveform from the filtered spectrum
    raw_wave_filtered = raw_spectrum.make_wave()
    # apply hanning windowing to the result waveform
    raw_wave_filtered.window(np.hanning(len(raw_wave_filtered)))

    # obtain the envelop of the result waveform
    raw_wave_filtered_envelop = thinkdsp.Wave(ys=np.abs(hilbert(raw_wave_filtered.ys)),
                                              ts=raw_wave_filtered.ts,
                                              framerate=raw_wave_filtered.framerate)

    # obtain the spectrum from the envelop
    result = raw_wave_filtered_envelop.make_spectrum(full=False)

    # returns the result
    return result

def spectrum_to_dict(s):
    ''' returns a given dictionary of a spectrum object

    :param s: thinkdsp.Spectrum object
    :return: result dictionary
    '''

    # create a result dictionary
    result = {}
    result.update({'real': s.real})
    result.update({'imag': s.imag})
    result.update({'angles': s.angles})
    result.update({'amps': s.amps})
    result.update({'power': s.power})
    result.update({'freqs': s.fs})

    return result

def wave_to_dict(w):
    ''' returns a given dictionary of a wave object

    :param w: thinkdsp.Wave object
    :return: result dictionary
    '''

    # create a result dictionary
    result = {}
    result.update({'samples': w.ys})
    result.update({'sample_time': w.ts})
    result.update({'fs': w.framerate})

    return result

def diff_wave(w0, w1):
    '''Computes and returns the  difference between waves w1-w0.
    Restriction: Both waves need to be similar(same sampling frequency, same length)

    :param w0: wave 0
    :param w1: wave 1
    :return: wave difference
    '''

    # returns result wave
    return thinkdsp.Wave(ys=np.abs(w1.ys-w0.ys), ts=w0.ts, framerate=w0.framerate)

def diff_spectrum(s0, s1):
    '''Computes and returns the  difference between spectra s1-s0.
    Restriction: Both waves need to be similar(same sampling frequency, same length)

    :param s0: spectrum 0
    :param s1: spectrum 1
    :return: wave difference
    '''

    # returns result spectrum
    return thinkdsp.Spectrum(hs=np.abs(s1.amps-s0.amps), fs=s0.fs, framerate=s0.framerate)

'''=================================================='''

def main():
    # DO SOMETHING
    dummy = 1

if __name__ == "__main__":
    # execute only if run as a script
    main()

