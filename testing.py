from ThinkX import thinkdsp
import pandas as pd
import numpy as np
import fft_eng

import influxdb_conn
import matplotlib.pyplot as plt

from influxdb import InfluxDBClient
from influxdb import DataFrameClient
from databases_conn import Config


def test1():
    print('test1 ran!')

    '''
    # read time domain waveforms
    wave1 = fft_eng.read_wave(sde_tag='test', n_tdw=8192, fs=20000)
    wave2 = fft_eng.read_wave2(sde_tag='test', n_tdw=8192, fs=20000)

    # get the spectrum. default windowing = 'hanning'
    spectrum1 = fft_eng.get_spectrum(wave=wave1)
    spectrum2 = fft_eng.get_spectrum(wave=wave2)

    # obtain the spectrogram of a wave
    spectrogram1 = wave1.make_spectrogram(seg_length=128)

    # get maximum absolute difference between spectra
    wdiff = fft_eng.diff_wave(w0=wave1, w1=wave2)
    sdiff = fft_eng.diff_spectrum(s0=spectrum1, s1=spectrum2)

    # reconstruct the wave
    rec_wave1 = fft_eng.get_wave(spectrum=spectrum1)
    rec_wave2 = fft_eng.get_wave(spectrum=spectrum2)

    #  get envelope from spectra
    envelope1 = fft_eng.demodulation_wave(wave=wave1, fc1=2000, fc2=5000)
    envelope2 = fft_eng.demodulation_wave(wave=wave1, fc1=2000, fc2=5000)

    # testing functions
    # demodulate_steps(wave=wave1)
    # envelope = demodulation_spectrum(spectrum=spectrum1, fc=2000)
    '''
    '''============================================='''
    '''
    # testing plots
    # create a figure
    fig1 = plt.figure()
    # create a subplot
    ax = fig1.add_subplot(321)
    # plot a wave1
    wave1.plot(label='wave 1', color='b')
    ax.legend()
    # create a subplot
    ax = fig1.add_subplot(322)
    # plot a wave1
    rec_wave1.plot(label='rec wave 1', color='b')
    ax.legend()
    # create another subplot
    ax = fig1.add_subplot(323)
    # plot a wave2
    wave2.plot(label='wave 2', color='g')
    ax.legend()
    # create a subplot
    ax = fig1.add_subplot(324)
    # plot a wave1
    rec_wave2.plot(label='rec wave 2', color='g')
    ax.legend()
    # create a subplot
    ax = fig1.add_subplot(325)
    # plot a difference
    wdiff.plot(label='difference', color='r')
    ax.legend()
    '''

    '''
    # create a figure
    fig2 = plt.figure()
    # create a subplot
    ax = fig2.add_subplot(321)
    # plot a spectrum1
    spectrum1.plot(label='spectrum1', color='b')
    ax.legend()
    # create a subplot
    ax = fig2.add_subplot(322)
    # plot a envelope 1
    envelope1.plot(label='envelope1', color='b')
    ax.legend()
    # create another subplot
    ax = fig2.add_subplot(323)
    # plot a wave2
    spectrum2.plot(label='spectrum2', color='g')
    ax.legend()
    ax = fig2.add_subplot(324)
    # plot a envelope
    envelope2.plot(label='envelope2', color='g')
    ax.legend()
    # create a subplot
    ax = fig2.add_subplot(325)
    # plot a difference
    sdiff.plot(label='difference', color='r')
    ax.legend()
    '''

    '''
    # create a pulse wave
    pulse = np.zeros(1000)
    pulse[500] = 1
    t_pulse = np.linspace(0, 1, 1000)
    wave_pulse = thinkdsp.Wave(ys=pulse, ts=t_pulse, framerate=thinkdsp.infer_framerate(ts=t_pulse))
    # create a pulse spectrum
    spectrum_pulse = wave_pulse.make_spectrum()

    # create a figure
    fig3 = plt.figure()
    # create a subplot
    ax = fig3.add_subplot(121)
    # plot a wave
    wave_pulse.plot(label='pulse', color='b')
    ax.legend()
    # create a subplot
    ax = fig3.add_subplot(122)
    # plot a spectrum
    spectrum_pulse.plot(label='spectrum', color='b')
    ax.legend()

    plt.show()
    '''

    '''
    # create a figure
    fig4 = plt.figure()
    # create a subplot
    ax = fig4.add_subplot(311)
    # plot a wave
    wave1.plot(label='wave', color='b')
    ax.legend()
    # create a subplot
    ax = fig4.add_subplot(312)
    # plot a spectrum
    spectrum1.plot(label='spectrum', color='g')
    ax.legend()
    # create a subplot
    ax = fig4.add_subplot(313)
    # plot a spectrogram
    spectrogram1.plot()  # (label='spectrogram', color='r')
    # ax.legend()

    plt.show()
    '''

def test2():
    '''testing differentiation & integration filters

    :return:
    '''

    print('test2 ran!')

    # create a cosine wave
    wave1 = (thinkdsp.SinSignal(freq=1, amp=1, offset=0)).make_wave(duration=1, start=0, framerate=100)

    # add some dc level
    # wave1.ys += 1

    # apply differentiation filter
    wave2 = fft_eng.derivate(wave1)
    wave3 = fft_eng.derivate(wave2)

    # print dc levels for all signals
    print('starting points: {}, {}, {}'.format(wave1.ys[0], wave2.ys[0], wave3.ys[0]))

    # print maximum for all signals
    print('max amplitudes: {}, {}, {}'.format(np.max(wave1.ys), np.max(wave2.ys), np.max(wave3.ys)))

    # plot the waves
    fig = plt.figure()
    ax = fig.add_subplot()
    wave1.plot(label='orig')
    plt.legend()
    wave2.plot(label='1st deriv')
    plt.legend()
    wave3.plot(label='2nd deriv')
    plt.legend()
    plt.plot(wave1.ts, np.ones(len(wave1)), color='r')

    # plt.show()

    # apply integration filter
    wave4 = fft_eng.integrate(wave3)
    wave5 = fft_eng.integrate(wave4)

    # print dc levels for all signals
    print('starting points: {}, {}, {}'.format(wave3.ys[0], wave4.ys[0], wave5.ys[0]))

    # print maximum for all signals
    print('max amplitudes: {}, {}, {}'.format(np.max(wave3.ys), np.max(wave4.ys), np.max(wave5.ys)))

    # plot the waves
    fig2 = plt.figure()
    ax = fig2.add_subplot()
    wave3.plot(label='orig')
    plt.legend()
    wave4.plot(label='1st integ')
    plt.legend()
    wave5.plot(label='2nd integ')
    plt.legend()
    plt.plot(wave1.ts, np.ones(len(wave1)), color='r')

    plt.show()

def test3():
    ''' testing writing data to influxdb

    :return:
    '''

    print('test3 ran!')

    # write values to influxdb for testing grafana dashboard
    influxdb_conn.writeTestValues2()

def test4():
    ''' testing kurtosis and kurtogram analysis

    :return:
    '''

    # create a cosine wave
    wave1 = (thinkdsp.SinSignal(freq=1, amp=1, offset=0)).make_wave(duration=1, start=0, framerate=100)

    # computes the kurtosis of a wave
    print(fft_eng.get_kurtosis(a=wave1.ys))

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

def test5():
    dic = {'VIB_SEN1': [('SCAL___X_VRMS', 283), ('SCAL___X_HF_ACC', 284)]}
    for x in dic.keys():
        for tag, id in dic[x]:
            print(id)

def test6():
    from influxdb import InfluxDBClient
    from influxdb import DataFrameClient
    from databases_conn import Config

    # create a testing wave
    wave = [1.1]*10
    time = [1583448055962001, 1583448055962002, 1583448055962003, 1583448055962004, 1583448055962005,
            1583448055962006, 1583448055962007, 1583448055962008, 1583448055962009, 1583448055962010]


    # create an influxdb client
    client = InfluxDBClient(**Config.influx)

    # Generating test data
    for k in range(len(wave)):

         points = [{
            "measurement": 'TEST2',
            'time': time[k],
            "fields": {"x": wave[k], "y": -999.99}
         }]

         client.write_points(points)  # , time_precision='ms'

    client.close()

def test777():
    from influxdb import InfluxDBClient
    from influxdb import DataFrameClient
    from databases_conn import Config

    # create an instance of DBinflux
    db1 = DataFrameClient(**Config.influx)

    # create query
    sql = "select * from TEST2"

    # Execute query
    datasets_dic = db1.query(sql)

    # Get pandas data frame
    pdf_wave = datasets_dic['TEST2']
    print('>>> pdf read')
    # print(pdf_wave)

    # Create new pandas dataframe
    spec_dic = {'fft': pdf_wave['x'][:5]+10}
    pdf_spec = pd.DataFrame(spec_dic, index=pdf_wave.index[:5])
    print('>>> pdf written')
    # print(pdf_spec)

    # write data frame to influx database
    db1.write_points(dataframe=pdf_spec, measurement='TEST2')

    db1.close()

def test8():
    from influxdb import InfluxDBClient
    from influxdb import DataFrameClient
    from databases_conn import Config

    # create an instance of DBinflux
    db1 = InfluxDBClient(**Config.influx)

    # create query
    sql = "select x from TEST"

    # Execute query
    resultset = list(db1.query(sql).get_points(measurement='TEST'))

    for row in resultset:
        # print('{} -- {}'.format(row['time'], row['x']))
        points = [{
            "measurement": 'TEST',
            "time": row['time'],
            "fields": {"y":  row['x']+1}
        }]

        db1.write_points(points)  # , time_precision='ms'

    db1.close()

def test9():
    from influxdb import InfluxDBClient
    from influxdb import DataFrameClient
    from databases_conn import Config

    # create an instance of DBinflux
    db1 = DataFrameClient(**Config.influx)

    # create query
    sql = "select * from TEST1"

    # Execute query
    datasets_dic = db1.query(sql)

    # Get pandas data frame
    pdf_read = datasets_dic['TEST1']
    print('>>> pdf read')
    # print(pdf_read)

    # Dummy fft
    fft = [66.6, 77.7, 88.8]
    fft_len = len(fft)
    fft_index = 0

    # Replace values with new numbers
    for ind in pdf_read.index:
        if fft_index < fft_len:
            pdf_read['fft'][ind] = fft[fft_index]
            fft_index += 1
        else:
            break

    print('>>> pdf written')
    # print(pdf_read)

    # write dataframe back to influx database
    db1.write_points(pdf_read, 'TEST1')


def test_yandy():
    """Instantiate the connection to the InfluxDB client."""
    user = ''
    password = ''
    dbname = 'YANDY_TEST_DB'
    protocol = 'line'

    client = DataFrameClient(**Config.influx)  #host, port, user, password, dbname

    print("Create pandas DataFrame")
    data_csv_path = "C:\\Users\\cmolina\\Desktop\\test_forecast.csv"
    df = pd.read_csv(data_csv_path, delimiter=',')

    df['Time'] = pd.to_datetime(df['Time'])
    df.set_index('Time', inplace=True)

    # print("Create database: " + dbname)
    # client.create_database(dbname)

    print("Write DataFrame FIRST TIME")
    client.write_points(df, 'MEAS_TEST', protocol=protocol)

    client.close()

    # ===============================================================================
    client = DataFrameClient(**Config.influx)  # host, port, user, password, dbname

    print("Read DataFrame FIRST TIME")
    datasets_dic = client.query("select * from MEAS_TEST")
    pdf_read = datasets_dic['MEAS_TEST']
    # datasets_dic = client.query("select * from TEST")
    # pdf_read = datasets_dic['TEST']

    # Dummy fft
    fft = [66.6, 77.7, 88.8, 66.6]
    fft_len = len(fft)
    fft_index = 0

    # Replace values with dummy numbers
    for ind in pdf_read.index:
        if fft_index < fft_len:
            pdf_read['target'][ind] = fft[fft_index]
            fft_index += 1
        else:
            break

    print("Write DataFrame SECOND TIME")
    client.write_points(pdf_read, 'MEAS_TEST')
    # client.write_points(pdf_read, 'TEST')

    client.close()

def test16():
    from influxdb import InfluxDBClient
    from influxdb import DataFrameClient
    from databases_conn import Config

    # create a testing wave
    wave = [1.1]*10
    fft = [-999.99]*10
    time = [1583448055962001, 1583448055962002, 1583448055962003, 1583448055962004, 1583448055962005,
            1583448055962006, 1583448055962007, 1583448055962008, 1583448055962009, 1583448055962010]

    # create an influxdb client
    client = DataFrameClient(**Config.influx)

    # create points dataframe
    points_df = pd.DataFrame({'time': time, 'wave': wave, 'fft': fft})

    points_df['time'] = pd.to_datetime(points_df['time'])
    points_df.set_index('time', inplace=True)

    # write test data
    print("Write DataFrame FIRST TIME")
    client.write_points(points_df, 'TEST1')

    # read test data
    print("Read DataFrame FIRST TIME")
    datasets_dic = client.query("select * from TEST1")
    pdf_read = datasets_dic['TEST1']

    # Dummy fft
    fft = [11.1, 22.2, 33.3, 44.4]
    fft_len = len(fft)
    fft_index = 0

    # Replace values with dummy numbers
    for ind in pdf_read.index:
        if fft_index < fft_len:
            pdf_read['fft'][ind] = fft[fft_index]
            fft_index += 1
        else:
            break

    print("Write DataFrame SECOND TIME")
    client.write_points(pdf_read, 'TEST1')












if __name__ == "__main__":
    '''execute only if run as a main script'''
    print('testing ran as main!')

    # write_csv()
    # write_csv2()
    # test5()
    # test6()
    test777()
    # test8()
    # test9()
    # test_yandy()
    # test16()