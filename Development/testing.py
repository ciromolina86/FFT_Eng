import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from influxdb import InfluxDBClient
from influxdb import DataFrameClient

from ThinkX import thinkdsp
from databases_conn import Config
from databases_conn import DBmysql
from databases_conn import DBinflux
import fft_eng


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


'''================================================'''

# testing
# test1()

# testing derivation and integration of waves
# test2()

# testing writing data to influxdb
# test3()

# testing kurtosis and kurtograms
# test4()


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


def test_mysql():
    # # define database configuration parameters
    # db_info = {}
    # db_info.update({'host': "192.168.21.134"})
    # db_info.update({'port': 3306})
    # db_info.update({'user': "root"})
    # db_info.update({'password': "sbrQp10"})
    # db_info.update({'database': "data"})

    # create an instance of DBmysql
    db1 = DBmysql(Config.mysql)

    # get the asset list
    asset_list = db1.get_vib_asset_list()

    # get the asset dictionary
    asset_dic = db1.get_vib_asset_dic()

    # close cursor and connection
    db1.exit()

    # print records
    print(asset_list)
    print(asset_dic)

def test_influx():
    # Initialization
    DATABASE_NAME = 'VIB_DB'
    ASSET_NAME = "VIB_SEN1"
    _timestamp = '_timestamp'
    X_EVTID = 'WF___X_EVTID'
    X_EVT_CHG_ID = 'WF___X_EVT_CHG_ID'
    X_FFT = 'WF___X_FFT'
    X_FFT_RED = 'WF___X_FFT_RED'
    X_FREQ = 'WF___X_FREQ'
    X_FREQ_RED = 'WF___X_FREQ_RED'
    X_TDW = 'WF___X_TDW'
    X_TDW_RED = 'WF___X_TDW_RED'
    Z_EVTID = 'WF___Z_EVTID'
    Z_EVT_CHG_ID = 'WF___Z_EVT_CHG_ID'
    Z_FFT = 'WF___Z_FFT'
    Z_FFT_RED = 'WF___Z_FFT_RED'
    Z_FREQ = 'WF___Z_FREQ'
    Z_FREQ_RED = 'WF___Z_FREQ_RED'
    Z_TDW = 'WF___Z_TDW'
    Z_TDW_RED = 'WF___Z_TDW_RED'

    # define database configuratin parameters
    db_info = {}
    db_info.update({'host': "192.168.21.134"})  #localhost, 192.168.1.118
    db_info.update({'port': 8086})
    db_info.update({'database': DATABASE_NAME})

    # create an instance of DBinflux
    db1 = DBinflux(config=db_info)

    # sql = "select * from " + asset_name
    sql = "select {}, {} from {} order by time".format(_timestamp, X_TDW, ASSET_NAME)
    binds = {}

    # Execute query
    datasets_dic = db1.query(sql)

    # Get pandas dataframe
    pdf_wave = datasets_dic[ASSET_NAME]
    pdf_wave.to_csv(path_or_buf='C://Users//cmolina//Desktop//pdf_wave.csv')
    # print(pdf_wave)
    print('TDW shape: {}'.format(pdf_wave.shape))

    # create testing wave and spectrum
    wave = thinkdsp.Wave(ys=pdf_wave[X_TDW], ts=np.linspace(0,1,100), framerate=100)
    spectrum = fft_eng.get_spectrum(wave=wave, window='hanning')
    spectrum_red = spectrum.copy()

    # create dictionary to use on pandas dataframe creation
    spec_dic = {X_FFT: spectrum.amps, X_FREQ: spectrum.fs,
                X_FFT_RED: spectrum_red.amps, X_FREQ_RED: spectrum_red.fs}
    # print(spec_dic)

    # convert pandas dataframe
    pdf_spec = pd.DataFrame(spec_dic, index=pdf_wave.index[:len(spectrum)])
    # print(pdf_spec)
    pdf_spec.to_csv(path_or_buf='C://Users//cmolina//Desktop//pdf_fft.csv')
    print('FFT shape: {}'.format(pdf_spec.shape))

    # write dataframe to influx database
    db1.write_points(pdf=pdf_spec, meas=ASSET_NAME)

    # prepare sql query
    sql2 = "select {}, {}, {}, {} from {} order by time".format(_timestamp, X_TDW, X_FFT, X_FREQ, ASSET_NAME)

    # query influx database
    datasets_dic = db1.query(sql2)

    # get the pandas dataframe
    pdf_all = datasets_dic[ASSET_NAME]
    pdf_all.to_csv(path_or_buf='C://Users//cmolina//Desktop//pdf_all.csv')
    # print(pdf_all)
    print('ALL shape {}'.format(pdf_all.shape))

def write_influx_test_data():
    # create a testing wave
    wave = thinkdsp.SinSignal(freq=10, amp=1, offset=0).make_wave(duration=1, start=0, framerate=100)

    # create an influxdb client
    client = InfluxDBClient(**Config.influx)

    # Generating test data
    for i in range(10):
        # print('>>> loop #1, iter #{}'.format(i))
        count = 0
        _time = np.int64(time.time() * 1000)

        for k in wave.ys:
            # print('>>> loop #1, iter #{} >>> loop #2, iter #{}'.format(i, count))

            if count == 0:

                points = [{
                    "measurement": 'VIB_SEN1',
                    "time": _time + count,
                    "fields": {
                        "WF___X_TDW": k,
                        "WF___X_EVTID": str(_time),
                        "WF___X_EVT_CHG_ID": str(_time),
                        "WF___X_FFT": -1.0,
                        "WF___Z_TDW": k,
                        "WF___Z_EVTID": str(_time),
                        "WF___Z_EVT_CHG_ID": str(_time),
                        "WF___Z_FFT": -1.0
                    }
                },
                    {
                        "measurement": 'VIB_SEN2',
                        "time": _time + count,
                        "fields": {
                            "WF___X_TDW": k,
                            "WF___X_EVTID": str(_time),
                            "WF___X_EVT_CHG_ID": str(_time),
                            "WF___X_FFT": -1.0,
                            "WF___Z_TDW": k,
                            "WF___Z_EVTID": str(_time),
                            "WF___Z_EVT_CHG_ID": str(_time),
                            "WF___Z_FFT": -1.0
                        }
                    }
                ]

                client.write_points(points)  #, time_precision='ms'

            else:

                points = [{
                    "measurement": 'VIB_SEN1',
                    "time": _time + count,
                    "fields": {
                        "WF___X_TDW": k,
                        "WF___X_EVTID": str(_time),
                        "WF___X_FFT": -1.0,
                        "WF___Z_TDW": k,
                        "WF___Z_EVTID": str(_time),
                        "WF___Z_FFT": -1.0
                    }
                },
                    {
                        "measurement": 'VIB_SEN2',
                        "time": _time + count,
                        "fields": {
                            "WF___X_TDW": k,
                            "WF___X_EVTID": str(_time),
                            "WF___X_FFT": -1.0,
                            "WF___Z_TDW": k,
                            "WF___Z_EVTID": str(_time),
                            "WF___Z_FFT": -1.0
                        }
                    }
                ]

                client.write_points(points)  #, time_precision='ms'

            count += 1

        time.sleep(1)

    client.close()

def test_downsamplig():
    def get_process_pdf(tdw_pdf, framerate, red_rate=0.5, acc=True, window='hanning', axis='X'):
        """
        It returns the pandas data frame of the acceleration wave
        :param tdw_pdf: pandas dataframe
        :param framerate: real
        :param window: string
        :return: pandas data frame
        """
        tdw_name = 'WF___{}_TDW'.format(axis)
        acc_tdw_name = 'WF___{}_TDW'.format(axis)
        acc_fft_name = 'WF___{}_FFT'.format(axis)
        acc_freq_name = 'WF___{}_FREQ'.format(axis)
        vel_fft_name = 'WF___{}_FFT_V'.format(axis)
        vel_tdw_name = 'WF___{}_TDW_V'.format(axis)

        acc_tdw_red_name = 'WF___{}_TDW_RED'.format(axis)
        acc_fft_red_name = 'WF___{}_FFT_RED'.format(axis)
        acc_freq_red_name = 'WF___{}_FREQ_RED'.format(axis)
        vel_tdw_red_name = 'WF___{}_TDW_V_RED'.format(axis)
        vel_fft_red_name = 'WF___{}_FFT_V_RED'.format(axis)

        # init
        final_pdf_list = []

        # compute tdw duration (in time)
        tdw_wave_duration = len(tdw_pdf[tdw_name]) / framerate
        tdw_wave_N = len(tdw_pdf[tdw_name])

        # create  wave from pdf
        tdw_wave = thinkdsp.Wave(ys=tdw_pdf[tdw_name], ts=np.linspace(0, tdw_wave_duration, tdw_wave_N),
                                 framerate=framerate)

        # if collecting acceleration
        if acc:
            acc_wave = tdw_wave.copy()
            vel_wave = fft_eng.integrate(tdw_wave)
            acc_fft = fft_eng.get_spectrum(wave=acc_wave, window=window)
            vel_fft = fft_eng.get_spectrum(wave=vel_wave, window=window)

            # get reduced signal
            acc_tdw_red = get_downsampled_data_ts(input_mtx_ts=acc_wave.ts, input_mtx=acc_wave.ys,
                                                  max_datapoints=red_rate, field_name=acc_tdw_name)
            acc_fft_red = get_downsampled_data(input_mtx=[acc_fft.amps],
                                               max_datapoints=red_rate, field_name=acc_fft_name)
            vel_tdw_red = get_downsampled_data_ts(input_mtx_ts=vel_wave.ts, input_mtx=vel_wave.ys,
                                                  max_datapoints=red_rate, field_name=vel_tdw_name)
            vel_fft_red = get_downsampled_data(input_mtx=vel_fft.amps,
                                               max_datapoints=red_rate, field_name=vel_fft_name)
            acc_freq_red = get_downsampled_data(input_mtx=acc_fft.fs,
                                                max_datapoints=red_rate, field_name=acc_freq_name)

            # create dictionary to use on pandas data frame creation
            dic_list = [
                {acc_fft_name: acc_fft.amps},
                {acc_freq_name: acc_fft.fs},
                {vel_tdw_name: vel_wave.amps},
                {vel_fft_name: vel_fft.amps},

                {acc_tdw_red_name: acc_tdw_red},
                {acc_fft_red_name: acc_fft_red},
                {acc_freq_red_name: acc_freq_red},
                {vel_tdw_red_name: vel_tdw_red},
                {vel_fft_red_name: vel_fft_red},
            ]

        # if collecting velocity
        else:
            vel_wave = tdw_wave.copy()
            acc_wave = fft_eng.derivate(vel_wave)
            acc_fft = fft_eng.get_spectrum(wave=acc_wave, window=window)
            vel_fft = fft_eng.get_spectrum(wave=vel_wave, window=window)

            # get reduced signal
            acc_tdw_red = get_downsampled_data_ts(input_mtx_ts=acc_wave.ts, input_mtx=acc_wave.ys,
                                                  max_datapoints=red_rate, field_name=acc_tdw_name)
            acc_fft_red = get_downsampled_data(input_mtx=acc_fft.amps,
                                               max_datapoints=red_rate, field_name=acc_fft_name)
            vel_tdw_red = get_downsampled_data_ts(input_mtx_ts=vel_wave.ts, input_mtx=vel_wave.ys,
                                                  max_datapoints=red_rate, field_name=vel_tdw_name)
            vel_fft_red = get_downsampled_data(input_mtx=vel_fft.amps,
                                               max_datapoints=red_rate, field_name=vel_fft_name)
            acc_freq_red = get_downsampled_data(input_mtx=acc_fft.fs,
                                                max_datapoints=red_rate, field_name=acc_freq_name)

            # create list of dictionary to use on pandas data frame creation
            dic_list = [
                {vel_fft_name: vel_fft.amps},
                {acc_tdw_name: acc_wave},
                {acc_fft_name: acc_fft.amps},
                {acc_freq_name: acc_fft.fs},

                {acc_tdw_red_name: acc_tdw_red},
                {acc_fft_red_name: acc_fft_red},
                {vel_tdw_red_name: vel_tdw_red},
                {vel_fft_red_name: vel_fft_red},
                {acc_freq_red_name: acc_freq_red}
            ]

        # Create list of pdf from list of dictionaries

        for d in dic_list:
            for key, value in d.items():
                final_pdf_list.append(pd.DataFrame(d, index=tdw_pdf.index[:len(value)]))

        return final_pdf_list

def test_vib_model_object():

    from databases_conn import VibModel

    model = VibModel()
    print(model.get_model())

    # for asset in model.get_asset_list():
    #     print(asset)
    #
    #     for group in model.get_group_list(asset):
    #         print('\t'+group)
    #
    #         for _tag, _id in model.get_tag_id_list(asset, group):
    #             print('\t\t'+'{}, {}'.format(_tag, _id))

def test_read_rt_value_using_model():
    from redisdb import RedisDB
    from databases_conn import VibModel
    from databases_conn import getinrtmatrix

    ################################################################################################
    # MAIN CODE
    ################################################################################################
    # Initialization
    rt_redis_data = RedisDB()
    db1 = VibModel()

    # Connect to Redis DB
    rt_redis_data.open_db()

    ################################################################################################
    # READ TAG VALUES
    ################################################################################################
    # Sensor tags ids: example: tags_ids_str = "460,461,462"
    tags_ids_str = str(db1.model['VIB_SEN1']['BAND']['X_B1_PV']['internalTagID'])

    while True:
        # Read ts and values
        tags_timestamp, tags_current_value = getinrtmatrix(rt_redis_data, tags_ids_str)
        print("###########################################")
        print("TAGS TS: %s" % tags_timestamp)
        print("TAGS VALUES: %s" % tags_current_value)
        print("###########################################")

        # Sleep
        time.sleep(1)





# def write_influx_test_data2():
#     ''' this did not work '''
#
#     # create time domain waveform
#     wave_acc = thinkdsp.SinSignal(freq=10, amp=1, offset=0).make_wave(duration=1, start=0, framerate=100)
#
#     event_id_list = ['eventid1'] * 100
#     event_id_chg_list = ['eventid1']
#     none99_list = [''] * 99
#     event_id_chg_list.extend(none99_list)
#
#     # create wave spectrum
#     spec_acc = fft_eng.get_spectrum(wave_acc)
#     wave_vel = fft_eng.integrate(wave_acc)
#
#     # create dictionaries
#     wave_acc_dic = {'WF___X_TDW': wave_acc.ys}
#     wave_vel_dic = {'WF___X_TDW_VEL': wave_vel.ys}
#     spec_acc_dic = {'WF___X_FREQ': spec_acc.fs, 'WF___X_FFT': spec_acc.amps}
#     event_id_dic = {'WF___X_EVTID': event_id_list}
#     event_id_chg_dic = {'WF___X_EVT_CHG_ID': event_id_chg_list}
#
#     # convert dictionaries to pandas dataframe
#     wave_acc_df = pd.DataFrame(wave_acc_dic)  # , index=None
#     wave_vel_df = pd.DataFrame(wave_vel_dic)  # , index=None
#     spec_acc_df = pd.DataFrame(spec_acc_dic)  # , index=None
#     event_id_df = pd.DataFrame(event_id_dic)  # , index=None
#     event_id_chg_df = pd.DataFrame(event_id_chg_dic)  # , index=None
#
#     # concatenate both dataframes into one dataframe
#     pdf = pd.concat([event_id_chg_df, event_id_df, wave_acc_df], axis=1)  # ignore_index=True,
#     # print(pdf)
#     # print(pdf.shape)
#
#     client = DataFrameClient(Config.influx)
#     client.write_points(dataframe=pdf, measurement='VIB_SEN1')
#     client.close()
#
# def read_influx_test_data():
#     # Initialization
#     DATABASE_NAME = 'VIB_DB'
#     ASSET_NAME = "VIB_SEN1"
#     _timestamp = '_timestamp'
#     X_EVTID = 'WF___X_EVTID'
#     X_EVT_CHG_ID = 'WF___X_EVT_CHG_ID'
#     X_FFT = 'WF___X_FFT'
#     X_FFT_RED = 'WF___X_FFT_RED'
#     X_FREQ = 'WF___X_FREQ'
#     X_FREQ_RED = 'WF___X_FREQ_RED'
#     X_TDW = 'WF___X_TDW'
#     X_TDW_RED = 'WF___X_TDW_RED'
#     Z_EVTID = 'WF___Z_EVTID'
#     Z_EVT_CHG_ID = 'WF___Z_EVT_CHG_ID'
#     Z_FFT = 'WF___Z_FFT'
#     Z_FFT_RED = 'WF___Z_FFT_RED'
#     Z_FREQ = 'WF___Z_FREQ'
#     Z_FREQ_RED = 'WF___Z_FREQ_RED'
#     Z_TDW = 'WF___Z_TDW'
#     Z_TDW_RED = 'WF___Z_TDW_RED'
#
#     X_EVTID1 = '2020-03-03 01:02:00.000000+00:00'
#
#     # create an instance of DBinflux
#     db1 = DBinflux(config=Config.influx)
#
#     # sql = "select * from " + ASSET_NAME
#     # sql = "select " + X_EVTID + " from " + ASSET_NAME
#     # sql = "select {}, {}, {} from {} order by time".format(X_TDW, X_EVTID, X_EVT_CHG_ID, ASSET_NAME)
#     # sql = "select {}, {}, {} from {} where WF___X_EVTID=$X_EVTID1;".format(X_TDW, X_EVTID, X_EVT_CHG_ID, ASSET_NAME)
#     sql = "select {}, {}, {} from {} where WF___X_FFT<>0;".format(X_TDW, X_EVTID, X_EVT_CHG_ID, ASSET_NAME)
#     # sql = "select {} from {} where {}=$X_EVTID1;".format(X_EVT_CHG_ID, ASSET_NAME, X_EVTID)
#     bind_params = {'X_EVTID1': X_EVTID1}
#
#     # Execute query
#     datasets_dic = db1.query(sql, bind_params)
#
#     # Get pandas dataframe
#     pdf_wave = datasets_dic[ASSET_NAME]
#     # pdf_wave.to_csv(path_or_buf='C://Users//cmolina//Desktop//pdf_wave.csv')
#     print(pdf_wave)
#     # print('TDW shape: {}'.format(pdf_wave.shape))
#     # print(pdf_wave.keys)


#########################
#     MAIN CODE
#########################
if __name__ == "__main__":
    '''execute only if run as a main script'''
    print('>>>>>>>>>>>>>>>>>>>>')
    print('running testing as main script')
    print('>>>>>>>>>>>>>>>>>>>>')

    # write_csv()
    # write_csv2()
    # test5()
    # test6()
    # test777()
    # test8()
    # test9()
    # test_yandy()
    # test16()
    # write data to MySQL
    # test_mysql()

    # testing influx
    # write_influx_test_data()
    # read_influx_test_data()

    test_read_rt_value_using_model()
