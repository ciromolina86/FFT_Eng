'''================================================'''
'''import useful modules'''
import matplotlib.pyplot as plt
from ThinkX import thinkdsp
from ThinkX import thinkplot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import hilbert
import json
import fft_eng
import time
import datetime
import influxdb_conn
import databases_conn
# from redisdb import RedisDB
from databases_conn import Config


def update_config_data():
    '''read config data from MySQL

    :return: asset_list, asset_dic, tags_ids_dic
    '''

    print('>>>>>>>>>>>> updating config data')

    # define database configuration parameters
    db_info = {}
    db_info.update({'host': "192.168.21.134"})
    db_info.update({'port': 8086})
    db_info.update({'username': "root"})
    db_info.update({'password': "sbrQp10"})
    db_info.update({'database': "VIB_DB"})

    # create a connection to MySQL database
    db1 = databases_conn.DBmysql(Config.mysql)

    # get config data from MySQL database
    asset_list = db1.get_vib_asset_list()
    asset_dic = db1.get_vib_asset_dic()
    tags_ids_dic = db1.get_vib_tags_id_dic()

    # return config data
    return asset_list, asset_dic, tags_ids_dic


def check_for_new_tdw(asset, axis):
    """
    Get last two event_changes
    Verify to event_changes are coming
    And check that the first one doesnt have FFT

    :param asset: (Sensor name)
    :param axis: (X or Z)
    :return: True/False
    """
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print('>>>>>>>>>>>> process_trigger')
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

    # Initialize trigger in False
    p_trigger = False
    even_change_id = 0
    # print("influx config")
    # print(Config.influx)
    # create an instance of DBinflux
    db1 = databases_conn.DBinflux(config=Config.influx)

    # Build the field name from the axis (X or Z)
    select_field = "WF___{}_EVT_CHG_ID".format(axis)
    where_field = "WF___{}_FFT".format(axis)
    wf_event_id = "WF___{}_EVTID".format(axis) 

    # query to get the first two event ids of a time domain waveform without fft
    sql = "SELECT {} FROM {} WHERE {} = -1.0 AND {} <> '' ORDER BY time".format(select_field, asset, where_field, wf_event_id)
    # print(sql)
    # Execute query
    datasets_dic = db1.query(sql)
    print(datasets_dic)
    # get the pandas dataframe out of the query result
    datasets_pdf = datasets_dic[asset]

    # Get a number of rows and columns of the query result
    rows, cols = datasets_pdf.shape
    print('{}, {}'.format(rows, cols))

    # if there is at least one whole waveform
    # set trigger and copy the first whole waveform
    if rows > 1:
        p_trigger = True
        even_change_id = datasets_pdf[select_field][0]
    else:
        p_trigger = False
        even_change_id = ''
        print('>>>>>> No changes<<<<<<<<<')

    # return trigger and first event id
    return p_trigger, even_change_id

# *********************************************************************************
# **************************new functions *******************************************************


def read_acc_tdw(asset_name, event_id, axis='X'):
    """
    It returns a data frame with the Influxdb reading for the specific axis

    :param asset_name: string
    :param event_id: string
    :param axis: string
    :return: pandas data frame
    """

    # Initialization

    _timestamp = 'time'
    fft = 'WF___{}_FFT'.format(axis)
    fft_red = 'WF___{}_FFT_RED'.format(axis)
    freq = 'WF___{}_FREQ'.format(axis)
    freq_red = 'WF___{}_FREQ_RED'.format(axis)
    tdw = 'WF___{}_TDW'.format(axis)
    wf_evt_id = 'WF___{}_EVTID'.format(axis)
    evt_id = "'{}'".format(event_id)

    # create an instance of DBinflux
    db1 = databases_conn.DBinflux(config=Config.influx)

    # sql = "select * from " + asset_name
    sql = "select {}, {} from {} where {} = {} order by time".format(_timestamp, tdw, asset_name, wf_evt_id, evt_id)
    binds = {}

    # Execute query
    datasets_dic = db1.query(sql)

    # Get pandas data frame
    acc_tdw_pdf = datasets_dic[asset_name]

    return acc_tdw_pdf


def pdf_to_influxdb(process_pdf, asset_name):
    """
    It writes the data frame to Influxdb
    :param process_pdf:
    :param asset_name:
    :return:
    """
    # create an instance of DBinflux
    db1 = databases_conn.DBinflux(config=Config.influx)

    # write data frame to influx database
    db1.write_points(pdf=process_pdf, meas=asset_name)  # , batch_size=batch_size

    print('>>>>>>> data_process done')


def get_precess_pdf(acc_tdw_pdf, framerate, window='hanning'):
    """
    It returns the pandas data frame of the acceleration wave
    :param acc_tdw_pdf:
    :param framerate:
    :param window:
    :return: pandas data frame
    """
    acceleration_fft = 'WF___{}_FFT'.format(axis)
    acceleration_freq = 'WF___{}_FREQ'.format(axis)
    velocity_fft = 'WF___{}_FFT_V'.format(axis)
    velocity_freq = 'WF___{}_FREQ_V'.format(axis)
    velocity_tdw = 'WF___{}_TDW_V'.format(axis)

    acceleration_tdw_red = 'WF___{}_FFT_RED'.format(axis)
    acceleration_fft_red = 'WF___{}_FFT_RED'.format(axis)
    acceleration_freq_red = 'WF___{}_FREQ_RED'.format(axis)
    velocity_tdw_red = 'WF___{}_TDW_V_RED'.format(axis)
    velocity_fft_red = 'WF___{}_FFT_V_RED'.format(axis)
    velocity_freq_red = 'WF___{}_FREQ_V_RED'.format(axis)



    # create  acc_wave and spectrum
    acc_wave = thinkdsp.Wave(ys=acc_tdw_pdf[tdw], ts=np.linspace(0, 1, 100), framerate=framerate)
    vel_tdw = fft_eng.integrate(acc_wave)
    acc_spectrum = fft_eng.get_spectrum(wave=acc_wave, window=window)
    vel_spectrum = fft_eng.get_spectrum(wave=vel_wave, window=window)

    acc_tdw_red = get_signal_red_version(acc_wave.ys)
    acc_fft_red = get_signal_red_version(acc_spectrum.amps)
    vel_tdw_red = get_signal_red_version(vel_tdw)
    vel_fft_red = get_signal_red_version(vel_spectrum.amps)

    # create dictionary to use on pandas data frame creation
    final_dic = {
                acceleration_fft: acc_spectrum.amps, acceleration_freq: acc_spectrum.fs,
                velocity_fft: vel_spectrum.amps, velocity_freq: vel_spectrum.fs,
                velocity_tdw: vel_tdw,
                acceleration_tdw_red: acc_tdw_red, acceleration_fft_red: acc_fft_red,
                acceleration_freq_red: acc_spectrum.fs,
                velocity_tdw_red: vel_tdw_red, velocity_fft_red: vel_fft_red,
                velocity_freq_red: vel_spectrum.fs
                }

    # convert pandas dataframe
    # final_pdf = pd.DataFrame(final_dic, index=pdf_wave.index[:len(spectrum)])
    final_pdf = pd.DataFrame(final_dic, index=pdf_wave.index)
    return final_pdf


def process(asset_name, event_id, axis='X'):


    acc_tdw_pdf = read_acc_tdw(asset_name, event_id, axis='X')

    process_pdf = get_precess_pdf(acc_tdw_pdf, framerate, window='hanning')

    pdf_to_influxdb(process_pdf, asset_name)




def get_vel_tdw_pdf(acc_tdw_pdf):


    pass

def get_vel_fft_pdf(acc_fft_pdf):
    pass


def get_signal_red_version(signal):
    """
    TODO
    :param signal:
    :return:
    """


    return signal

# ***********************************************************************************
# **********************************************************************************


def data_process(asset_name, event_id, axis='X'):
    # Initialization

    _timestamp = 'time'
    fft = 'WF___{}_FFT'.format(axis)
    fft_red = 'WF___{}_FFT_RED'.format(axis)
    freq = 'WF___{}_FREQ'.format(axis)
    freq_red = 'WF___{}_FREQ_RED'.format(axis)
    tdw = 'WF___{}_TDW'.format(axis)
    wf_evt_id = 'WF___{}_EVTID'.format(axis)
    evt_id = "'{}'".format(event_id)

    # create an instance of DBinflux
    db1 = databases_conn.DBinflux(config=Config.influx)

    # sql = "select * from " + asset_name
    sql = "select {}, {} from {} where {} = {} order by time".format(_timestamp, tdw, asset_name, wf_evt_id, evt_id)
    binds = {}

    # Execute query
    datasets_dic = db1.query(sql)

    # Get pandas data frame
    pdf_wave = datasets_dic[asset_name]

    # create  wave and spectrum
    wave = thinkdsp.Wave(ys=pdf_wave[tdw], ts=np.linspace(0, 1, 100), framerate=100)
    spectrum = fft_eng.get_spectrum(wave=wave, window='hanning')
    spectrum_red = spectrum.copy()

    # create dictionary to use on pandas data frame creation
    spec_dic = {fft: spectrum.amps, freq: spectrum.fs,
                fft_red: spectrum_red.amps, freq_red: spectrum_red.fs}

    # convert pandas dataframe
    pdf_spec = pd.DataFrame(spec_dic, index=pdf_wave.index[:len(spectrum)])
    batch_size = len(spectrum)

    pdf_wave.index.astype(np.int64)

    # write data frame to influx database
    db1.write_points(pdf=pdf_spec, meas=asset_name)  #, batch_size=batch_size
    print('>>>>>>> spectrum dataframe: {}'.format(pdf_spec))
    # print('>>>>>>> spectrum dataframe: {}'.format(pdf_spec['time'])
    print('>>>>>>> data_process done')


def get_axis_list(asset_name):
    """

    :param asset_name:
    :return:
    """
    axis_list = ["X", "Z"]
    return axis_list


def init():
    '''

    :return:
    '''
    print('initializing config data')

    # TODO


def main():
    ''' execute main code '''
    print('main ran!')

    # creating global variables
    asset_dic = {}
    asset_list = []
    tags_ids_dic = {}
    

    # update config data the first time
    asset_list, asset_dic, tags_ids_dic = update_config_data()
    print('asset_list >> {}'.format(asset_list))
    print('asset_dic >> {}'.format(asset_dic))
    print('tags_ids_dic >> {}'.format(tags_ids_dic))

    #=========================
    # Redis DB Initialization
    # =========================
    # rt_redis_data = RedisDB()

    # Connect to Redis DB
    # rt_redis_data.open_db()

    while True:
        print('>>>>>>>>>>>> while true cycle')

        # update real time data
        # Read Apply changes status (Reload Status) from Redis
        reload_status = "0"  #databases_conn.redis_get_value("rt_control:reload:fft")
        # print("APPLY CHANGES STATUS: %s" % reload_status)

        # if "Apply Changes" is set
        if reload_status == "1":
            # update config data once again
            asset_list, asset_dic, tags_ids_dic = update_config_data()

            # Reset Apply Changes flag
            # databases_conn.redis_set_value("rt_control:reload:fft", str(0))
            print("RESETTING APPLY CHANGES FLAG")

        # scan all assets for the current database
        for asset in asset_list:
            
            # Get axis list
            axis_list = get_axis_list(asset_name=asset)

            # scan all axises for the current asset
            for axis in axis_list:

                # check for new time domain waveforms without processing
                trigger, even_change_id = check_for_new_tdw(asset=asset, axis=axis)

                # if a new time domain waveform is ready to process
                if trigger:
                    print("trigger: {}".format(trigger))

                    # Run data process function to get the FFT of the TDW for the event change ID
                    data_process(asset_name=asset, event_id=even_change_id, axis=axis)
                    print('>>>>>> let"s go processing \tasset: {}, axis: {}'.format(asset, axis))
                    # break

        print('time: {}'.format(time.time()))
        # wait for 30 second
        time.sleep(30)



'''================================================'''

if __name__ == "__main__":
    '''execute only if run as a main script'''

    # initialization function
    # init()

    main()


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
