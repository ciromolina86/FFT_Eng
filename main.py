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
import databases_conn
from databases_conn import Config
from databases_conn import DBinflux
from databases_conn import DBmysql
from redisdb import RedisDB


def update_config_data():
    '''read config data from MySQL

    :return: asset_list, asset_dic, tags_ids_dic
    '''
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print('>>>>>>>>>>>> update config data')
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

    # create a connection to MySQL database
    db1 = DBmysql(Config.mysql)

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

    # create influxdb object
    db1 = DBinflux(config=Config.influx)

    # Build the field name from the axis (X or Z)
    select_field = "WF___{}_EVT_CHG_ID".format(axis)
    where_field = "WF___{}_FFT".format(axis)
    wf_event_id = "WF___{}_EVTID".format(axis) 

    # query to get the first two event ids of a time domain waveform without fft
    sql = "SELECT {} FROM {} WHERE {} = -1.0 AND {} <> '' ORDER BY time".format(select_field, asset, where_field, wf_event_id)

    # Execute query
    datasets_dic = db1.query(sql)

    # get the pandas dataframe out of the query result
    datasets_pdf = datasets_dic[asset]

    # Get a number of rows and columns of the query result
    rows, cols = datasets_pdf.shape

    # if there is at least one whole waveform
    # set trigger and copy the first whole waveform
    if rows > 1:
        p_trigger = True
        even_change_id = datasets_pdf[select_field][0]
        print('>>>>>> New waveform <<<<<<<<<')
    else:
        p_trigger = False
        even_change_id = ''
        print('>>>>>> No changes <<<<<<<<<')

    # return trigger and first event id
    return p_trigger, even_change_id

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
    tdw_pdf = datasets_dic[asset_name]

    return tdw_pdf

def get_process_pdf(tdw_pdf, framerate, acc = True, window='hanning', axis='X'):
    """
    It returns the pandas data frame of the acceleration wave
    :param tdw_pdf: pandas dataframe
    :param framerate: real
    :param window: string
    :return: pandas data frame
    """
    tdw = 'WF___{}_TDW'.format(axis)
    acceleration_tdw = 'WF___{}_TDW'.format(axis)
    acceleration_fft = 'WF___{}_FFT'.format(axis)
    acceleration_freq = 'WF___{}_FREQ'.format(axis)
    velocity_fft = 'WF___{}_FFT_V'.format(axis)
    velocity_tdw = 'WF___{}_TDW_V'.format(axis)

    acceleration_tdw_red = 'WF___{}_FFT_RED'.format(axis)
    acceleration_fft_red = 'WF___{}_FFT_RED'.format(axis)
    acceleration_freq_red = 'WF___{}_FREQ_RED'.format(axis)
    velocity_tdw_red = 'WF___{}_TDW_V_RED'.format(axis)
    velocity_fft_red = 'WF___{}_FFT_V_RED'.format(axis)

    # init
    final_pdf_list = []

    # compute tdw duration (in time)
    duration = len(tdw_pdf[tdw]) / framerate
    N = len(tdw_pdf[tdw])

    # create  wave from pdf
    wave = thinkdsp.Wave(ys=tdw_pdf[tdw], ts=np.linspace(0, duration, N), framerate=framerate)

    # if collecting acceleration
    if acc:
        # TODO: acc_tdw = wave.copy()
        vel_wave = fft_eng.integrate(wave)
        vel_tdw = vel_wave.ys
        acc_spectrum = fft_eng.get_spectrum(wave=wave, window=window)
        vel_spectrum = fft_eng.get_spectrum(wave=vel_wave, window=window)

        # get reduced signal
        # TODO
        acc_tdw_red = get_signal_red_version(wave.ys)
        acc_fft_red = get_signal_red_version(acc_spectrum.amps)
        vel_tdw_red = get_signal_red_version(vel_tdw)
        vel_fft_red = get_signal_red_version(vel_spectrum.amps)

        # create dictionary to use on pandas data frame creation
        dic_list = [
            {acceleration_fft: acc_spectrum.amps},
            {velocity_tdw: vel_tdw},
            {velocity_fft: vel_spectrum.amps},

            {acceleration_tdw_red: acc_tdw_red},
            {acceleration_fft_red: acc_fft_red},
            {velocity_tdw_red: vel_tdw_red},
            {velocity_fft_red: vel_fft_red},
            {acceleration_freq: acc_spectrum.fs}
        ]

    # if collecting velocity
    else:
        acc_wave = fft_eng.derivate(wave)
        acc_tdw = acc_wave.ys
        acc_spectrum = fft_eng.get_spectrum(wave=acc_wave, window=window)
        vel_spectrum = fft_eng.get_spectrum(wave=wave, window=window)

        # get reduced signal
        acc_tdw_red = get_signal_red_version(acc_tdw)
        acc_fft_red = get_signal_red_version(acc_spectrum.amps)
        vel_tdw_red = get_signal_red_version(wave.ys)
        vel_fft_red = get_signal_red_version(vel_spectrum.amps)

        # create list of dictionary to use on pandas data frame creation
        dic_list = [
            {velocity_fft: vel_spectrum.amps},
            {acceleration_tdw: acc_tdw},
            {acceleration_fft: acc_spectrum.amps},

            {acceleration_tdw_red: acc_tdw_red},
            {acceleration_fft_red: acc_fft_red},
            {velocity_tdw_red: vel_tdw_red},
            {velocity_fft_red: vel_fft_red},
            {acceleration_freq: acc_spectrum.fs}
        ]

    # Create list of pdf from list of dictionaries

    for d in dic_list:
        for key, value in d.iteritems():
            final_pdf_list.append(pd.DataFrame(d, index=tdw_pdf.index[:len(value)]))

    return final_pdf_list

def pdf_to_influxdb(process_pdf_list, asset_name):
    """
    It writes the data frame to Influxdb
    :param process_pdf: list of pandas dataframe
    :param asset_name: string
    :return:
    """
    # create an instance of DBinflux
    db1 = DBinflux(config=Config.influx)

    for process_pdf in process_pdf_list:
        # write data frame to influx database
        db1.write_points(pdf=process_pdf, meas=asset_name)

def process(asset_name, event_id, framerate, axis='X'):
    """
    Process data
    :param asset_name:
    :param event_id:
    :param axis:
    :return:
    """

    # Get the time domain waveform in a python data frame
    tdw_pdf = read_acc_tdw(asset_name, event_id, axis=axis)

    # Get a python data frame per column that we need as el list
    process_pdf_list = get_process_pdf(tdw_pdf, framerate, window='hanning', axis=axis)

    # write to influxdb all the pandas data frame in a provided list
    pdf_to_influxdb(process_pdf_list, asset_name)

def get_signal_red_version(signal):
    """
    TODO
    :param signal:
    :return:
    """


    return signal

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

    # write dataframe to influxdb
    db1.write_points(pdf=pdf_spec, meas=asset_name)
    # print('>>>>>>> spectrum dataframe: {}'.format(pdf_spec))
    print('>>>>>>> data_process done')

def get_axis_list(asset_name):
    """

    :param asset_name:
    :return:
    """
    # create hardcoded axis list
    axis_list = ["X", "Z"]

    # return axis list
    return axis_list

def init():
    '''

    :return:
    '''
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print('>>>>>>>>>>>> running INIT! ')
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

def main():
    ''' execute main code '''
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print('>>>>>>>>>>>> running MAIN CODE! ')
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

    #=========================
    # Config Data Initialization
    # =========================
    # creating global variables
    asset_dic = {}
    asset_list = []
    tags_ids_dic = {}

    # update config data the first time
    asset_list, asset_dic, tags_ids_dic = update_config_data()
    # print('asset_list >> {}'.format(asset_list))
    # print('asset_dic >> {}'.format(asset_dic))
    # print('tags_ids_dic >> {}'.format(tags_ids_dic))

    #=========================
    # Redis DB Initialization
    # =========================
    rt_redis_data = RedisDB()

    # Connect to Redis DB
    rt_redis_data.open_db()

    while True:
        # print('>>>>>>>>>>>> while true cycle')

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

            # get sampling frequency internalTagID
            for tag, id in tags_ids_dic.get(asset):
                if tag == 'CFG___FS':
                    framerate_id_str = str(id)

            # get real time sampling frequency value
            framerate_ts, framerate_current_value = databases_conn.getinrtmatrix(framerate_id_str)

            # scan all axises for the current asset
            for axis in axis_list:

                # check for new time domain waveforms without processing
                trigger, even_change_id = check_for_new_tdw(asset=asset, axis=axis)

                # if a new time domain waveform is ready to process
                if trigger:
                    # Run data process function to get the FFT of the TDW for the event change ID
                    # data_process(asset_name=asset, event_id=even_change_id, axis=axis)

                    # Data Processing
                    process(asset_name=asset, event_id=even_change_id, framerate=framerate_current_value, axis=axis)
                    # print('>>>>>> let"s go processing \tasset: {}, axis: {}'.format(asset, axis))
                    # break

        print('cycle time: {}'.format(np.int64(time.time()*1000)))
        # wait for 30 second
        time.sleep(30)



'''================================================'''

if __name__ == "__main__":
    '''execute only if run as a main script'''

    # run main code
    main()