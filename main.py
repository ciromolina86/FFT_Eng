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
from databases_conn import VibModel
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

    # Initialization
    event_chg_id_tmp = ''

    # create influxdb object
    db1 = DBinflux(config=Config.influx)

    # Build the field name from the axis (X or Z)
    select_field = "WF___EVT_CHG_ID"
    where_field = "WF___{}_FFT".format(axis)

    # query to get the first two event ids of a time domain waveform without fft
    sql = "SELECT {} " \
          "FROM (SELECT {}, {} FROM {} FILL(-999.99) ) " \
          "WHERE ({} = -999.99 AND {} <> '-999.99')  " \
          "ORDER BY time".format(select_field, where_field, select_field, asset, where_field, select_field)

    # Execute query
    datasets_dic = db1.query(sql)

    # get the pandas dataframe out of the query result
    datasets_pdf = datasets_dic[asset]

    # Get a number of rows and columns of the query result
    rows, cols = datasets_pdf.shape

    # if there is at least one whole waveform
    # set trigger and copy the first whole waveform
    if rows > 1:
        # Perform validation
        for row_index in range(rows):
            # Get tmp event_change_id
            event_chg_id_tmp = datasets_pdf[select_field][row_index]

            # Build the query main parameters
            select_field1 = "WF___EVT_CHG_ID"
            select_field2 = "WF___EVTID"
            where_field = "WF___EVT_CHG_ID"

            # query to get the WF___EVTID base on WF___EVT_CHG_ID and compare (both values must be the same)
            sql_validation = "SELECT {}, {} FROM {} WHERE {} = \'{}\'".format(select_field1, select_field2, asset, where_field, event_chg_id_tmp)

            # Execute query
            datasets_dic_validation = db1.query(sql_validation)

            # get the pandas dataframe out of the query result
            datasets_pdf_validation = datasets_dic_validation[asset]

            # Get WF___EVT_CHG_ID and WF___EVTID from dataframe
            wf_event_chg_id_tmp = datasets_pdf_validation[select_field1].values
            wf_event_id_tmp = datasets_pdf_validation[select_field2].values

            # If WF___EVT_CHG_ID and WF___EVTID matched, validation process was met for the current Waveform
            if wf_event_chg_id_tmp == wf_event_id_tmp:
                # Validation met
                break

        # Set p_trigger and event_chg_id
        p_trigger = True
        event_chg_id = event_chg_id_tmp
        print('>>>>>> New waveform <<<<<<<<<')
    else:
        # Set p_trigger and event_chg_id
        p_trigger = False
        event_chg_id = ''
        print('>>>>>> No changes <<<<<<<<<')

    # return trigger and first event id
    return p_trigger, event_chg_id


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
    freq = 'WF___FREQ'.format(axis)
    freq_red = 'WF___FREQ_RED'.format(axis)
    tdw = 'WF___{}_TDW'.format(axis)
    wf_evt_id = 'WF___EVTID'
    evt_id = "'{}'".format(event_id)

    # create an instance of DBinflux
    db1 = databases_conn.DBinflux(config=Config.influx)

    # sql = "select * from " + asset_name
    sql = "select {}, {}, {} from {} where {} = {} order by time".format(_timestamp, tdw, wf_evt_id, asset_name, wf_evt_id, evt_id)

    # Execute query
    datasets_dic = db1.query(sql)
    #print(datasets_dic)

    # Get pandas data frame
    tdw_pdf = datasets_dic[asset_name]

    return tdw_pdf


def get_downsampled_data_ts(input_mtx_ts, input_mtx, max_datapoints, field_name=''):
    '''DOWN-SAMPLING using Numpy matrix with timestamp

    :param input_mtx_ts: matrix
    :param input_mtx: matrix
    :param max_datapoints: int
    :param field_name: string
    :return:
    '''

    # Training Input Reduced for Overview using LTTB
    row_count, column_count = fft_eng.get_col_and_rows_numpy_array(input_mtx)
    downsampled_mtx, downsampled_mtx_ts = fft_eng.dataset_downsampling_lttb_ts(np, input_mtx, input_mtx_ts,
                                                                               max_datapoints,
                                                                               row_count, column_count)

    # return downsampled data
    return downsampled_mtx, downsampled_mtx_ts


def get_downsampled_data(input_mtx, max_datapoints, field_name=''):
    '''DOWN-SAMPLING using Numpy matrix without timestamp

    :param input_mtx: matrix
    :param max_datapoints: int
    :param field_name: string
    :return:
    '''

    # Get number of rows and columns
    row_count, column_count = fft_eng.get_col_and_rows_numpy_array(input_mtx)

    # Training Input Reduced for Overview using LTTB
    downsampled_mtx = fft_eng.dataset_downsampling_lttb(np, input_mtx, max_datapoints, row_count, column_count)

    # return downsampled data
    return downsampled_mtx


def get_process_pdf(tdw_pdf, framerate, red_rate=1.0, acc=True, window='hanning', axis='X'):
    """
    It returns the pandas data frame of the acceleration wave
    :param tdw_pdf: pandas dataframe
    :param framerate: real
    :param red_rate:
    :param acc:
    :param window: string
    :param axis:
    :return: pandas data frame
    """
    timestamp = 'time'
    tdw_name = 'WF___{}_TDW'.format(axis)
    acc_tdw_name = 'WF___{}_TDW'.format(axis)
    acc_fft_name = 'WF___{}_FFT'.format(axis)
    freq_name = 'WF___FREQ'.format(axis)
    vel_fft_name = 'WF___{}_FFT_V'.format(axis)
    vel_tdw_name = 'WF___{}_TDW_V'.format(axis)
    evtid_name = 'WF___EVTID'

    acc_tdw_red_name = 'WF___{}_TDW_RED'.format(axis)
    acc_fft_red_name = 'WF___{}_FFT_RED'.format(axis)
    freq_red_name = 'WF___FREQ_RED'.format(axis)
    vel_tdw_red_name = 'WF___{}_TDW_V_RED'.format(axis)
    vel_fft_red_name = 'WF___{}_FFT_V_RED'.format(axis)
    evtid_red_name = 'WF___EVTID_RED'

    # Fill with zero data read from influx with NaN values
    tdw_pdf[tdw_name].fillna(0, inplace=True)

    # compute tdw duration (in time)
    tdw_duration = len(tdw_pdf[tdw_name]) / framerate
    tdw_N = len(tdw_pdf[tdw_name])

    # create  wave from pdf
    tdw = thinkdsp.Wave(ys=tdw_pdf[tdw_name], ts=np.linspace(0, tdw_duration, tdw_N), framerate=framerate)

    # if collecting acceleration
    if acc:
        # create acceleration and velocity waveform
        acc_tdw = tdw.copy()
        vel_tdw = fft_eng.integrate(tdw)
    else:
        # create acceleration and velocity waveform
        vel_tdw = tdw.copy()
        acc_tdw = fft_eng.derivate(tdw)

    # create acceleration and velocity spectrum (FFT)
    acc_fft = fft_eng.get_spectrum(wave=acc_tdw, window=window)
    vel_fft = fft_eng.get_spectrum(wave=vel_tdw, window=window)

    # create acceleration and velocity result time domain waveform pandas dataframe
    acc_tdw_pdf = pd.DataFrame({acc_tdw_name: acc_tdw.ys}, index=tdw_pdf.index)
    vel_tdw_pdf = pd.DataFrame({vel_tdw_name: vel_tdw.ys}, index=tdw_pdf.index)

    # create acceleration and velocity result spectrum (FFT) pandas dataframe
    acc_fft_pdf = pd.DataFrame({acc_fft_name: acc_fft.amps,
                                freq_name: acc_fft.fs},
                               index=tdw_pdf.index[:len(acc_fft)])
    vel_fft_pdf = pd.DataFrame({vel_fft_name: vel_fft.amps,
                                freq_name: vel_fft.fs},
                               index=tdw_pdf.index[:len(vel_fft)])

    '''============================================================================'''
    # create matrix of acceleration and velocity time domain waveform to downsample
    # shape = (rows = N, cols = 3)
    tdw_mtx = np.array([acc_tdw_pdf[acc_tdw_name].values]).T
    tdw_mtx = np.append(tdw_mtx, np.array([vel_tdw_pdf[vel_tdw_name].values]).T, axis=1)

    # create matrix of time to downsample
    tdw_mtx_ts = np.array([acc_tdw_pdf.index], dtype=object).T

    # Get number of rows and columns
    tdw_mtx_row_count, tdw_mtx_column_count = fft_eng.get_col_and_rows_numpy_array(tdw_mtx)

    # get downsampled matrix of acceleration and velocity time domain waveform
    tdw_mtx_red, tdw_mtx_red_ts = get_downsampled_data_ts(input_mtx_ts=tdw_mtx_ts,
                                                          input_mtx=tdw_mtx,
                                                          max_datapoints=int(tdw_mtx_row_count*tdw_mtx_column_count*red_rate))

    # create pandas dataframe from downsampled acceleration and velocity spectra
    tdw_pdf_red = pd.DataFrame(tdw_mtx_red, columns=[acc_tdw_red_name, vel_tdw_red_name], index=pd.DatetimeIndex(np.array(tdw_mtx_red_ts)[:, 0]))


    '''============================================================================'''

    # create matrix of acceleration and velocity spectra to downsample
    # shape = (rows = N, cols = 3)
    fft_mtx = np.array([acc_fft_pdf[freq_name].values]).T
    fft_mtx = np.append(fft_mtx, np.array([acc_fft_pdf[acc_fft_name].values]).T, axis=1)
    fft_mtx = np.append(fft_mtx, np.array([vel_fft_pdf[vel_fft_name].values]).T, axis=1)

    # Get number of rows and columns
    fft_mtx_row_count, fft_mtx_column_count = fft_eng.get_col_and_rows_numpy_array(fft_mtx)

    # get downsampled matrix of acceleration and velocity spectra
    fft_mtx_red = get_downsampled_data(input_mtx=fft_mtx,
                                       max_datapoints=int(fft_mtx_row_count*fft_mtx_column_count*red_rate))

    # create pandas dataframe from downsampled acceleration and velocity spectra
    fft_pdf_red = pd.DataFrame(fft_mtx_red,
                               columns=[freq_red_name, acc_fft_red_name, vel_fft_red_name],
                               index=tdw_pdf.index[:len(fft_mtx_red)])

    # add the event id reduced column to the result pandas dataframe
    temp_pdf = pd.DataFrame({evtid_red_name: tdw_pdf[evtid_name][:len(fft_pdf_red)]},
                            index=tdw_pdf.index[:len(fft_mtx_red)])
    fft_pdf_red = pd.concat([fft_pdf_red, temp_pdf], axis=1)

    '''============================================================================'''

    # return list of pandas dataframe
    if acc:
        return [vel_tdw_pdf, tdw_pdf_red, acc_fft_pdf, vel_fft_pdf, fft_pdf_red]
    else:
        return [acc_tdw_pdf, tdw_pdf_red, acc_fft_pdf, vel_fft_pdf, fft_pdf_red]


def pdf_to_influxdb(process_pdf_list, asset_name):
    """
    It writes the data frame to Influxdb
    :param process_pdf_list: list of pandas dataframe
    :param asset_name: string
    :return:
    """
    # create an instance of DBinflux
    db1 = DBinflux(config=Config.influx)

    for process_pdf in process_pdf_list:
        # write data frame to influx database
        db1.write_points(pdf=process_pdf, meas=asset_name)
        # print(process_pdf.keys())


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
    process_pdf_list = get_process_pdf(tdw_pdf, framerate, window='hanning', axis=axis, red_rate=0.3)

    # write to influxdb all the pandas data frame in a provided list
    pdf_to_influxdb(process_pdf_list, asset_name)


def get_axis_list(asset_name):
    """

    :param asset_name:
    :return:
    """
    # create hardcoded axis list
    axis_list = ["X", "Z"]

    # return axis list
    return axis_list


def main():
    ''' execute main code '''
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print('>>>>>>>>>>>> running MAIN CODE! ')
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

    #=========================
    # Config Data Initialization
    # =========================
    vib_model = VibModel()

    #=========================
    # Downsampling Data Initialization
    # =========================
    max_points = {}
    max_points.update({'WF___X_TDW': 1000})

    # Get vibration config model
    config_model = vib_model.model_mysql

    # =========================
    # Redis DB Initialization
    # =========================
    rt_redis_data = RedisDB()

    # Connect to Redis DB
    rt_redis_data.open_db()

    while True:
        # update real time data
        # Read Apply changes status (Reload Status) from Redis
        reload_status = "0"  #databases_conn.redis_get_value("rt_control:reload:fft")
        print("APPLY CHANGES STATUS: %s" % reload_status)

        # if "Apply Changes" is set
        if reload_status == "1":
            # Update vibration config model
            vib_model.update_model()

            # Reset Apply Changes flag
            databases_conn.redis_set_value("rt_control:reload:fft", str(0))
            print("RESETTING APPLY CHANGES FLAG")

        # scan all assets for the current database
        for asset in config_model.keys():
            
            # Get axis list
            axis_list = get_axis_list(asset_name=asset)

            # get sampling frequency internalTagID
            framerate_id_str = str(config_model[asset]['CFG']['FS']['internalTagID'])

            # get real time sampling frequency value
            framerate_ts, framerate_current_value = databases_conn.getinrtmatrix(rt_redis_data, framerate_id_str)

            # Check if Sample Frequency is not None
            if (framerate_current_value is None) or (framerate_current_value == 0):
                print("[WARN] There is not sample frequency defined (None or 0)")

            else:
                # scan all axises for the current asset
                for axis in axis_list:

                    # check for new time domain waveforms without processing
                    trigger, even_change_id = check_for_new_tdw(asset=asset, axis=axis)

                    # if a new time domain waveform is ready to process
                    if trigger:
                         # Data Processing
                        print('asset: {}'.format(asset))
                        print('even_change_id: {}'.format(even_change_id))
                        print('framerate_current_value: {}'.format(framerate_current_value))
                        print('axis: {}'.format(axis))
                        process(asset_name=asset, event_id=even_change_id, framerate=framerate_current_value, axis=axis)

        # print actual time in milliseconds
        print('cycle time: {}'.format(np.int64(time.time()*1000)))

        # wait for 10 second
        time.sleep(10)


'''================================================'''

if __name__ == "__main__":
    '''execute only if run as a main script'''

    # run main code
    main()
