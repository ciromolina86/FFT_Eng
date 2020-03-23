
from ThinkX import thinkdsp
import numpy as np
import pandas as pd
import fft_eng
import time
import databases_conn
from databases_conn import Config
from databases_conn import DBinflux
from databases_conn import DBmysql
from databases_conn import VibModel
from redisdb import RedisDB
import influx_helper_sde
from collections import deque


def update_config_data():
    """
    Read config data from MySQL
    :return: Asset_list, asset_dic, tags_ids_dic
    """

    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print('>>>> update config data >>>>')
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

    # create a connection to MySQL database
    db1 = DBmysql(Config.mysql)

    # get config data from MySQL database
    asset_list = db1.get_vib_asset_list()
    asset_dic = db1.get_vib_asset_dic()
    tags_ids_dic = db1.get_vib_tags_id_dic()

    # return config data
    return asset_list, asset_dic, tags_ids_dic


def check_for_new_tdw(asset, n_tdw, asset_dic_queue, axis_list):
    """
    Get last two event_changes
    Verify to event_changes are coming
    And check that the first one doesnt have FFT

    :param asset: (Sensor name)
    :param n_tdw:
    :param asset_dic_queue:
    :param axis_list: (X or Z)
    :return: True/False
    """

    # Initialization
    p_trigger_list = []
    event_chg_id_list = []
    tdw_pdf_list = []

    # create influxdb object
    db1 = DBinflux(config=Config.influx)

    for axis in axis_list:
        print("[INFO] Checking for new TDW (Asset: %s, Axis: %s)" % (asset, axis))

        # Initialization per Axis
        event_chg_id_tmp = ''

        # Build the field name from the axis (X or Z)
        select_field = "WF___EVT_CHG_ID"
        where_field = "WF___{}_FFT".format(axis)

        # query to get the time domain waveform without fft
        sql = "SELECT {} " \
              "FROM (SELECT {}, {} FROM {} FILL(-999.99)) " \
              "WHERE ({} = -999.99 AND {} <> '-999.99')  " \
              "ORDER BY time".format(select_field, where_field, select_field, asset, where_field, select_field)

        # Execute query
        datasets_dic = db1.query(sql)

        # get the pandas dataframe out of the query result
        if bool(datasets_dic):
            # Get dataframe
            datasets_pdf = datasets_dic[asset]

            # Get a number of rows and columns of the query result
            rows, cols = datasets_pdf.shape

            # if there is at least one whole waveform
            # set trigger and copy the first whole waveform
            if rows >= 1:
                # valid rows initialization
                validation_completed = False

                # Perform WF___EVT_CHG_ID and WF___EVTID match validation
                for row in range(rows):
                    # Rows local initialization
                    validation_completed = False

                    # Get tmp event_change_id
                    event_chg_id_tmp = datasets_pdf[select_field][row]

                    # Check if event id was already processed
                    # print("[INFO] event_chg_id_tmp: %s" % event_chg_id_tmp)
                    # print("[INFO] asset_dic_queue: %s" % asset_dic_queue)
                    processed_event_id_queue = asset_dic_queue.get(asset)
                    if str(event_chg_id_tmp) not in processed_event_id_queue:
                        # Build the query main parameters
                        select_field1 = "WF___EVT_CHG_ID"
                        select_field2 = "WF___EVTID"
                        where_field = "WF___EVT_CHG_ID"
                        event_chg_id_tmp_field = "'{}'".format(event_chg_id_tmp)

                        # query to get the WF___EVTID base on WF___EVT_CHG_ID and compare (both values must be the same)
                        sql_validation = "SELECT {}, {} FROM {} WHERE {} = {}".format(select_field1, select_field2, asset, where_field, event_chg_id_tmp_field)

                        # Execute query
                        datasets_dic_validation = db1.query(sql_validation)

                        # get the pandas dataframe out of the query result
                        datasets_pdf_validation = datasets_dic_validation[asset]

                        # Get WF___EVT_CHG_ID and WF___EVTID from dataframe
                        wf_event_chg_id_tmp = datasets_pdf_validation[[select_field1]].values
                        wf_event_id_tmp = datasets_pdf_validation[[select_field2]].values

                        # If WF___EVT_CHG_ID and WF___EVTID matched, validation process was met for the current Waveform
                        wf_event_chg_id_tmp_list = wf_event_chg_id_tmp.tolist()
                        wf_event_id_tmp_list = wf_event_id_tmp.tolist()

                        for evt_id, evt_chg_id in zip(wf_event_id_tmp_list, wf_event_chg_id_tmp_list):
                            if evt_chg_id == evt_id:
                                # validation completed/met for current event id
                                validation_completed = True
                                print("[INFO] Asset: %s, Axis: %s. EVENT_ID is the same than EVENT_CHG_ID" % (asset, axis))
                                break

                        # If validation is completed break from loop
                        if validation_completed:
                            break

                if validation_completed:
                    # Set p_trigger and event_chg_id
                    p_trigger = True
                    event_chg_id = event_chg_id_tmp
                    print('[INFO] Asset: %s, Axis: %s. There is new TDW available' % (asset, axis))
                else:
                    p_trigger = False
                    event_chg_id = ''

            else:
                # Set p_trigger and event_chg_id
                p_trigger = False
                event_chg_id = ''
                print('[INFO] Asset: %s, Axis: %s. There is not new TDW available' % (asset, axis))

        else:
            p_trigger = False
            event_chg_id = ''
            print('[INFO] Asset: %s, Axis: %s. There is not new TDW available (Dic empty)' % (asset, axis))

        if p_trigger:
            # Get the time domain waveform in a python data frame
            tdw_pdf = read_acc_tdw(asset, event_chg_id, axis=axis)

            # Get a number of rows and columns of the TDW dataframe
            tdw_pdf_rows, tdw_pdf_cols = tdw_pdf.shape

            print("[INFO] Asset: %s, Axis: %s. Dataframe rows: %s, N_TDW: %s" % (asset, axis, int(tdw_pdf_rows), int(n_tdw)))

            # Check if TDW has the right number of elements
            if int(tdw_pdf_rows) == int(n_tdw):
                # Check if Time Wave Form (TDW) exist on dataframe
                axis_tdw_col = "WF___{}_TDW".format(axis)
                if axis_tdw_col in tdw_pdf.columns:
                    p_trigger = True
                else:
                    p_trigger = False
                    print("[WARN] Asset: %s, Axis: %s. There is not TDW on the retrieved data" % (asset, axis))

            else:
                p_trigger = False
                print("[WARN] Asset: %s, Axis: %s. The TDW does not have the right amount of elements: %s" % (asset, axis, int(n_tdw)))
        else:
            p_trigger = False
            tdw_pdf = None

        # Update result from validation per Axis
        p_trigger_list.append(p_trigger)
        event_chg_id_list.append(event_chg_id)
        tdw_pdf_list.append(tdw_pdf)

    return p_trigger_list, event_chg_id_list, tdw_pdf_list


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
    tdw = 'WF___{}_TDW'.format(axis)
    wf_evt_id = 'WF___EVTID'
    evt_id = "'{}'".format(event_id)

    # create an instance of DBinflux
    db1 = databases_conn.DBinflux(config=Config.influx)

    # sql = "select * from " + asset_name
    sql = "select {}, {}, {} from {} where {} = {} order by time".format(_timestamp, tdw, wf_evt_id, asset_name, wf_evt_id, evt_id)

    # Execute query
    datasets_dic = db1.query(sql)

    # Get pandas data frame
    tdw_pdf = datasets_dic[asset_name]

    return tdw_pdf


def get_downsampled_data_ts(input_mtx_ts, input_mtx, max_datapoints):
    """
    DOWN-SAMPLING using Numpy matrix with timestamp
    :param input_mtx_ts: matrix
    :param input_mtx: matrix
    :param max_datapoints: int
    :return:
    """

    # Training Input Reduced for Overview using LTTB
    row_count, column_count = fft_eng.get_col_and_rows_numpy_array(input_mtx)
    downsampled_mtx, downsampled_mtx_ts = fft_eng.dataset_downsampling_lttb_ts(np, input_mtx, input_mtx_ts,
                                                                               max_datapoints,
                                                                               row_count, column_count)

    return downsampled_mtx, downsampled_mtx_ts


def get_downsampled_data(input_mtx, max_datapoints):
    """
    DOWN-SAMPLING using Numpy matrix without timestamp
    :param input_mtx: matrix
    :param max_datapoints: int
    :return:
    """

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
    # Create query elements
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
    tdw_n = len(tdw_pdf[tdw_name])

    # create  wave from pdf
    tdw = thinkdsp.Wave(ys=tdw_pdf[tdw_name], ts=np.linspace(0, tdw_duration, tdw_n), framerate=framerate)

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
    acc_fft_pdf = pd.DataFrame({acc_fft_name: acc_fft.amps, freq_name: acc_fft.fs}, index=tdw_pdf.index[:len(acc_fft)])
    vel_fft_pdf = pd.DataFrame({vel_fft_name: vel_fft.amps, freq_name: vel_fft.fs}, index=tdw_pdf.index[:len(vel_fft)])

    # ============================================================================
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

    # ============================================================================
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

    # return list of pandas data-frame
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


def process(asset_name, framerate, tdw_pdf_list, axis_list):
    """
    Process data
    :param asset_name:
    :param framerate:
    :param tdw_pdf_list:
    :param axis_list:
    :return:
    """

    # Process both axis
    for axis, tdw_pdf in zip(axis_list, tdw_pdf_list):
        # Get a python data frame per column that we need as el list
        process_pdf_list = get_process_pdf(tdw_pdf, framerate, window='hanning', axis=axis, red_rate=0.3)

        # write to influxdb all the pandas data frame in a provided list
        pdf_to_influxdb(process_pdf_list, asset_name)

    return True


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
    """

    :return:
    """

    print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    print("<<<  INITIALIZATION - FIRST RUN STARTED <<<<<<<<<<<<<<<<<<<<<<<<<<<")
    print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")

    # Initialization
    vib_model = VibModel()
    rt_redis_data = RedisDB()
    rt_redis_data.open_db()
    max_points = {}
    asset_dic_queue = {}
    framerate_id_str = None
    n_tdw_id_str = None
    max_points.update({'WF___X_TDW': 1000})
    cycle_sleep = 10    # Seconds

    # Get vibration config model
    config_model = vib_model.model_mysql

    # Get Asset List
    asset_list = config_model.keys()

    print("[INFO] Asset loaded to be processed: %s" % asset_list)

    # Get tag ids for frame rate (FS), number of samples on wave form (N_TDW) and create asset_dic_queue
    for asset in asset_list:
        # get sampling frequency and N_TDW internalTagID
        framerate_id_str = str(config_model[asset]['CFG']['FS']['internalTagID'])
        n_tdw_id_str = str(config_model[asset]['CFG']['N_TDW']['internalTagID'])

        # Create dictionary with processed event id queue per asset
        processed_event_id_queue = deque(maxlen=100)
        asset_dic_queue.update({asset: processed_event_id_queue})

    # Get real time sampling frequency and N_TDW values
    framerate_ts, framerate_current_value = databases_conn.getinrtmatrix(rt_redis_data, framerate_id_str)
    n_tdw_ts, n_tdw_current_value = databases_conn.getinrtmatrix(rt_redis_data, n_tdw_id_str)

    print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    print("<<<  INITIALIZATION - FIRST RUN FINISHED <<<<<<<<<<<<<<<<<<<<<<<<<<")
    print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")

    while True:
        print("##################################################")
        print("[INFO] Cycle started")
        # Start Cycle Timestamp
        start_cycle_ts = int(round(time.time() * 1000))

        # Read Apply changes status (Reload Status) from Redis
        reload_status = databases_conn.redis_get_value(rt_redis_data, "rt_control:reload:fft")

        # if "Apply Changes" is set
        if reload_status == "1":
            print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            print("<<<  INITIALIZATION - APPLY CHANGES STARTED <<<<<<<<<<<<<<<<<<<<<<<")
            print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            # Update vibration config model
            vib_model.update_model()

            # Get vibration config model
            config_model = vib_model.model_mysql

            # Get Asset List
            asset_list = config_model.keys()

            # Get tag ids for frame rate (FS), number of samples on wave form (N_TDW) and create asset_dic_queue
            for asset in asset_list:
                # get sampling frequency and N_TDW internalTagID
                framerate_id_str = str(config_model[asset]['CFG']['FS']['internalTagID'])
                n_tdw_id_str = str(config_model[asset]['CFG']['N_TDW']['internalTagID'])

                # Create dictionary with processed event id queue per asset
                processed_event_id_queue = deque(maxlen=100)
                asset_dic_queue.update({asset: processed_event_id_queue})

            # Get real time sampling frequency and N_TDW values
            framerate_ts, framerate_current_value = databases_conn.getinrtmatrix(rt_redis_data, framerate_id_str)
            n_tdw_ts, n_tdw_current_value = databases_conn.getinrtmatrix(rt_redis_data, n_tdw_id_str)

            # Reset Apply Changes flag
            databases_conn.redis_set_value(rt_redis_data, "rt_control:reload:fft", str(0))

            print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            print("<<<  INITIALIZATION - APPLY CHANGES FINISHED <<<<<<<<<<<<<<<<<<<<<<")
            print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")

        # Check for new wave form and process FFT per asset
        for asset in asset_list:
            # Get axis list
            axis_list = get_axis_list(asset_name=asset)

            # Check if Sample Frequency is not None
            if (framerate_current_value is None) or (framerate_current_value == 0):
                print("[WARN] There is not sample frequency defined (None or 0)")

            else:
                # check for new time domain waveforms without processing
                trigger_list, even_change_id_list, tdw_pdf_list = check_for_new_tdw(asset=asset, n_tdw=n_tdw_current_value, asset_dic_queue=asset_dic_queue,
                                                                                    axis_list=axis_list)

                # if a new time domain waveform is ready to process
                if False not in trigger_list:
                    print('-----------------------------------------------------------------')
                    print('[INFO] FFT processing started')
                    print('[INFO] Asset: {}'.format(asset))
                    print('[INFO] Even_change_id: {}'.format(even_change_id_list))
                    print('[INFO] Frame-rate: {}'.format(framerate_current_value))
                    print('[INFO] Axis: {}'.format(axis_list))

                    # Data Processing
                    process(asset_name=asset, framerate=framerate_current_value, tdw_pdf_list=tdw_pdf_list, axis_list=axis_list)

                    print('[INFO] FFT processing ended')
                    print('-----------------------------------------------------------------')

                    # Execute influx index adder after the Asset has been processed (Add EVT_ID Index)
                    asset_dic_queue = influx_helper_sde.influx_index_adder(asset_name=asset, event_ids_list=even_change_id_list, asset_dic_queue=asset_dic_queue)

        # End Cycle Timestamp
        end_cycle_ts = int(round(time.time() * 1000))

        print("[INFO] Cycle time: %s sec" % (end_cycle_ts - start_cycle_ts))
        print("[INFO] Cycle ended")
        print("##################################################")

        # wait for cycle_sleep second
        time.sleep(cycle_sleep)


'''================================================'''

if __name__ == "__main__":
    '''execute only if run as a main script'''

    # run main code
    main()
