import mysql.connector
from influxdb import DataFrameClient
from influxdb import InfluxDBClient

import pandas as pd
import numpy as np
from ThinkX import thinkdsp
import fft_eng

class DBmysql:
    # define database configuratin parameters
    db_info = {}
    db_info.update({'host': "192.168.1.118"})  #localhost
    db_info.update({'user': "root"})
    db_info.update({'password': "sbrQp10"})
    db_info.update({'database': "data"})

    def __init__(self, config=db_info):
        self._conn = mysql.connector.connect(**config)
        self._cursor = self._conn.cursor()

    @property
    def connection(self):
        return self._conn

    @property
    def cursor(self):
        return self._cursor

    def commit(self):
        self.connection.commit()

    def execute(self, sql):
        self.cursor.execute(sql)

    def fetchall(self):
        return self.cursor.fetchall()

    def fetchone(self):
        return self.cursor.fetchone()

    def query(self, sql):
        self.cursor.execute(sql)
        return self.fetchall()

    def exit(self):
        self.cursor.close()
        self.connection.close()

    def get_vib_asset_list(self):
        '''

        :return: an asset list like this: ['VIB_SENSOR1', 'VIB_SENSOR2', 'VIB_SENSOR3']
        '''
        # define empty list for assets
        asset_list = []

        # define sql query to get all the vibration assets
        sql = 'SELECT processName ' \
               'FROM config.rt_process ' \
               'WHERE rt_process.processName LIKE "VIB_%"' \
               'ORDER BY rt_process.processName ASC'

        # query the database
        assets = self.query(sql)

        # create the asset list
        for asset, in assets:
            asset_list.append(asset)

        # return asset list
        return asset_list

    def get_vib_asset_dic(self):
        '''

        :return:an asset dictionary like this: {'VIB_SENSOR1': ['CFG___AREA', 'WF___TDW_X', 'WF___TDW_Z', 'WF___FFT_X', 'WF___FFT_Z', 'WF___EVENTID_X', 'WF___EVENTID_Z']}
        '''
        # define empty list for assets
        asset_dic = {}

        # get the asset list
        assets = self.get_vib_asset_list()

        for asset in assets:
            # define sql query to get all the groups and tags from vibration assets
            sql = 'SELECT groupName, tagName ' \
                   'FROM config.rt_tags_dic ' \
                   'INNER JOIN config.rt_groups ON rt_tags_dic.groupID = rt_groups.groupID ' \
                   'INNER JOIN config.rt_process ON rt_groups.processID = rt_process.processID ' \
                   'WHERE rt_process.processName = "{}"'.format(asset)

            # query the database
            groups_tags = self.query(sql)

            # define empty list
            group___tag = []

            # create the asset dictionary
            for group, tag in groups_tags:
                group___tag.append(group + '___' + tag)
                asset_dic.update({asset: group___tag})

        # return the asset dictionary
        return asset_dic

class DBinflux:
    '''# define database configuration parameters
    db_info = {}
    db_info.update({'host': "localhost"})
    db_info.update({'port': 8086})
    # db_info.update({'username': "root"})
    # db_info.update({'password': "sbrQp10"})
    db_info.update({'database': "VIB_DB"})'''

    def __init__(self, config):
        self._client = DataFrameClient(**config)

    @property
    def client(self):
        return self._client

    def query(self, sql, bind_params={}):
        return self.client.query(query=sql, bind_params=bind_params)

    def read_tdw(self, meas):
        # TODO
        pdf = 0
        return pdf

    def write_points(self, pdf, meas):
        self.client.write_points(dataframe=pdf, measurement=meas, time_precision='ms')


def test_mysql():
    # create an instance of DBmysql
    # database information is hardcoded within object
    db1 = DBmysql()

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
    client = InfluxDBClient(host='192.168.21.134', port=8086, database='VIB_DB')

    # Generating point for WF___X_EVT_CHG_ID
    points = [{
        "measurement": 'VIB_SEN1',
        "fields": {
            "WF___X_EVT_CHG_ID": '2020-03-03 01:01:00.000000+00:00'
        }
    }]
    client.write_points(points, time_precision='ms')

    # Generating points for WF___X_TDW and WF___X_EVTID
    for k in wave.ys:
        points = [{
            "measurement": 'VIB_SEN1',
            "fields": {
                "WF___X_TDW": k,
                "WF___X_EVTID": '2020-03-03 01:01:00.000000+00:00'
            }
        }]
        client.write_points(points, time_precision='ms')


def read_influx_test_data():
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

    X_EVTID1 = '2020-03-03 01:01:00.000000+00:00'

    # define database configuratin parameters
    db_info = {}
    db_info.update({'host': "192.168.21.134"})  # localhost, 192.168.1.118
    db_info.update({'port': 8086})
    db_info.update({'database': DATABASE_NAME})

    # create an instance of DBinflux
    db1 = DBinflux(config=db_info)

    # sql = "select * from " + ASSET_NAME
    # sql = "select " + X_EVTID + " from " + ASSET_NAME
    # sql = "select {}, {}, {} from {} order by time".format(X_TDW, X_EVTID, X_EVT_CHG_ID, ASSET_NAME)
    sql = "select {}, {}, {} from {} where WF___X_EVTID=$X_EVTID1;".format(X_TDW, X_EVTID, X_EVT_CHG_ID, ASSET_NAME)
    # sql = "select {} from {} where {}=$X_EVTID1;".format(X_EVT_CHG_ID, ASSET_NAME, X_EVTID)
    bind_params = {'X_EVTID1': X_EVTID1}

    # Execute query
    datasets_dic = db1.query(sql, bind_params)

    # Get pandas dataframe
    pdf_wave = datasets_dic[ASSET_NAME]
    # pdf_wave.to_csv(path_or_buf='C://Users//cmolina//Desktop//pdf_wave.csv')
    print(pdf_wave)
    # print('TDW shape: {}'.format(pdf_wave.shape))
    # print(pdf_wave.keys)


if __name__ == "__main__":
    # execute only if run as a script
    print('==================================')
    print('databases_conn ran as main script!')
    print('==================================')

    # write data to influx
    # import test_influxdb_conn
    # test_influxdb_conn.writeTestValues2()

    # read data from influx
    # test_influx()
    # write_influx_test_data()
    read_influx_test_data()