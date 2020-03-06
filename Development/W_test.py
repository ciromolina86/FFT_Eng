import databases_conn
import fft_eng
import pandas as pd
from ThinkX import thinkdsp
import numpy as np
import datetime
from influxdb import InfluxDBClient
from influxdb import DataFrameClient


# Read wave formm from Influxdb
def read2_influx():
    # Initialization
    asset_name = "VIB_SENSOR1"
    group_tagnames = "_timestamp,WF___TDW_X"
    last_retraining_time = 1581525105637

    # Instantiate the connection to the InfluxDB client
    host = '127.0.0.1'
    port = 8086
    user = ''
    password = ''
    dbname = 'VIB_DB'

    # Connect client to influxdb
    client = DataFrameClient(host, port, user, password, dbname)

    # Create query string
    query_str = "select " + group_tagnames + " from " + asset_name

    # Execute query
    datasets_dic = client.query(query=query_str, epoch='ms')

    # Get pandas dataframe
    pdf_from_influx = datasets_dic[asset_name]

    # Covert timestamp to milliseconds
    # pdf_from_influx['_timestamp'] = pd.to_datetime(pdf_from_influx['_timestamp'], unit='ms')

    return pdf_from_influx


def read_influx():
    # create an instance of DBinflux
    # database information is hardcoded within object
    db1 = databases_conn.DBinflux()

    # Initialization
    asset_name = "VIB_SENSOR1"
    group_tagnames = "Date, WF___TDW_X, WF___FFT_X"

    # sql = "select * from " + asset_name
    sql = "select " + group_tagnames + " from " + asset_name

    # Execute query
    datasets_dic = db1.query(sql)

    # Get pandas dataframe
    pdf_from_influx = datasets_dic[asset_name]

    return pdf_from_influx


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



def pdf_column_to_influx(InfluxDBClient_object, pdf, pdf_column ,measurement,field):
    """

    :param InfluxDBClient_object:
    :param pdf:
    :param pdf_column:
    :param measurement:
    :param field:
    :return:
    """
    count =0
    for date, row in pdf.iterrows():
        json_body = [{
            "measurement": measurement,
            "time": date,
            "fields": {field: row[pdf_column]}
        }]
        if row[pdf_column] != 0:
            InfluxDBClient_object.write_points(json_body)
            count += 1
    return count


df = read_influx()
# print(df)

wave_array = df["WF___TDW_X"].values
wave_array_size = wave_array.size
wave = thinkdsp.Wave(ys=wave_array, framerate=100)

# Apply FFT
spectrum_object = wave.make_spectrum()
spectrum_array = spectrum_object.real
spectrum_array_size = spectrum_array.size


# final_spectrum_array = np.append(spectrum_array, add_array)
spectrum_pdf = pd.DataFrame(data={'WF___FFT_X': spectrum_array}, index=df.index[:spectrum_array_size])
print(spectrum_pdf)

# df['WF___FFT_X'] = final_spectrum_array
#
# print(df)

NewMeasurementName = "VIB_SENSOR1"
client = DataFrameClient(host='127.0.0.1', port=8086)
client.switch_database('VIB_DB')

client.write_points(dataframe=spectrum_pdf, measurement=NewMeasurementName, time_precision='ms')


df1 = read_influx()
print(df1)












