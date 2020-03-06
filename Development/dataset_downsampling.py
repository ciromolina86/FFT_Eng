
import lttb
import pandas as pd
import numpy as np


def get_col_and_rows_numpy_array(numpy_array):
    """

    :param numpy_array:
    :return:
    """
    # Check Array Dimension
    numpy_array_dim = numpy_array.ndim

    if numpy_array_dim == 2:
        # Get number of columns of the numpy array
        numpy_array_row, numpy_array_col = numpy_array.shape
    elif numpy_array_dim == 1:
        numpy_array_row = numpy_array.size
        numpy_array_col = 1
    else:
        numpy_array_row = 0
        numpy_array_col = 0
        print("Error: Wrong Input Matrix Dimension")

    return numpy_array_row, numpy_array_col


def dataset_downsampling_lttb_ts(np, data_v_in, data_ts_in, overview_max_datapoints, row_count_in, column_count_in):
    """

    :param np:
    :param data_v_in:
    :param data_ts_in:
    :param overview_max_datapoints:
    :param row_count_in:
    :param column_count_in:
    :return:
    """

    #############################################################################
    # Down-sampling Data Set if it is needed
    #############################################################################
    # Get number of  data points
    datapoints_count = int(column_count_in) * int(row_count_in)
    overview_max_row = int(int(overview_max_datapoints) / int(column_count_in))

    if datapoints_count > overview_max_datapoints:
        # Create Index Array
        data_v = data_v_in.copy()
        data_ts = data_ts_in.copy()
        ind = np.linspace(1, len(data_v[:, 0]), len(data_v[:, 0]))
        ind = np.asmatrix(ind)
        ind = ind.T

        # Concatenate Index and Data Value array
        data_v = np.concatenate((ind, data_v), axis=1)

        # Convert to array
        data_v = np.asarray(data_v)
        data_ts = np.asarray(data_ts)

        # Downsample using LTTB
        data_v_out_tmp, data_ts_out = lttb.lttb_downsample_ts(np, data_v, data_ts, overview_max_row)

        # Check Array Dimension for Array with real values
        data_v_out_dim = data_v_out_tmp.ndim

        if data_v_out_dim == 2:
            # Get number of columns of the numpy array
            data_v_out_row, data_v_out_col = data_v_out_tmp.shape
        elif data_v_out_dim == 1:
            data_v_out_col = 1
        else:
            data_v_out_col = 0
            print("Error: Wrong Input Matrix Dimension")

        # Remove Index Column
        data_v_out = data_v_out_tmp[:, 1:data_v_out_col]

    else:
        data_v_out = data_v_in.copy()
        data_ts_out = data_ts_in.copy()

    # Transform to numpy matrix
    data_v_out = np.asmatrix(data_v_out)
    data_ts_out = np.asmatrix(data_ts_out)

    return data_v_out, data_ts_out


def dataset_downsampling_lttb(np, data_v_in, overview_max_datapoints, row_count_in, column_count_in):
    """

    :param np:
    :param data_v_in:
    :param overview_max_datapoints:
    :param row_count_in:
    :param column_count_in:
    :return:
    """

    #############################################################################
    # Down-sampling Data Set if it is needed
    #############################################################################
    # Get number of  data points
    datapoints_count = int(column_count_in) * int(row_count_in)
    overview_max_row = int(int(overview_max_datapoints) / int(column_count_in))

    if datapoints_count > overview_max_datapoints:
        # Create Index Array
        data_v = data_v_in.copy()
        ind = np.linspace(1, len(data_v[:, 0]), len(data_v[:, 0]))
        ind = np.asmatrix(ind)
        ind = ind.T

        # Concatenate Index and Data Value array
        data_v = np.concatenate((ind, data_v), axis=1)

        # Convert to array
        data_v = np.asarray(data_v)

        # Downsample using LTTB
        data_v_out_tmp = lttb.lttb_downsample(np, data_v, overview_max_row)

        # Check Array Dimension for Array with real values
        data_v_out_dim = data_v_out_tmp.ndim

        if data_v_out_dim == 2:
            # Get number of columns of the numpy array
            data_v_out_row, data_v_out_col = data_v_out_tmp.shape
        elif data_v_out_dim == 1:
            data_v_out_col = 1
        else:
            data_v_out_col = 0
            print("Error: Wrong Input Matrix Dimension")

        # Remove Index Column
        data_v_out = data_v_out_tmp[:, 1:data_v_out_col]

    else:
        data_v_out = data_v_in.copy()

    return data_v_out


###########################################################################
# MAIN
###########################################################################
# Initialization
args = {}
args.update({'timestamp_fieldname': "Time"})
args.update({'selected_feature': ["tag1", "target"]})
max_datapoints = 60

# Read dummy data
data_csv_path = "C:\\Users\\yramos\\Downloads\\vpn\\Local Training tmp\\test_forecast.csv"
pdf = pd.read_csv(data_csv_path, delimiter=',')

##################################
# DOWN-SAMPLING using Numpy matrix with timestamp
##################################
# Create timestamps array
input_mtx_ts = pdf[args.get('timestamp_fieldname')].values

# Training Input Reduced for Overview using LTTB
input_mtx = pdf[args['selected_feature']].values
row_count, column_count = get_col_and_rows_numpy_array(input_mtx)
downsampled_mtx, downsampled_mtx_ts = dataset_downsampling_lttb_ts(np, input_mtx, input_mtx_ts, max_datapoints, row_count, column_count)


##################################
# DOWN-SAMPLING using Numpy matrix without timestamp
##################################
#input_mtx = pdf[args['selected_feature']].values
#row_count, column_count = get_col_and_rows_numpy_array(input_mtx)
#downsampled_mtx = dataset_downsampling_lttb(np, input_mtx, max_datapoints, row_count, column_count)

print downsampled_mtx.shape
