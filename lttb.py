

def lttb_areas_of_triangles(a, bs, c):
    """
    Calculate areas of triangles from duples of vertex coordinates.
    Uses implicit numpy broadcasting along first axis of "bs".

    :param a:
    :param bs:
    :param c:
    :return: numpy.array. Array of areas of shape (len(bs),)
    """

    bs_minus_a = bs - a
    a_minus_bs = a - bs
    return 0.5 * abs((a[0] - c[0]) * (bs_minus_a[:, 1]) - (a_minus_bs[:, 0]) * (c[1] - a[1]))


def lttb_downsample(np, data_in, n_out):
    """
    Downsample "data" to "n_out" points using the LTTB algorithm.

    Constraints
    -----------
      - 3 <= n_out <= nrows(data)
      - "data" should be sorted on the first column.

    :param np:
    :param data_in:
    :param n_out:
    :return: numpy.array. Array of shape (n_out, n_columns)
    """

    # Initialization
    data = data_in.copy()

    if any(data[:, 0] != np.sort(data[:, 0])):
        raise ValueError('data should be sorted on first column')

    if n_out >= data.shape[0]:
        return data

    if n_out < 3:
        raise ValueError('Can only downsample to a minimum of 3 points')

    # Split data into bins
    n_bins = n_out - 2
    data_bins = np.array_split(data[1: len(data) - 1], n_bins)

    # Prepare output array
    # First and last points are the same as in the input.
    # Check Array Dimension
    data_dim = data.ndim

    if data_dim == 2:
        # Get number of columns of the numpy array
        data_row, data_col = data.shape
    elif data_dim == 1:
        data_col = 1
    else:
        data_col = 0
        print("Error: Wrong Input Matrix Dimension")

    out = np.zeros((n_out, data_col))
    out[0] = data[0]
    out[len(out) - 1] = data[len(data) - 1]

    # Largest Triangle Three Buckets (LTTB):
    # In each bin, find the point that makes the largest triangle
    # with the point saved in the previous bin
    # and the centroid of the points in the next bin.
    for i in range(len(data_bins)):
        this_bin = data_bins[i]

        if i < n_bins - 1:
            next_bin = data_bins[i + 1]
        else:
            next_bin = data[len(data) - 1:]

        a = out[i]
        bs = this_bin
        c = next_bin.mean(axis=0)

        areas = lttb_areas_of_triangles(a, bs, c)

        out[i + 1] = bs[np.argmax(areas)]

    return out


def lttb_downsample_ts(np, data_in, data_ts_in, n_out):
    """
    Downsample "data" to "n_out" points using the LTTB algorithm.

    Constraints
    -----------
      - 3 <= n_out <= nrows(data)
      - "data" should be sorted on the first column.
      - number of raw should be equal on timestamp and values arrays

    :param np:
    :param data_in:
    :param data_ts_in:
    :param n_out:
    :return: numpy.array. Array of shape (n_out, n_columns)
    """

    # Initialization
    data = data_in.copy()
    data_ts = data_ts_in.copy()

    if any(data[:, 0] != np.sort(data[:, 0])):
        raise ValueError('data should be sorted on first column')

    if n_out >= data.shape[0]:
        return data, data_ts

    if n_out < 3:
        raise ValueError('Can only downsample to a minimum of 3 points')

    # Check Array Dimension for Array with real values
    data_dim = data.ndim

    if data_dim == 2:
        # Get number of columns of the numpy array
        data_row, data_col = data.shape
    elif data_dim == 1:
        data_row = data.size
        data_col = 1
    else:
        data_row = 0
        data_col = 0
        print("Error: Wrong Input Matrix Dimension")

    # Check Array Dimension for Array with timestamps
    data_dim_ts = data_ts.ndim

    if data_dim_ts == 2:
        # Get number of columns of the numpy array
        data_row_ts, data_col_ts = data_ts.shape
    elif data_dim_ts == 1:
        data_row_ts = data_ts.size
    else:
        data_row_ts = 0
        print("Error: Wrong Input Matrix Dimension")

    if data_row != data_row_ts:
        raise ValueError('Timestamp and Data array have different number of rows')

    # Split data into bins
    n_bins = n_out - 2
    data_bins = np.array_split(data[1: len(data) - 1], n_bins)
    data_bins_ts = np.array_split(data_ts[1: len(data_ts) - 1], n_bins)

    # Prepare output array
    # First and last points are the same as in the input.
    # Create Dummy Arry filled with zeros
    out = np.zeros((n_out, data_col))
    out_ts = data_ts[0:n_out]
    out[0] = data[0]
    out_ts[0] = data_ts[0]
    out[len(out) - 1] = data[len(data) - 1]
    out_ts[len(out_ts) - 1] = data_ts[len(data_ts) - 1]

    # Largest Triangle Three Buckets (LTTB):
    # In each bin, find the point that makes the largest triangle
    # with the point saved in the previous bin
    # and the centroid of the points in the next bin.
    for i in range(len(data_bins)):
        this_bin = data_bins[i]
        this_bin_ts = data_bins_ts[i]

        if i < n_bins - 1:
            next_bin = data_bins[i + 1]
        else:
            next_bin = data[len(data) - 1:]

        # Calculate triangle axis and area for data values
        a = out[i]
        bs = this_bin
        c = next_bin.mean(axis=0)

        # Calculate Area
        areas = lttb_areas_of_triangles(a, bs, c)

        # Calculate output values
        out[i + 1] = bs[np.argmax(areas)]

        # Check Array Dimension for Array with timestamps
        this_bin_dim_ts = this_bin_ts.ndim

        if this_bin_dim_ts == 2:
            # Get number of columns of the numpy array
            this_bin_row_ts, this_bin_col_ts = this_bin_ts.shape
        elif this_bin_dim_ts == 1:
            this_bin_row_ts = this_bin_ts.size
        else:
            this_bin_row_ts = 0
            print("Error: Wrong Input Matrix Dimension")

        # Get middle timestamp on the bins
        this_bin_row_mean_ts = int(this_bin_row_ts / 2)
        this_bin_mean_ts = this_bin_ts[this_bin_row_mean_ts]

        # Calculate timestamp output
        out_ts[i + 1] = this_bin_mean_ts

    return out, out_ts
