from influxdb import InfluxDBClient

# Initialization
client = InfluxDBClient('127.0.0.1', 8086, '', '', 'sorba_sde')
vib_measurement_list = []

# Get list of measurements
all_measurement_list = client.get_list_measurements()

# Filter the list the get only the vibration measurements(start measurement name with VIB_)
for measurement in all_measurement_list:
    measurement_name = measurement.get('name')
    if measurement_name.find('VIB_', 0, 4) != -1:
        vib_measurement_list.append(measurement_name)

# Get data from interval selected
for vib_measurement in vib_measurement_list:
    print("[INFO] Started processing measurement: %s" % vib_measurement)
    # Cycle Initialization
    json_list_to_write = []
    first_cycle_finished = False
    field_name_list = []
    point_counter = 0
    first_ts = None
    last_ts = None

    # Create sql query
    sql_get_data = "SELECT * FROM " + vib_measurement + " WHERE time > now() - 7d"

    # Execute query
    result = client.query(sql_get_data, epoch='ms')

    # Get data points
    points = result.get_points(vib_measurement)

    for point in points:
        # Check if EvtID and Index are in the current data
        if ("WF___EVTID_INDEX" in point) and (point['WF___EVTID_INDEX'] is not None):
            print("[INFO] Data point was already processed")
        else:
            if point['WF___EVTID'] is None:
                print("[INFO] There is not data on this range")
            else:
                # Get all field names in first cycle
                if first_cycle_finished is False:
                    # Get Initial Timestamp
                    first_ts = point['time']

                    # Get all field names
                    field_names = point.keys()

                    # Filter field names
                    for fname in field_names:
                        if fname != "WF___EVTID_INDEX" and fname != "time":
                            field_name_list.append(fname)

                    # Set Flag to True
                    first_cycle_finished = True
                else:
                    # Construct field json per point
                    field_json_point = {}
                    for fname in field_name_list:
                        # Get field value
                        field_value = point.get(fname)

                        # Check if field value is a number and if it is convert it to float
                        if type(field_value) == int:
                            field_value = float(field_value)

                        # Update dictionary
                        field_json_point.update({fname: field_value})

                    # Create json to append to the json_list
                    json_to_write = {"measurement": vib_measurement,
                                     "tags": {
                                         "WF___EVTID_INDEX": point['WF___EVTID']
                                     },
                                     "time": point['time'],
                                     "fields": field_json_point}

                    # Append json to
                    json_list_to_write.append(json_to_write)

                    # Capture last timestamp
                    last_ts = point['time']

    # Write tags
    client.write_points(json_list_to_write, time_precision='ms', batch_size=1000)

    print("[INFO] Finished processing measurement: %s" % vib_measurement)
