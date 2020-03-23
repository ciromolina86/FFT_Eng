from influxdb import InfluxDBClient


def influx_index_adder(asset_name, event_ids_list, asset_dic_queue):
    """

    :param asset_name:
    :param event_ids_list:
    :param asset_dic_queue:
    :return:
    """
    # Initialization
    client = InfluxDBClient('127.0.0.1', 8086, '', '', 'sorba_sde')

    # Get Any element from the list since the axis has the same EventID
    event_id = event_ids_list[0]

    print("[INFO] Started adding index to measurement: %s" % asset_name)
    print("[INFO] Asset: %s" % asset_name)
    print("[INFO] Event ID LIST: %s" % event_ids_list)
    print("[INFO] Event ID: %s" % event_id)

    # Cycle Initialization
    json_list_to_write = []
    first_cycle_finished = False
    field_name_list = []

    # Create sql query metadata
    wf_evt_id = 'WF___EVTID'
    evt_id = "'{}'".format(event_id)

    # Create sql query string
    sql_get_data = "select * from {} where {} = {}".format(asset_name, wf_evt_id, evt_id)

    # Execute query
    result = client.query(sql_get_data, epoch='ms')

    # Get data points
    points = result.get_points(asset_name)

    for point in points:
        # New Points Initialization
        row_already_processed_counter = 0
        no_data_in_range_counter = 0

        # Check if EvtID and Index are in the current data
        if ("WF___EVTID_INDEX" in point) and (point['WF___EVTID_INDEX'] is not None):
            row_already_processed_counter += 1
        else:
            if point['WF___EVTID'] is None:
                no_data_in_range_counter += 1
            else:
                # Get all field names in first cycle
                if first_cycle_finished is False:
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
                    json_to_write = {"measurement": asset_name.replace("VIB_", "V_"),
                                     "tags": {
                                         "WF___EVTID_INDEX": point['WF___EVTID']
                                     },
                                     "time": point['time'],
                                     "fields": field_json_point}

                    # Append json to
                    json_list_to_write.append(json_to_write)

    # Write tags
    client.write_points(json_list_to_write, time_precision='ms', batch_size=1000)

    # Add event ID to the processed queue
    processed_event_id_queue = asset_dic_queue.get(asset_name)
    processed_event_id_queue.appendleft(str(event_id))
    asset_dic_queue.update({asset_name: processed_event_id_queue})

    print("[INFO] Finished adding index to measurement: %s" % asset_name)
    # print("[INFO] Processed Event IDs DIC: %s" % asset_dic_queue)

    return asset_dic_queue
