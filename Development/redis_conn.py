# ===================================================================================================
#  Revision History (Release 1.0.0.0)
# ===================================================================================================
# VERSION             AUTHOR/                                     DESCRIPTION OF CHANGE
# OLD/NEW             DATE
# ===================================================================================================
# --/1.0.0.0	    | Ciro Molina CM                           |
#                   | Yandy Peres YP                           |
#                   | William Quintana WQ	                   | Initial Development.

#         		    | 28-Feb-2020   	                       |
# ====================================================================================================
# 1.0.0.0/1.0.0.1	| XXX        	                           | Update class and methods  comments.
#         		    | XX-XXX-XXXX 	                           |
# ====================================================================================================


import json
import time
import numpy as np


def getinrtmatrix(intagsstr):
    """
    This function will read SDE tags values based on the tag IDs
    :param
        intagsstr: string  (List of tags IDs, comma separated )
    :return:
        input_timestamp: string  (timestamp)
        input_tags_values: numpy.ndarray (List of tag values)

    """


    # Local Initialization
    intagsstr_redis_list = []
    input_tags_values = []
    input_tags_timestamp = []
    redis_retry = True
    redis_retry_counter = 0
    rt_redis_data = redis_conn.RedisDB()

    # Convert Input Tags strings to List
    internaltagidlist = intagsstr.split(",")

    # Get Tags Amount
    n = len(internaltagidlist)

    # Create a Tags IDs List to be use with Redis
    for i in range(n):
        intagsstr_redis_list.append("rt_data:" + str(internaltagidlist[i]))

    while (redis_retry is True) and (redis_retry_counter <= 10):
        # Get List of Values from Redis
        redis_info_list = rt_redis_data.get_value_list(intagsstr_redis_list)

        # Create the Input Tags List with Values and Timestamp
        for k in range(n):
            if redis_info_list[k] is not None:
                redis_info_temp = json.loads(redis_info_list[k])
                input_tags_values.append(float(redis_info_temp["value"]))
                input_tags_timestamp.append(redis_info_temp["timestamp"])
                redis_retry = False
            else:
                # Disconnect from Redis DB
                rt_redis_data.close_db()

                # Sleep to create Scan Cycle = Configuration Loop Frequency
                time.sleep(0.1)

                # Connect to Redis DB
                rt_redis_data.open_db()
                redis_retry = True
                redis_retry_counter += 1
                print("{Warning} Real Time DB is empty")

    # Get the Latest Timestamp Value for the Tags
    if not input_tags_timestamp:
        input_timestamp = None
        input_tags_values = None
    else:
        input_timestamp = max(input_tags_timestamp)
        input_tags_values = np.asarray(input_tags_values)

    return input_timestamp, input_tags_values


def saveoutrtmatrix(ts, outtagsstr, outputvaluesdic):
    """
    This function will write SDE values to tags in redis based on theirs tagIds

    :param ts: string
    :param outtagsstr: string
    :param outputvaluesdic:
    :return: python list
    """
    # Local Initialization
    intagsstr_redis_list = []
    output_list = []
    rt_redis_data = redis_conn.RedisDB()

    # Convert Input Tags strings to List
    internaltagidlist = outtagsstr.split(",")

    # Get Tags Amount
    n = len(internaltagidlist)

    # Create a Tags IDs List to be use with Redis
    for k in range(n):
        intagsstr_redis_list.append("rt_data:" + str(internaltagidlist[k]))

    # Check if there is not tags assigned to the inputs
    if outtagsstr is not None:
        for i in range(len(outputvaluesdic)):
            outputdict = {}
            if i != '-1':
                outputdict.update({'id': int(internaltagidlist[i])})
                outputdict.update({'isOnServer': 1})
                outputdict.update({'quality': 1})
                outputdict.update({'timestamp': ts})
                outputdict.update({'value': str(outputvaluesdic[i])})

                # Append dictionaries to the list
                output_list.append(json.dumps(outputdict))
            else:
                print("{Warning} Missing Output Tags")
        print(output_list)
        print(intagsstr_redis_list)
        #rt_redis_data.set_value_list(intagsstr_redis_list, output_list)
        rt_redis_data.push_value_list("write_queue", output_list)

    else:
        print("None Output Tags have been defined")

    return True