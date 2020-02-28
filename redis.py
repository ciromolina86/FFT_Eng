#===================================================================================================
#  Revision History (Release 1.0.0.0)
#===================================================================================================
# VERSION             AUTHOR/                                     DESCRIPTION OF CHANGE
# OLD/NEW             DATE
#===================================================================================================
# --/1.0.0.0	    | Ciro Molina CM                           |
#                   | Yandy Peres YP                           |
#                   | William Quintana WQ	                   | Initial Development.

#         		    | 28-Feb-2020   	                       |
#====================================================================================================
# 1.0.0.0/1.0.0.1	| XXX        	                           | Update class and methods  comments.
#         		    | XX-XXX-XXXX 	                           |
#====================================================================================================





import json
import time
import os
import numpy as np
from redisdb import RedisDB





def getinrtmatrix(intagsstr):
    """
    This function will read tags values based on the tag IDs
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