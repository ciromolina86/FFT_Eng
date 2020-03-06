import mysql.connector
from influxdb import DataFrameClient

import json
import time
import numpy as np
import pandas as pd


# ******************* Dtaabases config class *******************************
class Config:
    # define database configuration parameters
    mysql = {}
    mysql.update({'host': "192.168.21.134"})
    mysql.update({'port': 3306})
    mysql.update({'user': "root"})
    mysql.update({'password': "sbrQp10"})
    mysql.update({'database': "data"})

    influx = {'host': "192.168.21.134", 'port': 8086, 'username': "", 'password': "", 'database': "VIB_DB"}


# ******************* MySQL Database class *******************************
class DBmysql:

    def __init__(self, info):
        self._conn = mysql.connector.connect(**info)
        self._cursor = self._conn.cursor()
        # print('MySQL object was created')

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

        :return: an asset list like this: ['asset1', 'asset2', ...]
        '''
        # print('get_vib_asset_list was executed')

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

        :return:an asset dictionary like this: {'asset1': ['group1___tag1', 'group2___tag1', ...]}
        '''
        # print('get_vib_asset_dic was executed')

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

    def get_vib_tags_id_dic(self):
        '''

        :return:a tag:id dictionary like this: {'asset': ['group1___tag1': internalTagID1, ...]}
        '''
        # print('get_vib_tags_id_dic was executed')

        # define empty dictionary for tags
        tag_id_dic = {}

        # get the asset list
        assets = self.get_vib_asset_list()

        for asset in assets:
            # define sql query to get all the groups and tags from vibration assets
            sql = 'SELECT groupName, tagName, internalTagID ' \
                   'FROM config.rt_tags_dic ' \
                   'INNER JOIN config.rt_groups ON rt_tags_dic.groupID = rt_groups.groupID ' \
                   'INNER JOIN config.rt_process ON rt_groups.processID = rt_process.processID ' \
                   'WHERE rt_process.processName = "{}"'.format(asset)

            # query the database
            groups_tags = self.query(sql)

            # define empty list
            group___tag = []

            # create the asset dictionary
            for group, tag, internalTagID in groups_tags:
                group___tag.append((group + '___' + tag, internalTagID))
                tag_id_dic.update({asset: group___tag})

        # return the tags dictionary
        return tag_id_dic


# ******************* Influx Database class *******************************
class DBinflux:

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
        self.client.write_points(dataframe=pdf, measurement=meas)


# ******************* getinrtmatrix Function *******************************
def getinrtmatrix(rt_redis_data, in_tags_str):
    # Local Initialization
    intagsstr_redis_list = []
    input_tags_values = []
    input_tags_timestamp = []
    redis_retry = True
    redis_retry_counter = 0

    # Convert Input Tags strings to List
    internaltagidlist = in_tags_str.split(",")

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


# ******************* redis_get_value Function *******************************
def redis_get_value(rt_redis_data, redis_key):
    """

    :param redis_key:
    :return:
    """

    # Initialization
    redis_retry = True
    redis_value = None
    redis_retry_counter = 0
    max_retry = 30

    while (redis_retry is True) and (redis_retry_counter <= max_retry):
        # Get Value from Redis
        redis_value = rt_redis_data.get_value(redis_key)

        if redis_value is not None:
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

    return redis_value
    pass


# ******************* redis_set_value Function *******************************
def redis_set_value(rt_redis_data, redis_key, redis_value):
    """

    :param redis_key:
    :param redis_value:
    :return:
    """

    # Initialization
    redis_retry = True
    redis_retry_counter = 0
    max_retry = 30

    while (redis_retry is True) and (redis_retry_counter <= max_retry):
        # noinspection PyBroadException
        try:
            # Set Value to Redis
            rt_redis_data.set_value(redis_key, redis_value)
            redis_retry = False
        except Exception:
            # Disconnect from Redis DB
            rt_redis_data.close_db()

            # Sleep to create Scan Cycle = Configuration Loop Frequency
            time.sleep(0.1)

            # Connect to Redis DB
            rt_redis_data.open_db()
            redis_retry = True
            redis_retry_counter += 1

    pass


if __name__ == "__main__":
    # execute only if run as a script
    print('==================================')
    print('databases_conn ran as main script!')
    print('==================================')