import mysql.connector

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


if __name__ == "__main__":
    '''execute only if run as a main script'''
    print('databases_conn ran as main!')

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

