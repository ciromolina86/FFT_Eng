import mysql.connector

# create MySQL connection to a database
mydb = mysql.connector.connect(host='192.168.1.147', user='root', passwd='sbrQp10', database='testdb')

# create a database cursor
mycursor = mydb.cursor()

# create a SQL query
sql_query = 'SELECT * FROM table1 WHERE row="value"'

# execute SQL query
mycursor.execute(sql_query)

# fetch results
myresults = mycursor.fetchall()

# process the results
for row in myresults:
    print(row)


def william_test():
    # Author William Quintana (WQ)
    # Version 0.0.1
    #
    # Description:
    # Make a tagReport.csv.
    # SDC tag report that include
    # Accses Name, Grop Name,Tag Name and Tag ID

    import json
    import time
    import datetime
    import numpy as np
    # from redisdb import RedisDB
    import os
    import csv
    import pandas as pd
    import MySQLdb
    # from datetime import datetime

    version = "0.0.1"

    print("Version Number: %s" % version)

    print("Import finished")

    def mysqlconnect(db_info, query):
        # Trying to connect
        try:
            db_connection = MySQLdb.connect(db_info.get('hostname'), db_info.get('dbusername'), db_info.get('password'),
                                            db_info.get('dbname'))
            # If connection is not successful
        except:
            print("Can't connect to database")
            return 0
        # If Connection Is Successful
        # print("Connected to Mysql")
        # Making Cursor Object For Query Execution
        cursor = db_connection.cursor()
        # Executing Query
        cursor.execute(query)
        # Fetching Data
        records = cursor.fetchall()
        # Closing Database Connection
        db_connection.close()
        return records

    def getTagPath(db_info):
        """
        Return tag list
        """
        query = "SELECT proc.processName,groups.groupName,tags.tagName, tags.internalTagID FROM config.rt_tags_dic tags inner join config.rt_groups groups on tags.groupID = groups.groupID inner join config.rt_process proc on proc.processID = groups.processID"
        tags = mysqlconnect(db_info, query)
        return tags

    def writeLisToCsv(tags, path):
        """
        Fill the object
        "tag_id_list":[id1,id2,....], "tag_path_list":["accsessName1/groupName1/tagName1","accsessName2/groupName2/tagName2"]
        crate pandas dataframe usin the created object
        """
        tag_id_list = []
        tag_path_list = []

        for tag in tags:
            tag_id_list.append(tag[3])
            tag_path_list.append("%s/%s/%s" % (tag[0], tag[1], tag[2]))

        # print(tag_path_list)

        df = pd.DataFrame(data={"tag_id_list": tag_id_list, "tag_path_list": tag_path_list})
        # print(df.values)
        df.to_csv(path, sep=',', index=False)

    ################################################################################################
    # MAIN CODE
    ################################################################################################

    # General variables declaration

    frequency = 1800  # 30 min
    firstCicle = True

    print("###########################################")
    print("# Initialization")
    print("###########################################")

    db_info = {}
    db_info.update({'hostname': "127.0.0.1"})
    db_info.update({'dbusername': "root"})
    db_info.update({'password': "sbrQp10"})
    db_info.update({'dbname': "data"})

    tag_report_path = "/home/sdc/backup/tagReport.csv"

    # intitialization of the sdc object

    tagList = getTagPath(db_info)
    writeLisToCsv(tagList, tag_report_path)
