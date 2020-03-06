# ===================================================================================================
#  Revision History (Release 1.0.0.0)
# ===================================================================================================
# VERSION             AUTHOR/                                     DESCRIPTION OF CHANGE
# OLD/NEW             DATE
# ===================================================================================================
# --/1.0.0.0	    | Ciro Molina CM                           |
#                   | Yandy Peres YP                           |
#                   | William Quintana WQ	                   | Initial Development.

#         		    | 29-Feb-2020   	                       |
# ====================================================================================================
# 1.0.0.0/1.0.0.1	| XXX        	                           | Update class and methods  comments.
#         		    | XX-XXX-XXXX 	                           |
# ====================================================================================================


import MySQLdb


# from datetime import datetime

# version = "0.0.1"
#
# print("Version Number: %s" % version)
#
# print("Import finished")


def mysqlconnect(db_info, query):
    """
    connect to the database

    :param db_info: dictionary {'hostname': "127.0.0.1",'dbusername': "root",'password': "sbrQp10",'dbname': "data"}
    :param query: string
    :return:
    """
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
    Uses query to look for all the Vibration tags info like
    Assets, Tags Groups, Tags Name, Tags ID

    :param db_info:
    :return:
    """
    query = "SELECT proc.processName,groups.groupName,tags.tagName, tags.internalTagID FROM config.rt_tags_dic tags inner join config.rt_groups groups on tags.groupID = groups.groupID inner join config.rt_process proc on proc.processID = groups.processID WHERE proc.processName LIKE 'VIB_%'"

    try:
        tags = mysqlconnect(db_info, query)
        return tags
    except:
        print("Failing function to get the tags ")



def getvibrationassets(db_info):
    """
    Uses query to look for all the vibration assets

    :return:
        assetslist: python tuple
    """

    query = "SELECT processName FROM config.rt_process WHERE processName LIKE 'VIB_%'"
    assets_list = ()
    try:
        temp_list = mysqlconnect(db_info, query)
        if temp_list != ():
            for item in temp_list:
                assets_list = assets_list + (item[0],)
            return assets_list
        else:
            print("there are not Vibration assets configured")
    except:
        print("Failing function to get Assets name from the SDE")


def getTagValuesfromAssetDatabase(db_info, asset_object):
    """
    Description
    Uses query to get tags values from the mysql db

    :param db_info:
    :param asset_object: {asset_name:(('WF', 'EVENTID_X', 15L), ('WF', 'EVENTID_Z', 16L),.....}
    :return: python dictionary
    """

    # Init
    try:
        tagDictLocal = {}
        queryTagValues = "SELECT internalTagID, value FROM data.data WHERE internalTagID ='%s' ORDER BY timeStamp DESC LIMIT 1 "  # tag_ID

        asset_tag_list = asset_object.values()

        for tag in asset_tag_list[0]:
            tagId = tag[2]

            # Execute query to database
            query = queryTagValues %(tagId)
            values = mysqlconnect(db_info, query)
            # Assign Values to tag dictionary
            tagDictLocal[tag[1]] = values

        # print(asset[0])
        # print(tagDictLocal)
        return tagDictLocal
    except:
        print("Issues with queries to the sdc database and getting tags data ")


def getassetsattributes(db_info, assets_list):
    """
    Description
    Uses query and others functions to return a list of dictionaries with information about the assets provided.
    Example of the structure :
        {'VIB_SENSOR': (('WF', 'EVENTID_X', 15L, 'a'), ('WF', 'EVENTID_Z', 16L, 'a'), ('WF', 'FFT_X', 17L, 'a'),
                        ('WF', 'FFT_Z', 18L, 'a'), ('WF', 'TDW_X', 19L, 'a'), ('WF', 'TDW_Z', 20L, 'a'), ('CFG', 'LINE', 22L, 'a'),
                        ('CFG', 'CONTROLER', 23L, 'a'), ('CFG', 'SENSOR_ID', 24L, 'a'), ('CFG', 'LOC', 25L, 'a') ....)}
    :param assets_list:
    :return: Python dictionary
    """

    query = "SELECT groups.groupName,tags.tagName, tags.internalTagID FROM config.rt_tags_dic tags inner join config.rt_groups groups on tags.groupID = groups.groupID inner join config.rt_process proc on proc.processID = groups.processID WHERE proc.processName = '%s'"
    asset_att_list = []

    try:
        if assets_list != ():
            for asset in assets_list:
                asset_group_tags_values = ()
                # getting group and tag info (('GROUP', 'TAG_NAME', TAG_ID),  .....
                asset_group_tags = mysqlconnect(db_info, query%(asset))
                # building an asset object
                asset_object = {asset: asset_group_tags}
                # getting tag values
                tag_object_dict = getTagValuesfromAssetDatabase(db_info, asset_object)
                # Add tag values to the tag info structure
                for asset_group_tag in asset_group_tags:
                    tag_name = asset_group_tag[1]
                    # print("tag name")
                    # print(tag_name)
                    value = tag_object_dict[tag_name][0][1]
                    # print("tag, value")
                    # print(tag_name, value)
                    asset_group_tag_value = asset_group_tag + (value,)
                    # print("asset with value")
                    # print(asset_group_tag_value)
                    asset_group_tags_values = asset_group_tags_values + (asset_group_tag_value,)
                    # print(" asset group ")
                    # print(asset_group_tags_values)
                # Building a general asset dictionary
                asset_att_list.append({asset: asset_group_tags_values})
            return asset_att_list
        else:
            print("No vibration asset when trying to get tags  ")

    except:
        print("Failing function to get the tags from Assets")





# ################################################################################################
# # MAIN CODE
# ################################################################################################

#
# print("###########################################")
# print("# Initialization")
# print("###########################################")
#
db_info = {}
db_info.update({'hostname': "192.168.1.118"})
db_info.update({'dbusername': "root"})
db_info.update({'password': "sbrQp10"})
db_info.update({'dbname': "data"})


assetlist = getvibrationassets(db_info)
# print(assetlist)
assets = getassetsattributes(db_info, assetlist)
print(assets)



