'''

### function1 (sql)
# query for general assets
assets <= query = 'SELECT processName FROM config.rt_process'
return assets

### function2 (sql)
# query groups and tags for each asset
groups___tags <= query = 'SELECT groups.groupName, tags.tagName FROM config.rt_tags_dic tags INNER JOIN config.rt_groups groups ON tags.groupID = groups.groupID'
return groups___tags

### functionI (influxdb)
# find ts where new change happened
query = 'SELECT _ts, WF___CHECK_EVENTID_X FROM <asset> WHERE F'


# global tags
tdw_x = 'WF___TDW_X'
tdw_z = 'WF___TDW_Z'
fft_x = 'WF___FFT_X'
fft_z = 'WF___FFT_Z'


# def init():
    # get assets from mysql config db
    asset_list = get_asset_list()
    asset_dic = get_asset_dic()


# Main
init()

while True:

    for asset in asset_list:
        col = asset_dic.get(asset)




'''