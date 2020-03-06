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
query = 'SELECT _ts, WF___CHECK_EVENTID_X FROM <asset> WHERE WF___CHECK_EVENTID_X<>0 LIMIT 2'
# returns
# CHECK_EVENTID_X1, ts1
# CHECK_EVENTID_X2, ts2


### functionII (influxdb)
# get data ts and tdw from asset where fft=null and evtid=evt_chg_id1
query = 'SELECT _ts, WF___X_TDW FROM <asset> WHERE _ts<ts2 and WF___X_FFT and WF___X_EVTID = EVTID1'
# get tdw from ts1 from ts1 where CHECK_EVENTID_X=CHECK_EVENTID_X1 and FFT =null

### functionIII (influxdb)
# compute fft of the sample

### functionIV (influxdb)
# update FFT values where ts>ts1


# global tags
tdw_x = 'WF___TDW_X'
tdw_z = 'WF___TDW_Z'
fft_x = 'WF___FFT_X'
fft_z = 'WF___FFT_Z'


# def update_config():
    # get assets from mysql config db
    asset_list = get_asset_list()
    asset_dic = get_asset_dic()

    # get sample frequency internal id
    fs_dic = get_fs_dic()


# Main
update_config()

while True:
    #check for apply changes and update config values
    # see yandy's code (redis example)
    if apply_change:
        update_config()

    for asset in asset_list:
        # check for new wave samples to compute
        # get waveform
        # compute fft, etc...
        # write fft, etc... back to influx


        col = asset_dic.get(asset)



    time.sleep(1)




'''