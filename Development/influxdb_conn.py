from influxdb import InfluxDBClient
import pandas as pd
from influxdb import DataFrameClient

'''=================================================='''

# test query to use on read_data function
query = 'SELECT "duration" FROM "pyexample"."autogen"."brushEvents" WHERE time > now() - 4d GROUP BY "user"'

# test data to use on write_data function
json_body = [{
            "measurement": 'table_name',
            "tags": {
                "user": "Carol",
                "brushId": "6c89f539-71c6-490d-a28d-6c5d84c0ee2f"
            },
            "time": "2018-03-28T8:01:00Z",
            "fields": {
                "tag_name": 'value'
            }
        }]



def read_data(host='localhost', port=8086, db='', query=''):
    # create a client
    client = InfluxDBClient(host=host, port=port)

    # execute query to database
    results = client.query(query)

    # get points from query results
    points = results.get_points(measurement=json_body['measurement'])





def write_data(host='localhost', port=8086, db='', json_body =''):
    # create a client
    client = InfluxDBClient(host=host, port=port, database=db)

    # write data
    client.write_points(json_body)


def yandy_test():
    # Initialization
    asset_name = "TESTAA1"
    group_tagnames = "_timestamp,GG1___T1,GG1___T2"
    last_retraining_time = 1581525105637

    # Instantiate the connection to the InfluxDB client
    host = '10.15.12.118'
    port = 8086
    user = ''
    password = ''
    dbname = 'sorba_carbon___super'

    # Connect client to influxdb
    client = DataFrameClient(host, port, user, password, dbname)

    # Create query string
    query_str = "select " + group_tagnames + " from " + asset_name + " WHERE time > " + str(last_retraining_time) + " and time < now()"

    # Execute query
    datasets_dic = client.query(query_str)

    # Get pandas dataframe
    pdf_from_influx = datasets_dic[asset_name]

    # Covert timestamp to milliseconds
    pdf_from_influx['_timestamp'] = pd.to_datetime(pdf_from_influx['_timestamp'], unit='ms')