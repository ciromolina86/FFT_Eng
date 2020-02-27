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
