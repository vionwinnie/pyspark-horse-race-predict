from pyspark.sql import SQLContext
from pyspark import SparkContext


def get_jdbc_sink():
    ## MySQL Connection
    hostname = "xxxxxxxxx"
    dbname="horse_race"
    jdbcPort=3306
    username="root"
    password="xxxxx"

    jdbc_url="jdbc:mysql://{}:{}/{}?user={}&password={}".format(hostname,jdbcPort,dbname,username,password)

    return jdbc_url

if __name__=='__main__':
    sc = SparkContext(appName="TestPySparkJDBC")
    sqlContext = SQLContext(sc)
    query = "(select * from horse_race.races) t1_alias"
    jdbc_url = get_jdbc_sink()
    df = sqlContext.read.format('jdbc').options(driver='com.mysql.jdbc.Driver',url=jdbc_url,dbtable=query).load()
    df.show()

    query2 = "(select * from horse_race.runs) t1_alias"
    jdbc_url = get_jdbc_sink()
    df2 = sqlContext.read.format('jdbc').options(driver='com.mysql.jdbc.Driver',url=jdbc_url,dbtable=query2).load()
