from pyspark.sql import SQLContext
from pyspark import SparkContext

sc = SparkContext(appName="TestPySparkJDBC")
sqlContext = SQLContext(sc)

## MySQL Connection
hostname = "35.226.103.69"
dbname="horse_race"
jdbcPort=3306
username="root"
password="123"

jdbc_url="jdbc:mysql://{}:{}/{}?user={}&password={}".format(hostname,jdbcPort,dbname,username,password)

query = "(select * from horse_race.tutorials_tbl) t1_alias"

df = sqlContext.read.format('jdbc').options(driver='com.mysql.jdbc.Driver',url=jdbc_url,dbtable=query).load()
df.show()

