## Loading Packages and Dependencies
import sys
import pyspark as ps
import warnings
import re
import pandas as pd
import numpy as np
import tensorflow as tf

from pyspark.sql.functions import isnan, when, count, col, pandas_udf, PandasUDFType
from pyspark.sql import functions as f
from pyspark.sql import types as t
from pyspark.sql.types import *
from pyspark.ml.feature import Tokenizer, NGram, CountVectorizer, IDF, StringIndexer, VectorAssembler,StandardScaler
from pyspark.ml import Pipeline,PipelineModel

## Import from modules
from utils import rename_col_dictionary,reset_number,rename_columns,to_array
from transform_data import transform_race_df,transform_runs_df,join_runs_and_race,scale_select_col
import connect_db as conn
import create_model

inputdir=sys.argv[1]
modeldir=sys.argv[2]
outputdir=sys.argv[3]

## Load Model Weights
@pandas_udf(ArrayType(FloatType()), PandasUDFType.SCALAR)
def predict_udf(data):
    batch_size = 8
    learning_rate = 3e-04
    dropout= 0.3
    model = create_model.build(lr=learning_rate,dropout=dropout)

    weight_path="/home/dcvionwinnie/output/callback/90-weight-validation-loss-2.4644.hdf5"
    model.load_weights(weight_path)
    preds = model.predict_classes(data, verbose=1)
    return pd.Series(preds)

def main(sc,sqlContext,local=False):

    ## Load Race and Run Table from MySQL Database
    runs_df=None
    race_df=None
    

    if local:
        race_file_path = inputdir+'races.csv'
        runs_file_path = inputdir+'runs.csv'

        ## Load raw data
        race_df = sqlContext.read.format('com.databricks.spark.csv')\
                    .options(header='true', inferschema='true')\
                    .load(race_file_path)

        runs_df = sqlContext.read.format('com.databricks.spark.csv')\
                    .options(header='true', inferschema='true')\
                    .load(runs_file_path)
    else: 
        run_query = "(select * from horse_race.runs) t1_alias"
        race_query = "(select * from horse_race.races) t1_alias"
        jdbc_url = conn.get_jdbc_sink()

        runs_df = sqlContext.read.format('jdbc').options(driver='com.mysql.jdbc.Driver',url=jdbc_url,dbtable=run_query).load()
        race_df = sqlContext.read.format('jdbc').options(driver='com.mysql.jdbc.Driver',url=jdbc_url,dbtable=race_query).load()

        print(runs_df.count())
        print(race_df.count())
    
    
    ## Processed Race Dataframe
    race_modeldir = modeldir+'/race_convert_model.pb'
    processed_race_df = transform_race_df(race_modeldir,race_df)
    print(processed_race_df.count())

    ## Processed Runs Dataframe
    runs_modeldir = modeldir+'/runs_convert_model.pb'
    processed_runs_df = transform_runs_df(runs_modeldir,runs_df)
    print(processed_runs_df.count())
    
    ## Join Runs and Race
    join_df = join_runs_and_race(processed_runs_df,processed_race_df)
    print(join_df.count())

    ## Vectorize and Scale Numeric Features of join_df
    scaler_modeldir = modeldir+'/scaler_model.pb'
    processed_join_df = scale_select_col(scaler_modeldir,join_df)
    print("data processing completed with {} rows".format(processed_join_df.count()))
    ## Convert into pandas dataframe for prediction
    processed_df_exploded = processed_join_df.withColumn("v", to_array("scaled_features"))
    processed_join_pdf = processed_df_exploded.select([col("v")[i] for i in range(104)]).toPandas()

    ## Load Model - need to preload model to each node from storage bucket
    learning_rate = 3e-04
    dropout= 0.3
    model = create_model.build(lr=learning_rate,dropout=dropout)
    weight_path="/home/dcvionwinnie/output/callback/90-weight-validation-loss-2.4644.hdf5"
    model.load_weights(weight_path)    
    preds = model.predict_classes(processed_join_pdf)
    preds = [int(item)+1 for item in preds]

    ## Export Results to MySQL
    race_id_col = join_df.select("race_id").collect()
    race_id_list = [row['race_id'] for row in race_id_col]

    tuple_result = [(race_id,pred_score) for race_id, pred_score in zip(race_id_list,preds)]
    result_df = sqlContext.createDataFrame(tuple_result,['race_id','draw'])

    tmp_run_df = runs_df.select("race_id","draw","horse_name")
    output_df = result_df.join(tmp_run_df,on=["race_id","draw"],how="left")
    output_df.show()

    output_df.write.jdbc(url=jdbc_url,table="prediction",mode="overwrite")
            
    #.options(driver='com.mysql.jdbc.Driver',url=jdbc_url,dbtable="prediction")
    print("output is exported")
    """

    ## to_array takes in the lambda function for list and return array type with float value
    to_array = f.udf(lambda v: v.toArray().tolist(), t.ArrayType(t.FloatType()))
    processed_join_exploded = processed_join_df.withColumn('array_scaled', to_array('scaled_features')).select('array_scaled')
    pred_df = processed_join_exploded.select(predict_udf(col("array_scaled")).alias("prediction"))
    pred_df.show()
    #pred_df_exploded.printSchema()
    """
    return None

if __name__=="__main__":
    
    ## Start spark session
    sc = ps.SparkContext()
    sc.setLogLevel("ERROR")
    sqlContext = ps.sql.SQLContext(sc)
    print('Created a SparkContext')

    final = main(sc,sqlContext,local=False)
    print('Spark job completed')
    sc.stop()
