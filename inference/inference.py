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
from utils import rename_col_dictionary,reset_number,rename_columns
from transform_data import transform_race_df,transform_runs_df,join_runs_and_race,scale_select_col
import connect_db as conn
import create_model

inputdir=sys.argv[1]
modeldir=sys.argv[2]
outputdir=sys.argv[3]

## Load Model Weights
@pandas_udf(ArrayType(FloatType()), PandasUDFType.SCALAR)
def predict_batch_udf(data):
    batch_size = 8
    learning_rate = 3e-04
    dropout= 0.3
    model = create_model.build(lr=learning_rate,dropout=dropout)
    weight_path = modeldir+'/output/callback/95-weight-validation-loss-2.4853.hdf5'
    model.load_weights(weight_path)

    dataset = tf.data.Dataset.from_tensor_slices(data_batch).batch(batch_size)
    preds = model.predict(dataset)
    print(preds.shape)
    results = pd.Series(list(preds))
    #print(results)

    return results

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

    pred_df = processed_join_df.select(predict_batch_udf(col("scaled_features")).alias("prediction"))
    pred_df.printSchema()

 
    return 1,2,3,4

if __name__=="__main__":
    
    ## Start spark session
    sc = ps.SparkContext()
    sc.setLogLevel("ERROR")
    sqlContext = ps.sql.SQLContext(sc)
    print('Created a SparkContext')

    X_train, X_test, y_train, y_test = main(sc,sqlContext,local=False)
    print('Spark job completed')
    sc.stop()
