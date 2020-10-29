## Loading Packages and Dependencies
import sys
import pyspark as ps
import warnings
import re
import pandas as pd
import numpy as np

from pyspark.sql.functions import isnan, when, count, col
from pyspark.sql import functions as f
from pyspark.sql import types as t
from pyspark.sql.types import StringType
from pyspark.ml.feature import Tokenizer, NGram, CountVectorizer, IDF, StringIndexer, VectorAssembler,StandardScaler
from pyspark.ml import Pipeline,PipelineModel

## Import from modules
from utils import rename_col_dictionary,reset_number,rename_columns
from transform_data import transform_race_df,transform_runs_df,join_runs_and_race,scale_select_col

inputdir=sys.argv[1]
modeldir=sys.argv[2]
outputdir=sys.argv[3]


def main(sc,sqlContext):

    race_file_path = inputdir+'races.csv'
    runs_file_path = inputdir+'runs.csv'

    ## Load raw data
    race_df = sqlContext.read.format('com.databricks.spark.csv')\
                    .options(header='true', inferschema='true')\
                    .load(race_file_path)

    runs_df = sqlContext.read.format('com.databricks.spark.csv')\
                    .options(header='true', inferschema='true')\
                    .load(runs_file_path)
    
    ## Processed Race Dataframe
    race_pipeline, processed_race_df = transform_race_df(race_df)
    ## Saving runs data conversion pipeline
    race_modeldir = modeldir+'/race_convert_model.pb'
    race_pipeline.write().overwrite().save(race_modeldir)

    ## Processed Runs Dataframe
    runs_pipeline, processed_runs_df = transform_runs_df(runs_df)
    ## Saving runs data conversion pipeline
    runs_modeldir = modeldir+'/runs_convert_model.pb'
    runs_pipeline.write().overwrite().save(runs_modeldir)

    ## Join Runs and Race
    join_df = join_runs_and_race(processed_runs_df,processed_race_df)

    ## Vectorize and Scale Numeric Features of join_df
    scaler_pipeline, processed_join_df = scale_select_col(join_df)
    ## Saving runs data conversion pipeline
    scaler_modeldir = modeldir+'/scaler_model.pb'
    scaler_pipeline.write().overwrite().save(scaler_modeldir)

    processed_join_df.show(5)

    return 1,2,3,4

if __name__=="__main__":
    
    ## Start spark session
    sc = ps.SparkContext()
    sc.setLogLevel("ERROR")
    sqlContext = ps.sql.SQLContext(sc)
    print('Created a SparkContext')

    X_train, X_test, y_train, y_test = main(sc,sqlContext)
    print('Spark job completed')
    sc.stop()
