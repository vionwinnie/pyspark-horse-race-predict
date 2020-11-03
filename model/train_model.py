## Loading Packages and Dependencies
import sys
import pyspark as ps
import warnings
import re
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

## PySpark functions
from pyspark.sql.functions import isnan, when, count, col
from pyspark.sql import functions as f
from pyspark.sql import types as t
from pyspark.sql.types import StringType
from pyspark.ml.feature import Tokenizer, NGram, CountVectorizer, IDF, StringIndexer, VectorAssembler,StandardScaler
from pyspark.ml import Pipeline,PipelineModel

## Import from modules
import create_model
from utils import rename_col_dictionary,reset_number,rename_columns,to_array
from transform_data import transform_race_df,transform_runs_df,join_runs_and_race,scale_select_col

inputdir=sys.argv[1]
modeldir=sys.argv[2]
outputdir=sys.argv[3]

import os,shutil

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

    #processed_join_df.show(5)
    selected_y = [s for s in processed_join_df.columns if 'result' in s]
    y_all = processed_join_df.select(selected_y)
    y_all = reset_number(y_all)

    ## Convert to Pandas Dataframe
    X_all_exploded = processed_join_df.withColumn("v", to_array("scaled_features"))
    X_all_exploded.printSchema()
    X_all_pdf = X_all_exploded.select([col("v")[i] for i in range(104)]).toPandas()
    y_all_pdf = y_all.select("*").toPandas()

    X_train, X_test, y_train, y_test = train_test_split(X_all_pdf, y_all_pdf, test_size=0.2, random_state=42)
    
    print(X_train.shape)
    print(y_train.shape)


    ## Model Training
    learning_rate = 3e-04
    epoch = 200
    batch_size=30
    dropout=0.3
    
    model = create_model.build(lr=learning_rate,dropout=dropout)
    weight_callback = create_model.set_callback()

    ## Create dataset 
    dataset = tf.data.Dataset.from_tensor_slices((X_train.values, y_train.values))
    train_dataset = dataset.shuffle(len(X_train)).batch(batch_size)
    dataset = tf.data.Dataset.from_tensor_slices((X_test.values, y_test.values))
    validation_dataset = dataset.shuffle(len(X_test)).batch(batch_size)

    print("Model Training Started..\n")
    history = model.fit(train_dataset,
                    epochs=epoch,
                    validation_data=validation_dataset,
                    callbacks=[weight_callback])
    print("Model Training Done.")

    ## Save Model
    tf.keras.backend.set_learning_phase(0)
    
    export_model_dir = "./output/export_model/"
    if os.path.exists(export_model_dir):
        shutil.rmtree(export_model_dir)
    os.makedirs(export_model_dir)

    export_path = '{}/batch_size-{}_lr-{}_epoch_{}_dropout_{}'.format(export_model_dir,batch_size,learning_rate,epoch,dropout)
    model.save(export_path) 
    print("Model exported")

    return X_train,X_test,y_train,y_test

if __name__=="__main__":
    
    ## Start spark session
    sc = ps.SparkContext()
    sc.setLogLevel("ERROR")
    sqlContext = ps.sql.SQLContext(sc)
    print('Created a SparkContext')

    X_train, X_test, y_train, y_test = main(sc,sqlContext)
    print('Spark job completed')
    sc.stop()
