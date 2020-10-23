import tensorflow as tf
import os
import shutil
import time
import pandas as pd
import numpy as np
import uuid

from pyspark.sql.functions import col, pandas_udf, PandasUDFType
import pyspark as ps
from pyspark.conf import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.types import *


import transform_data as tsf
import modeling as m
import config

## Load Model Weights
@pandas_udf(ArrayType(FloatType()), PandasUDFType.SCALAR_ITER)
def predict_batch_udf(data_batch_iter):
  batch_size = config.model_hyperparam['batch_size']
  learn_rate = config.model_hyperparam['learning_rate']
  dropout= config.model_hyperparam['dropout']
  model = m.create_model(learn_rate,dropout)
  weight_path = config.model_weight_file_path
  model.load_weights(weight_path)

  for data_batch in data_batch_iter:
    dataset = tf.data.Dataset.from_tensor_slices(data_batch).batch(batch_size)
    preds = model.predict(dataset)
    yield pd.Series(list(preds))

def main(sc,sqlContext):

    ## Load Preprocessed Data
    df = tsf.transform(sc,sqlContext,local=True,debug=True)

    ## Create prediction df
    predictions_df = df.select(predict_batch_udf(col("data")).alias("prediction"))
    predictions_df.printSchema()

    predictions_df.show(2)

    



if __name__=="__main__":
    

    ## Set Config
    # Decrease the batch size of the Arrorw reader 
    # to avoid OOM errors on smaller instance types.
    sc = ps.SparkContext()
    conf = sc._conf.setAll([("spark.sql.execution.arrow.maxRecordsPerBatch", "1024")])
    sc.stop()

    ## Create a Spark Session
    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    sc = spark.sparkContext
    sc.setLogLevel("ERROR")
    sqlContext = ps.sql.SQLContext(sc)
    print('Created a SparkContext')
    
    ## Run the main function 
    main(sc,sqlContext)
    sc.stop()
