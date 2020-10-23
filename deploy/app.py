import tensorflow as tf
import os
import shutil
import time
import pandas as pd
import numpy as np
import uuid
from pyspark.sql.functions import col, pandas_udf, PandasUDFType
import pyspark as ps

import transform_data as tsf
import modeling as m


## Load Model Weights
@pandas_udf(ArrayType(FloatType()), PandasUDFType.SCALAR_ITER)
def predict_batch_udf(data_batch_iter):
  batch_size = 5
  learn_rate = 1e-04
  dropout=0.5
  model = m.create_model(learn_rate,dropout)
  weight_name = "185-weight-validation-loss-2.4813.hdf5"
  home_dir = "/home/winnie/petProjects/jockeyClub/model_training/output/callback/"
  weight_path = home_dir+weight_name
  model.load_weights(weight_path)

  for data_batch in data_batch_iter:
    dataset = tf.data.Dataset.from_tensor_slices(data_batch).batch(batch_size)
    preds = model.predict(dataset)
    print(preds)
    yield pd.Series(list(preds))


if __name__=="__main__":
    ## Start spark session
    sc = ps.SparkContext()
    sc.setLogLevel("ERROR")
    sqlContext = ps.sql.SQLContext(sc)
    print('Created a SparkContext')

    # Decrease the batch size of the Arrorw reader 
    # to avoid OOM errors on smaller instance types.
    sc.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "1024")

    ## Load Preprocessed Data
    processed_data = tsf.transform(sc,sqlContext,local=True)


