import sys
import pyspark as ps
import warnings
import re
import pandas as pd
import numpy as np

from pyspark.sql import functions as f
from pyspark.sql import types as t
from pyspark.sql.types import StringType
from pyspark.ml.feature import StringIndexer, VectorAssembler,StandardScaler
from pyspark.ml import Pipeline, PipelineModel
import config 


def transform(sc,sqlContext,local,debug):

    """
    This function loads data from hive or load csv and undergo data transformation prior to inference

    - sc: Spark Context
    - sqlContext: Spark SQL Context
    - local: boolean flag


    return:

    - finalData: Spark Dataframe (n,104)

    """

    ## TODO: Query from hive race_df and runs_df
    if local:
        race_df = sqlContext.read.format('com.databricks.spark.csv')\
                        .options(header='true', inferschema='true')\
                        .load(config.races_data_path)
    
        runs_df = sqlContext.read.format('com.databricks.spark.csv')\
                        .options(header='true', inferschema='true')\
                        .load(config.runs_data_path)
    
    ## Clean Race Data
    # check to see if we have NaN, then drop NaN
    race_df = race_df.select('race_id','venue', 'config', 'surface', 'distance', 'going', 'race_class')
    
    ## Load Race Convert Model
    print('transforming race dataframe...')
    string_convert_model_fit = PipelineModel.load(config.race_pipeline_path)
    processed_race_df = string_convert_model_fit.transform(race_df)\
                        .drop('config','going','venue')
    print('transforming race dataframe completed')
    if debug:
        processed_race_df.show(2)
    
    ## Clean Run Data
    runs_df = runs_df.select('race_id', 'draw',
                       'horse_age', 'horse_country', 'horse_type', 'horse_rating', 'declared_weight', 'actual_weight', 'win_odds',
                       'result')
    
    ## drop strange observation
    # not sure why, but we got some strange draw in the dataset. Maximum shall be 14
    runs_df = runs_df.filter("draw<=14")
    runs_df.count()
    
    ## Load runs_convert_model
    print('transforming runs dataframe')
    runs_convert_model_fit = PipelineModel.load(config.runs_pipeline_path)
    runs_processed_df = runs_convert_model_fit.transform(runs_df).drop('horse_country','horse_type')
    print('transforming runs dataframe completed')
    if debug:
        runs_processed_df.show(2)
    
    ## Pivot the runs dataframe from long to wide format
    from utils import rename_col_dictionary, rename_columns
    # dummy dataframe with all race_id
    unique_val = runs_processed_df.select("race_id").distinct().collect()
    all_race_id_list = [val['race_id'] for val in unique_val ]
    # Generate a pandas DataFrame
    pdf = pd.DataFrame({'race_id':all_race_id_list})
    # Create a Spark DataFrame from a pandas DataFrame using Arrow
    final = sqlContext.createDataFrame(pdf)
    
    for i in range(1,15):
        print(i)
        tmp = runs_processed_df.filter("draw={}".format(i))
        mapping_cols = rename_col_dictionary(i)    
        tmp = rename_columns(tmp,mapping_cols)
        
        if tmp ==1:
            final = tmp
        else:
            final = final.join(tmp,["race_id"],how='left')
        
    ## Fill NA with zero
    runs_df = final.fillna(0)
   
    ## Join Race and Runs Dataframe together
    join_df = processed_race_df.join(runs_df, on='race_id', how='right')
    
    ## Drop race_id
    selected = [s for s in join_df.columns if 'result' not in s]
    selected = [s for s in selected if 'draw' not in s]
    selected.remove("race_id")
    
    X_all = join_df.select(selected)
    X_all.printSchema()
    
    ## Name output column
    input_cols_name = X_all.columns
    output_cols = [col + '_scaled' for col in X_all.columns]
    
    ## Scaling the variables using saved Scaler 
    scaler_fit = PipelineModel.load(config.scaler_pipeline_path)
    scaledData = scaler_fit.transform(X_all)
    selected = [s for s in scaledData.columns if 'scaled'  in s]
    scaledData_final = scaledData.select(selected)
    if debug:
        scaledData_final.show(2)

    ## Put all features into a dense vector using a Vector Assembler
    input_cols = list(scaledData_final.columns)
    output_col = "data"
    
    assembler = VectorAssembler(
            inputCols=input_cols,
            outputCol=output_col)

    vectorizedData = assembler.transform(scaledData_final)
    
    final_data = vectorizedData.select("data")
    
    return final_data
    
if __name__=="__main__":
    ## Start a PySpark session
    sc = ps.SparkContext()
    sc.setLogLevel("ERROR")
    sqlContext = ps.sql.SQLContext(sc)
    print('Created a SparkContext')
    local=True
    debug=True
    ## Data Processing
    print('Starting the Data Processing')
    processed_data = transform(sc,sqlContext,local,debug)
    processed_data.printSchema()

    ## Stop Spark Context
    sc.stop()
