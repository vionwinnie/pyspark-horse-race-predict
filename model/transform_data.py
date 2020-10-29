import pyspark as ps
from pyspark.sql.functions import isnan, when, count, col
from pyspark.sql import functions as f
from pyspark.sql import types as t
from pyspark.sql.types import StringType
from pyspark.ml.feature import StringIndexer, VectorAssembler,StandardScaler
from pyspark.ml import Pipeline,PipelineModel
from pyspark.sql.session import SparkSession

## Loading Packages and Dependencies
import sys
import pyspark as ps
import warnings
import re
import pandas as pd
import numpy as np

## Import Utils functions
from utils import rename_col_dictionary,reset_number,rename_columns


def build_pipeline_race():
    """
    encode ordinal columns: config,going,venue
    """
    config_stringIdx = [StringIndexer(inputCol='config', outputCol='config_tsf')]
    going_stringIdx = [StringIndexer(inputCol='going', outputCol='going_tsf')]
    venue_stringIdx = [StringIndexer(inputCol='venue', outputCol='venue_tsf')]
    pipeline = Pipeline(stages=config_stringIdx+going_stringIdx+venue_stringIdx)
    return pipeline

def build_pipeline_runs():
    """
    encode ordinal columns: horse_country_tsf, horse_type_tsf
    """
    horse_country_stringIdx = [StringIndexer(inputCol='horse_country', outputCol='horse_country_tsf')]
    horse_type_stringIdx = [StringIndexer(inputCol='horse_type', outputCol='horse_type_tsf')]
    pipeline = Pipeline(stages=horse_country_stringIdx+horse_type_stringIdx)
    return pipeline

def build_pipeline_scaling(cols):
    ## Vectorizer has finished running
    assembler = [VectorAssembler(inputCols=cols,outputCol="features").setHandleInvalid("skip")]
    scaler = [StandardScaler(inputCol="features", outputCol="scaled_features")]
    pipeline = Pipeline(stages=assembler+scaler)
    return pipeline

def transform_race_df(race_df):

    """
    input:
    - race_df: Spark dataframe

    return:
    - pipeline for data transformation
    - transformed race_df: Spark Dataframe
    """

    # check to see if we have NaN, then drop NaN
    race_df = race_df.select('race_id','venue', 'config', 'surface', 'distance', 'going', 'race_class')

    ## Instantiate the pipelne and fit with race data
    pipeline_race = build_pipeline_race()
    race_convert_model = pipeline_race.fit(race_df)
    print('transforming race dataframe...')
    processed_race_df = race_convert_model.transform(race_df).drop('config','going','venue')
    print('completed with {}'.format(len(processed_race_df.columns)))

    return race_convert_model,processed_race_df

def transform_runs_df(runs_df,sparkSession=None):
    """
    Input:
    runs_df: spark dataframe

    Output:
    race_convert_model: Pipeline object
    processed_runs_df: spark dataframe post-processed
    """

    runs_df = runs_df.select('race_id', 'draw',
                   'horse_age', 'horse_country', 'horse_type', 'horse_rating', 'declared_weight', 'actual_weight', 'win_odds',
                   'result')


    # not sure why, but we got some strange draw in the dataset. Maximum shall be 14
    runs_df = runs_df.filter("draw<=14")

    ## Dropping NA results
    runs_df = runs_df.na.drop()
    pipeline_runs = build_pipeline_runs()
    runs_convert_model = pipeline_runs.fit(runs_df)
    print('transforming run dataframe...')
    runs_processed_df = runs_convert_model.transform(runs_df).drop('horse_country','horse_type')

    print('completed with {}'.format(len(runs_processed_df.columns)))

    # Transforming from long to wide dataframe
    unique_val = runs_processed_df.select("race_id").distinct().collect()
    all_race_id_list = [val['race_id'] for val in unique_val ]
    # Generate a pandas DataFrame
    pdf = pd.DataFrame({'race_id':all_race_id_list})
    print(pdf.shape)

    # Create a Spark DataFrame from a pandas DataFrame using Arrow
    sc = None
    sqlContext=None

    if sparkSession is None:
        sparkSession = SparkSession.builder.getOrCreate()
        sc = sparkSession.sparkContext
        sqlContext = ps.sql.SQLContext(sc)

    final = sqlContext.createDataFrame(pdf)

    for i in range(1,15):
        print(i)
        tmp = runs_processed_df.filter("draw={}".format(i))
        print(tmp.count())
        mapping_cols = rename_col_dictionary(i)    
        tmp = rename_columns(tmp,mapping_cols)
        final = final.join(tmp,["race_id"],how='left')
        print("col size:{}".format(len(final.columns)))

    wide_runs_df = final
    wide_runs_df = wide_runs_df.fillna(0)
    #wide_runs_df.repartition(1).write.csv('X_wide_runs.csv',sep='|')

    return runs_convert_model,wide_runs_df

def join_runs_and_race(runs_df,race_df):
    join_df = race_df.join(runs_df, on='race_id', how='right').fillna(0)
    return join_df


def scale_select_col(join_df):
    """
    Input:
    - join_df: Spark dataframe

    Output:
    - scalerModel: Pipeline Object
    - vectorized_scaled_join_df: Spark dataframe
    """

    ## Drop race_id
    selected = [s for s in join_df.columns if 'result' not in s]
    selected = [s for s in selected if 'draw' not in s]
    selected.remove("race_id")
    print("selecting {} columns".format(len(selected)))
    
    scaled_pipeline = build_pipeline_scaling(selected)
    scalerModel = scaled_pipeline.fit(join_df)
    scaledData = scalerModel.transform(join_df)

    return scalerModel,scaledData






