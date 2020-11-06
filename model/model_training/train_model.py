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

inputdir=sys.argv[1]
modeldir=sys.argv[2]
outputdir=sys.argv[3]

# encode ordinal columns: config, going, 
def build_pipeline_race():
    config_stringIdx = [StringIndexer(inputCol='config', outputCol='config_tsf')]
    going_stringIdx = [StringIndexer(inputCol='going', outputCol='going_tsf')]
    venue_stringIdx = [StringIndexer(inputCol='venue', outputCol='venue_tsf')]
    pipeline = Pipeline(stages=config_stringIdx+going_stringIdx+venue_stringIdx)
    return pipeline


# encode ordinal columns: horse_country_tsf, horse_type_tsf
def build_pipeline_runs():

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

def rename_columns(df, columns):
    if isinstance(columns, dict):
        for old_name, new_name in columns.items():
            df = df.withColumnRenamed(old_name, new_name)
        return df
    else:
        raise ValueError("'columns' should be a dict, like {'old_name_1':'new_name_1', 'old_name_2':'new_name_2'}")

def rename_col_dictionary(i):
    mapping_cols = { "draw": "draw_{}".format(i),
     "horse_age": "horse_age_{}".format(i),
     "horse_country": "horse_country_{}".format(i),
     "horse_type": "horse_type_{}".format(i),
     "horse_rating": "horse_rating_{}".format(i),
     "declared_weight": "declared_weight_{}".format(i),
     "actual_weight": "actual_weight_{}".format(i),
     "win_odds": "win_odds_{}".format(i),
     "result": "result_{}".format(i),
     "horse_country_tsf": "horse_country_tsf_{}".format(i),
     "horse_type_tsf": "horse_type_tsf_{}".format(i)}

    return mapping_cols

def reset_number(df):    
    for i in range(1,15):
        df = df.withColumn("result_{}_msk".format(i), f.when((f.col("result_{}".format(i)) == 1),1).otherwise(0)).drop("result_{}".format(i))    
    return df

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

    # check to see if we have NaN, then drop NaN
    race_df = race_df.select('race_id','venue', 'config', 'surface', 'distance', 'going', 'race_class')

    ## Instantiate the pipelne and fit with race data
    pipeline_race = build_pipeline_race()
    race_convert_model = pipeline_race.fit(race_df)
    print('transforming race dataframe...')
    processed_race_df = race_convert_model.transform(race_df).drop('config','going','venue')
    print('completed with {}'.format(len(processed_race_df.columns)))

    ## Saving runs data conversion pipeline
    race_modeldir = modeldir+'/race_convert_model.pb'
    race_convert_model.write().overwrite().save(race_modeldir)
#    processed_race_df.repartition(1).write.csv('X_races.csv',sep='|')

    
    ## Runs df
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

 #   runs_processed_df.repartition(1).write.csv('X_runs.csv',sep='|')
    print('completed with {}'.format(len(runs_processed_df.columns)))

    ## Saving runs data conversion pipeline
    runs_modeldir = modeldir+'/runs_convert_model.pb'
    runs_convert_model.write().overwrite().save(runs_modeldir)
    
    # dummy dataframe with all race_id
    unique_val = runs_processed_df.select("race_id").distinct().collect()
    all_race_id_list = [val['race_id'] for val in unique_val ]
    # Generate a pandas DataFrame
    pdf = pd.DataFrame({'race_id':all_race_id_list})
    print(pdf.shape)

    # Create a Spark DataFrame from a pandas DataFrame using Arrow
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

    join_df = processed_race_df.join(wide_runs_df, on='race_id', how='right')
    join_df = join_df.fillna(0)
    #join_df.repartition(1).write.csv('X_joint.csv',sep='|')

    ## Drop race_id
    selected = [s for s in join_df.columns if 'result' not in s]
    selected = [s for s in selected if 'draw' not in s]
    selected.remove("race_id")
    X_all = join_df.select(selected)
    print("selecting {} columns".format(len(selected)))
    
    #X_all.repartition(1).write.csv('X_all.csv',sep='|')
    input_cols_name = X_all.columns
    output_cols = [col + '_scaled' for col in X_all.columns]
    

    ## Vectorizer has finished running
    scaled_pipeline = build_pipeline_scaling(input_cols_name)
    #assemblers = [VectorAssembler(inputCols=[col], outputCol=col + "_vec").setHandleInvalid("skip") for col in input_cols_name]
    #scalers = [StandardScaler(inputCol=col+"_vec", outputCol=col + "_scaled") for col in input_cols_name]
    #pipeline = Pipeline(stages=assemblers+scalers)
    scalerModel = scaled_pipeline.fit(X_all)
    scaledData = scalerModel.transform(X_all)
    #scaledData.repartition(1).write.csv('X_all.csv',sep='|')
    ## Saving runs data conversion pipeline
    scaler_modeldir = modeldir+'/scaler_model.pb'
    scalerModel.write().overwrite().save(scaler_modeldir)
    print("vectorizer completed")


    return 1,2,3,4
"""    
    ## Fill NA with zero
    runs_df = final
    join_df = processed_race_df.join(runs_df, on='race_id', how='right')
    join_df = join_df.na.fill(0)
    join_df.repartition(1).write.csv('X_join_df.csv')
    #join_df.fillna(0)

    ## Drop race_id
    selected = [s for s in join_df.columns if 'result' not in s]
    selected = [s for s in selected if 'draw' not in s]
    selected.remove("race_id")
    X_all = join_df.select(selected)
    print("selecting {} columns".format(len(selected)))

    X_all.repartition(1).write.csv(outputdir+'/X_all.csv')
    #X_all.printSchema()

    ## Name output column
    input_cols_name = X_all.columns
    output_cols = [col + '_scaled' for col in X_all.columns]
    
    check = X_all.select("*").toPandas()
    print(check.head())

    #df.select('c').withColumn('isNull_c',F.col('c').isNull()).where('isNull_c = True').count()

    #X_all.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in X_all.columns]).show()


    ## Vectorizer has finished running
    assemblers = [VectorAssembler(inputCols=[col], outputCol=col + "_vec").setHandleInvalid("skip") for col in input_cols_name]
    scalers = [StandardScaler(inputCol=col+"_vec", outputCol=col + "_scaled") for col in input_cols_name]
    pipeline = Pipeline(stages=assemblers)

    scalerModel = pipeline.fit(X_all)
    scaledData = scalerModel.transform(X_all)

    ## Saving runs data conversion pipeline
    scaler_modeldir = modeldir+'/scaler_model.pb'
    scalerModel.write().overwrite().save(scaler_modeldir)
    print("vectorizer completed")

    selected = [s for s in scaledData.columns if 'scaled'  in s]
    scaledData_final = scaledData.select(selected)

    selected_y = [s for s in runs_df.columns if 'result' in s]
    y_all = runs_df.select(selected_y)
    y_all = reset_number(y_all)
    

    X_all_pdf = scaledData_final.select("*").toPandas()
    y_all_pdf = y_all.select("*").toPandas()
    X_all_pdf_devectorized = X_all_pdf.applymap(lambda x: x.item())

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_all_pdf_devectorized, y_all_pdf, test_size=0.2, random_state=42)

    X_train.to_csv(outputdir+'/X_train.csv')
    X_test.to_csv(outputdir+'/X_test.csv')
    y_train.to_csv(outputdir+'/y_train.csv')
    y_test.to_csv(outputdir+'/y_test.csv')

    return X_train, X_test, y_train, y_test
"""
    ## Put all features into a dense vector using a Vector Assembler
    #input_cols = list(scaledData_final.columns)
    #output_col = "data"

    ## Final Vectorizer
    #assembler = VectorAssembler(
    #        inputCols=input_cols,
    #        outputCol=output_col)

    #vectorizedData = assembler.transform(scaledData_final)
    #assembler_path=modeldir+'/final_vectorizer.pb'
    #assembler.save(assembler_path)

    #final_data = vectorizedData.select("data")










if __name__=="__main__":
    
    ## Start spark session
    sc = ps.SparkContext()
    sc.setLogLevel("ERROR")
    sqlContext = ps.sql.SQLContext(sc)
    print('Created a SparkContext')

    X_train, X_test, y_train, y_test = main(sc,sqlContext)
    print('Spark job completed')
    sc.stop()
