import pandas as pd
import pyspark as ps

from pyspark.sql.functions import udf, col
from pyspark.sql.types import ArrayType, DoubleType
import pyspark.sql.functions as f

def to_array(col):
    def to_array_(v):
        return v.toArray().tolist()
    # Important: asNondeterministic requires Spark 2.3 or later
    # It can be safely removed i.e.
    # return udf(to_array_, ArrayType(DoubleType()))(col)
    # but at the cost of decreased performance
    return udf(to_array_, ArrayType(DoubleType())).asNondeterministic()(col)


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

