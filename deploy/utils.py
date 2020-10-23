import pyspark as ps
from pyspark.sql import functions as f
from pyspark.sql import types as t

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
