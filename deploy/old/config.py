## Root directory 
home_dir = ".."

## Data Directory
data_dir = home_dir+"/assets/data/"
races_data_path = data_dir+ "races.csv"
runs_data_path = data_dir + "runs.csv"

## Model Weight Directory
model_weight_dir = home_dir + "/assets/output/callback/"
model_weight_file_name = "185-weight-validation-loss-2.4813.hdf5"
model_weight_file_path = model_weight_dir + model_weight_file_name

## Data Pipeline Directory
pipeline_dir = home_dir + "/assets/pipeline/"
race_pipeline_path = pipeline_dir + "race_convert_model.pb"
runs_pipeline_path = pipeline_dir + "runs_convert_model.pb"
scaler_pipeline_path = pipeline_dir + "scaler_model.pb"

## Model Hyperparameters
model_hyperparam = { 'batch_size': 50,
                     'epoch':200,
                     'learning_rate':1e-04,
                     'dropout':0.5}


