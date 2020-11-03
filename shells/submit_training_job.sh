#!/bin/bash

# Set Project ID
PROJECT_ID='pyspark-jockey-club'
REGION='us-central1'
cluster_name="${PROJECT_ID}-cluster-3"

# Define copy function
scp_func () {
    script_directory='/home/winnie/petProjects/jockeyClub/repo/model'
    script_path="${script_directory}/$1"
    bucket_script_path="gs://${PROJECT_ID}/code/training/$1"
    echo $script_path $bucket_script_path 
    gsutil cp ${script_path} ${bucket_script_path}
}

# Copy the training dat to bucket
gsutil cp /home/winnie/petProjects/jockeyClub/repo/assets/data/races.csv "gs://${PROJECT_ID}/data/races.csv"
gsutil cp /home/winnie/petProjects/jockeyClub/repo/assets/data/runs.csv "gs://${PROJECT_ID}/data/runs.csv"

# Copy the local script to bucket before submitting the pyspark job
script_name1='train_model.py'
bucket_script_path1="gs://${PROJECT_ID}/code/training/${script_name1}"
scp_func $script_name1
script_name2='transform_data.py'
bucket_script_path2="gs://${PROJECT_ID}/code/training/${script_name2}"
scp_func $script_name2
script_name3='utils.py'
bucket_script_path3="gs://${PROJECT_ID}/code/training/${script_name3}"
scp_func $script_name3
script_name4='create_model.py'
bucket_script_path4="gs://${PROJECT_ID}/code/training/${script_name4}"
scp_func $script_name4

# Set up job arguments
inputdir="gs://${PROJECT_ID}/data/"
modeldir="gs://${PROJECT_ID}/model"
outputdir="gs://${PROJECT_ID}/result"

gcloud dataproc jobs submit pyspark \
    ${bucket_script_path1} \
    --files=${bucket_script_path2},${bucket_script_path3},${bucket_script_path4} \
    --cluster=${cluster_name} \
    --region=${REGION} \
    --properties=spark.sql.debug.maxToStringFields=1000,spark.sql.optimizer.maxIterations=250 \
    -- ${inputdir} ${modeldir} ${outputdir}

