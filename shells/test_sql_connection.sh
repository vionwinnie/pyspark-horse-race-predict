#!/bin/sh

# Set Project ID
PROJECT_ID='pyspark-jockey-club'
REGION='us-central1'
cluster_name="${PROJECT_ID}-cluster-4"

# Define Copy Function
scp_func () {
    script_directory='/home/winnie/petProjects/jockeyClub/repo/inference'
    script_path="${script_directory}/$1"
    bucket_script_path="gs://${PROJECT_ID}/code/inference/$1"
    echo $script_path $bucket_script_path
    gsutil cp ${script_path} ${bucket_script_path}
}

script_name='connect_db.py'
bucket_script_path="gs://${PROJECT_ID}/code/inference/${script_name}"
scp_func ${script_name}
bucket_jar_path="gs://${PROJECT_ID}/code/inference/mysql-connector-java-5.1.45/mysql-connector-java-5.1.45-bin.jar"

gcloud dataproc jobs submit pyspark \
    ${bucket_script_path} \
    --cluster=${cluster_name} \
    --region=${REGION} \
    --jars ${bucket_jar_path}


