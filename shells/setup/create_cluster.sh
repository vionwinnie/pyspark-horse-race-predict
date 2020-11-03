#!/bin/bash

# Reference: https://cloud.google.com/dataproc/docs/tutorials/python-configuration

# Set Project ID
PROJECT_ID='pyspark-jockey-club'
REGION='us-central1'

## Create a Hive Warehouse Bucket
#gsutil mb -l ${REGION} gs://${PROJECT_ID}-warehouse

## Create a Cloud SQL Instance
#gcloud sql instances create hive-metastore \
#    --database-version="MYSQL_5_7" \
#    --activation-policy=ALWAYS \
#    --zone ${ZONE}

# Copy the shell scripts from local machine over to bucket (storage)
#gsutil cp cloud-sql-proxy.sh gs://${PROJECT_ID}/init-actions/cloud-sql-proxy.sh
gsutil cp conda-install.sh gs://${PROJECT_ID}/init-actions/conda-install.sh
gsutil cp pip-install.sh gs://${PROJECT_ID}/init-actions/pip-install.sh

# Set a Dataproc Cluster
# 1 master node, 2 worker node (minimum)
# Configure the packages - install scipy, tensorflow, pandas, scikit-learn
cluster_name="${PROJECT_ID}-cluster-4"
#HIVE_DATA_BUCKET="${PROJECT_ID}-warehouse"
INSTANCE_NAME='pyspark-jockey-club'
echo $cluster_name
#echo $HIVE_DATA_BUCKET
echo $INSTANCE_NAME

gcloud dataproc clusters create ${cluster_name} \
    --image-version=1.4 \
    --region=${REGION} \
    --metadata='CONDA_PACKAGES=scipy=1.1.0 tensorflow' \
    --metadata='PIP_PACKAGES=pandas==0.23.0 scipy==1.1.0 scikit-learn' \
    --bucket ${PROJECT_ID} \
    --master-machine-type n1-standard-4 \
    --worker-machine-type n1-standard-4 \
    --initialization-actions=gs://${PROJECT_ID}/init-actions/conda-install.sh,gs://${PROJECT_ID}/init-actions/pip-install.sh
#     ,gs://${PROJECT_ID}/init-actions/cloud-sql-proxy.sh\
#    --scopes sql-admin \
#    --properties hive:hive.metastore.warehouse.dir=gs://${HIVE_DATA_BUCKET}/hive-warehouse \
 #   --metadata "hive-metastore-instance=${PROJECT_ID}:${REGION}:${INSTANCE_NAME}"
