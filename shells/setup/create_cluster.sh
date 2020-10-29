#!/bin/bash

# Reference: https://cloud.google.com/dataproc/docs/tutorials/python-configuration

# Set Project ID
PROJECT_ID='pyspark-jockey-club'
REGION='us-central1'

# Copy the shell scripts from local machine over to bucket (storage)
gsutil cp conda-install.sh gs://${PROJECT_ID}/python/conda-install.sh
gsutil cp pip-install.sh gs://${PROJECT_ID}/python/pip-install.sh

# Set a Dataproc Cluster
# 1 master node, 2 worker node (minimum)
# Configure the packages - install scipy, tensorflow, pandas, scikit-learn
cluster_name="${PROJECT_ID}-cluster-1"
gcloud dataproc clusters create ${cluster_name} \
    --image-version=1.4 \
    --region=${REGION} \
    --metadata='CONDA_PACKAGES=scipy=1.1.0 tensorflow' \
    --metadata='PIP_PACKAGES=pandas==0.23.0 scipy==1.1.0 scikit-learn' \
    --bucket ${PROJECT_ID} \
    --master-machine-type n1-standard-4 \
    --worker-machine-type n1-standard-4 \
    --initialization-actions=gs://${PROJECT_ID}/python/conda-install.sh,gs://${PROJECT_ID}/python/pip-install.sh

