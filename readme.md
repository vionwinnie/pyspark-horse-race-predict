# Integrating data pipeline with Tensorflow Model Deployment using PySpark
- Presentation for PyCon HK 2020 Fall Session (Cantonese Track)
- Speaker: Winnie Yeung
- [Slide Deck](https://bit.ly/35eXcqD)

## Problem Description
How can we predict the winning horse out of each race at Jockey Club horse race?

## Tech Stack
- GCP Dataproc, PySpark 2.4.7, Pandas, Tensorflow 2.0, Java

## Running jobs 
- Individual Script: ```pyspark < script.py```
- Submit job on GCP Dataproc: /shells/ ```nohup ./submit_inference_job.sh &```

## Credits
- Lantana Camara Dataset on Kaggle: https://www.kaggle.com/lantanacamara/hong-kong-horse-racing
- Cullen Sun's Tensorflow Model Design on this dataset: https://www.kaggle.com/cullensun/deep-learning-model-for-hong-kong-horse-racing/
- Jockey Club webscraping package: https://github.com/jaloo555/HK-Horse-Racing-Data-Scraper

## Useful links:
- Databricks Model Inferencing Guides: https://docs.databricks.com/applications/machine-learning/model-inference/dl-model-inference.html
- Medium posts on PySpark Pipeline Data Transformation: 
    - https://medium.com/@faiyaz.hasan1/tuning-and-training-machine-learning-models-using-pyspark-on-cloud-dataproc-b084e0105334



