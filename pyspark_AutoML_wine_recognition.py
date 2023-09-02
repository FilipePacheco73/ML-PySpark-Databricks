# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md <i18n value="2630af5a-38e6-482e-87f1-1a1633438bb6"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC # AutoML
# MAGIC 
# MAGIC <a href="https://docs.databricks.com/applications/machine-learning/automl.html" target="_blank">Databricks AutoML</a> helps you automatically build machine learning models both through a UI and programmatically. It prepares the dataset for model training and then performs and records a set of trials (using HyperOpt), creating, tuning, and evaluating multiple models. 
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson you will:<br>
# MAGIC  - Use AutoML to automatically train and tune your models
# MAGIC  - Run AutoML in Python and through the UI
# MAGIC  - Interpret the results of an AutoML run

# COMMAND ----------

# MAGIC %md <i18n value="7aa84cf3-1b6c-4ba4-9249-00359ee8d70a"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC Currently, AutoML uses a combination of XGBoost and sklearn (only single node models) but optimizes the hyperparameters within each.

# COMMAND ----------

# DBTITLE 1,Imports
import pandas as pd
import numpy as np

# COMMAND ----------

from sklearn.datasets import load_wine
df_ = load_wine()
df_input = pd.DataFrame(df_['data'],columns=df_['feature_names'])
df_output = pd.DataFrame(df_['target'],columns=['target'])
df = pd.concat([df_input,df_output],axis=1)
df_spark=spark.createDataFrame(df)
df_spark = df_spark.withColumn('ash',df_spark.ash.cast('string')) # Force as an example
df_spark = df_spark.withColumn('target',df_spark.target.cast('int')) # Force as an example
display(df_spark)

# COMMAND ----------

# DBTITLE 1,Split train and test data
train_df, test_df = df_spark.randomSplit([.8, .2], seed=42)
print(train_df.cache().count())

# COMMAND ----------

# MAGIC %md <i18n value="1b5c8a94-3ac2-4977-bfe4-51a97d83ebd9"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC We can now use AutoML to search for the optimal <a href="https://docs.databricks.com/applications/machine-learning/automl.html#regression" target="_blank">regression</a> model. 
# MAGIC 
# MAGIC Required parameters:
# MAGIC * **`dataset`** - Input Spark or pandas DataFrame that contains training features and targets. If using a Spark DataFrame, it will convert it to a Pandas DataFrame under the hood by calling .toPandas() - just be careful you don't OOM!
# MAGIC * **`target_col`** - Column name of the target labels
# MAGIC 
# MAGIC We will also specify these optional parameters:
# MAGIC * **`primary_metric`** - Primary metric to select the best model. Each trial will compute several metrics, but this one determines which model is selected from all the trials. One of **`r2`** (default, R squared), **`mse`** (mean squared error), **`rmse`** (root mean squared error), **`mae`** (mean absolute error) for regression problems. For classification use **`log_loss`**, **`f1_score`**, **`score`**, **`recall_score`**, **`accuracy_score`**.
# MAGIC * **`timeout_minutes`** - The maximum time to wait for the AutoML trials to complete. **`timeout_minutes=None`** will run the trials without any timeout restrictions
# MAGIC * **`max_trials`** - The maximum number of trials to run. When **`max_trials=None`**, maximum number of trials will run to completion.

# COMMAND ----------

from databricks import automl

summary = automl.regress(train_df, target_col='target', primary_metric='rmse', timeout_minutes=5, max_trials=10)
#summary = automl.classify(train_df, target_col='target', primary_metric='log_loss', timeout_minutes=5, max_trials=10)

# COMMAND ----------

# MAGIC %md <i18n value="57d884c6-2099-4f34-b840-a4e873308ffe"/>
# MAGIC 
# MAGIC 
# MAGIC  
# MAGIC 
# MAGIC After running the previous cell, you will notice two notebooks and an MLflow experiment:
# MAGIC * **`Data exploration notebook`** - we can see a Profiling Report which organizes the input columns and discusses values, frequency and other information
# MAGIC * **`Best trial notebook`** - shows the source code for reproducing the best trial conducted by AutoML
# MAGIC * **`MLflow experiment`** - contains high level information, such as the root artifact location, experiment ID, and experiment tags. The list of trials contains detailed summaries of each trial, such as the notebook and model location, training parameters, and overall metrics.
# MAGIC 
# MAGIC Dig into these notebooks and the MLflow experiment - what do you find?
# MAGIC 
# MAGIC Additionally, AutoML shows a short list of metrics from the best run of the model.

# COMMAND ----------

print(summary.best_trial)

# COMMAND ----------

# MAGIC %md <i18n value="3c0cd1ec-8965-4af3-896d-c30938033abf"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC Now we can test the model that we got from AutoML against our test data. We'll be using <a href="https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#mlflow.pyfunc.spark_udf" target="_blank">mlflow.pyfunc.spark_udf</a> to register our model as a UDF and apply it in parallel to our test data.

# COMMAND ----------

# Load the best trial as an MLflow Model
import mlflow

model_uri = f"runs:/{summary.best_trial.mlflow_run_id}/model"

predict = mlflow.pyfunc.spark_udf(spark, model_uri)
pred_df = test_df.withColumn("prediction", predict(*test_df.drop("target").columns))
display(pred_df)

# COMMAND ----------

# DBTITLE 1,Regression Evaluation
from pyspark.ml.evaluation import RegressionEvaluator

regression_evaluator = RegressionEvaluator(predictionCol='prediction', labelCol='target', metricName='rmse')
rmse = regression_evaluator.evaluate(pred_df)
print(f'RMSE on test dataset: {rmse:.3f}')

# COMMAND ----------

# DBTITLE 1,Classification Evaluation
#from pyspark.ml.evaluation import MulticlassClassificationEvaluator

#multiclass_evaluator = MulticlassClassificationEvaluator(predictionCol='prediction', labelCol='target', metricName='f1')
#f1_score = multiclass_evaluator.evaluate(pred_df)
#print(f'F1 score on test dataset: {f1_score:.3f}')

# COMMAND ----------

# MAGIC %md-sandbox <i18n value="5802ff47-58b5-4789-973d-2fb855bf347a"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ### Model Registry
# MAGIC 
# MAGIC The MLflow Model Registry component is a centralized model store, set of APIs, and UI, to collaboratively manage the full lifecycle of an MLflow Model. It provides model lineage (which MLflow Experiment and Run produced the model), model versioning, stage transitions (e.g. from staging to production), annotations (e.g. with comments, tags), and deployment management (e.g. which production jobs have requested a specific model version).
# MAGIC 
# MAGIC Model registry has the following features:<br><br>
# MAGIC 
# MAGIC * **Central Repository:** Register MLflow models with the MLflow Model Registry. A registered model has a unique name, version, stage, and other metadata.
# MAGIC * **Model Versioning:** Automatically keep track of versions for registered models when updated.
# MAGIC * **Model Stage:** Assigned preset or custom stages to each model version, like “Staging” and “Production” to represent the lifecycle of a model.
# MAGIC * **Model Stage Transitions:** Record new registration events or changes as activities that automatically log users, changes, and additional metadata such as comments.
# MAGIC * **CI/CD Workflow Integration:** Record stage transitions, request, review and approve changes as part of CI/CD pipelines for better control and governance.
# MAGIC 
# MAGIC <div><img src="https://files.training.databricks.com/images/eLearning/ML-Part-4/model-registry.png" style="height: 400px; margin: 20px"/></div>
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_24.png"/> See <a href="https://mlflow.org/docs/latest/registry.html" target="_blank">the MLflow docs</a> for more details on the model registry.

# COMMAND ----------

# MAGIC %md <i18n value="1322cac5-9638-4cc9-b050-3545958f3936"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC Create a unique model name so you don't clash with other workspace users. 
# MAGIC 
# MAGIC Note that a registered model name must be a non-empty UTF-8 string and cannot contain forward slashes(/), periods(.), or colons(:).

# COMMAND ----------

model_name = 'wine_recognition_automl_cop'
model_name

# COMMAND ----------

# MAGIC %md <i18n value="0777e3f5-ba7c-41c4-a477-9f0a5a809664"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC Register the model.

# COMMAND ----------

model_uri

# COMMAND ----------

model_details = mlflow.register_model(model_uri=model_uri, name=model_name)

# COMMAND ----------

# DBTITLE 1,Check the status
from mlflow.tracking.client import MlflowClient

client = MlflowClient()
model_version_details = client.get_model_version(name=model_name, version=1)

model_version_details.status

# COMMAND ----------

# MAGIC %md <i18n value="aaac467f-3a52-4428-a119-8286cb0ac158"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ### Deploying a Model
# MAGIC 
# MAGIC The MLflow Model Registry defines several model stages: **`None`**, **`Staging`**, **`Production`**, and **`Archived`**. Each stage has a unique meaning. For example, **`Staging`** is meant for model testing, while **`Production`** is for models that have completed the testing or review processes and have been deployed to applications. 
# MAGIC 
# MAGIC Users with appropriate permissions can transition models between stages.

# COMMAND ----------

# MAGIC %md <i18n value="dff93671-f891-4779-9e41-a0960739516f"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC Now that you've learned about stage transitions, transition the model to the **`Production`** stage.
# MAGIC 
# MAGIC You can execute an automated CI/CD pipeline against it to test it before going into production.  Once that is completed, you can push that model into production.

# COMMAND ----------

client.transition_model_version_stage(
    name=model_details.name,
    version=model_details.version,
    stage="Production",
    archive_existing_versions=True # Archieve existing model in production 
)

# COMMAND ----------

# MAGIC %md <i18n value="4dc7e8b7-da38-4ce1-a238-39cad74d97c5"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC Fetch the model's current status.

# COMMAND ----------

model_version_details = client.get_model_version(
    name=model_details.name,
    version=model_details.version
)
print(f"The current model stage is: '{model_version_details.current_stage}'")

# COMMAND ----------

# MAGIC %md <i18n value="ba563293-bb74-4318-9618-a1dcf86ec7a3"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC Fetch the latest model using a **`pyfunc`**.  Loading the model in this way allows us to use the model regardless of the package that was used to train it.
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_24.png"/> You can load a specific version of the model too.

# COMMAND ----------

import mlflow.pyfunc

model_version_uri = f"models:/{model_name}/1"

print(f"Loading registered model version from URI: '{model_version_uri}'")
model_version_prod = mlflow.pyfunc.load_model(model_version_uri)

# COMMAND ----------

# MAGIC %md <i18n value="e1bb8ae5-6cf3-42c2-aebd-bde925a9ef30"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC Apply the model.

# COMMAND ----------

model_version_prod.predict(test_df.withColumn("prediction", predict(*test_df.drop("target").columns)).toPandas())

# COMMAND ----------

display(dbutils.fs.ls('/databricks-datasets'))

# COMMAND ----------

# MAGIC %sql 
# MAGIC show tables in samples.nyctaxi

# COMMAND ----------

# DBTITLE 1,Create a temp view
import pandas as pd
from sklearn.datasets import fetch_california_housing
df_ = fetch_california_housing()
df_input = pd.DataFrame(df_['data'],columns=df_['feature_names'])
df_output = pd.DataFrame(df_['target'],columns=df_['target_names'])
df = pd.concat([df_input,df_output],axis=1)
df_spark=spark.createDataFrame(df)
#df_spark = df_spark.withColumn('HouseAge',df_spark.HouseAge.cast('string')) # Force as an example
display(df_spark)
df_spark.createOrReplaceTempView("california_house_dataset")