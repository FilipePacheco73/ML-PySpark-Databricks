# Databricks notebook source
# DBTITLE 1,Imports - Installation
import pandas as pd
import numpy as np
!pip install mlflow

# COMMAND ----------

# DBTITLE 1,Get data and adjust to fit and create pyspark df
from sklearn.datasets import fetch_california_housing
df_ = fetch_california_housing()
df_input = pd.DataFrame(df_['data'],columns=df_['feature_names'])
df_output = pd.DataFrame(df_['target'],columns=df_['target_names'])
df = pd.concat([df_input,df_output],axis=1)
df_spark=spark.createDataFrame(df)
display(df_spark)

# COMMAND ----------

# DBTITLE 1,Split train and test data
train_df, test_df = df_spark.randomSplit([.8, .2], seed=42)
print(train_df.cache().count())

# COMMAND ----------

# MAGIC %md <i18n value="9ab8c080-9012-4f38-8b01-3846c1531a80"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ### MLflow Tracking
# MAGIC 
# MAGIC MLflow Tracking is a logging API specific for machine learning and agnostic to libraries and environments that do the training.  It is organized around the concept of **runs**, which are executions of data science code.  Runs are aggregated into **experiments** where many runs can be a part of a given experiment and an MLflow server can host many experiments.
# MAGIC 
# MAGIC You can use <a href="https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.set_experiment" target="_blank">mlflow.set_experiment()</a> to set an experiment, but if you do not specify an experiment, it will automatically be scoped to this notebook.

# COMMAND ----------

# MAGIC %md <i18n value="82786653-4926-4790-b867-c8ccb208b451"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ### Track Runs
# MAGIC 
# MAGIC Each run can record the following information:<br><br>
# MAGIC 
# MAGIC - **Parameters:** Key-value pairs of input parameters such as the number of trees in a random forest model
# MAGIC - **Metrics:** Evaluation metrics such as RMSE or Area Under the ROC Curve
# MAGIC - **Artifacts:** Arbitrary output files in any format.  This can include images, pickled models, and data files
# MAGIC - **Source:** The code that originally ran the experiment
# MAGIC 
# MAGIC **NOTE**: For Spark models, MLflow can only log PipelineModels.

# COMMAND ----------

token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
dbutils.fs.put("file:///root/.databrickscfg","[DEFAULT]\nhost=https://community.cloud.databricks.com\ntoken = "+token,overwrite=True)

# COMMAND ----------

# DBTITLE 1,MLflow Tracking
import mlflow
import mlflow.spark
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator

with mlflow.start_run(run_name="LR-Single-Feature") as run:

  #Define pipeline
  vec_assembler = VectorAssembler(inputCols=['MedInc'], outputCol='features')
  lr = LinearRegression(featuresCol='features', labelCol='MedHouseVal')
  pipeline = Pipeline(stages=[vec_assembler, lr])
  pipeline_model = pipeline.fit(train_df)

  # Log parameters
  mlflow.log_param('label','MedHouseVal')
  mlflow.log_param('features','MedInc')

  # Log model
  mlflow.spark.log_model(pipeline_model, 'model', input_example=train_df.limit(5).toPandas())

  # Evaluate predictions
  pred_df = pipeline_model.transform(test_df)
  regression_evaluator = RegressionEvaluator(predictionCol='prediction', labelCol='MedHouseVal', metricName='rmse')
  rmse = regression_evaluator.evaluate(pred_df)

  # Log metrics
  mlflow.log_metric('rmse', rmse)  

# COMMAND ----------

# MAGIC %md <i18n value="44bc7cac-de4a-47e7-bfff-6d2eb58172cd"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC There, all done! Let's go through the other two linear regression models and then compare our runs. 
# MAGIC 
# MAGIC **Question**: Does anyone remember the RMSE of the other runs?
# MAGIC 
# MAGIC Next let's build our linear regression model but use all of our features.

# COMMAND ----------

from pyspark.ml.feature import RFormula

with mlflow.start_run(run_name='LR-All-Features') as run:
  
  # Create pipeline
  r_formula = RFormula(formula='MedHouseVal ~ .', featuresCol='features', labelCol='MedHouseVal', handleInvalid='skip')
  lr = LinearRegression(labelCol='MedHouseVal', featuresCol='features')
  pipeline = Pipeline(stages=[r_formula, lr])
  pipeline_model = pipeline.fit(train_df)
  
  # Log pipeline
  mlflow.spark.log_model(pipeline_model, 'model', input_example=train_df.limit(5).toPandas())
  
  # Log parameter
  mlflow.log_param('label','MedHouseVal')
  mlflow.log_param('features','all_features')
  
  # Create predictions and metrics
  pred_df = pipeline_model.transform(test_df)
  regression_evaluator = RegressionEvaluator(labelCol='MedHouseVal', predictionCol='prediction')
  rmse = regression_evaluator.setMetricName('rmse').evaluate(pred_df)
  r2 = regression_evaluator.setMetricName('r2').evaluate(pred_df)
  
  # Log both metrics
  mlflow.log_metric('rmse', rmse)
  mlflow.log_metric('r2', r2)

# COMMAND ----------

# MAGIC %md
# MAGIC Finally, we will use Linear Regression to predict the log of the price, due to its log normal distribution.
# MAGIC 
# MAGIC We'll also practice logging artifacts to keep a visual of our log normal histogram.

# COMMAND ----------

from pyspark.sql.functions import col, log, exp
import matplotlib.pyplot as plt

with mlflow.start_run(run_name='LR-Log-MedHouseVal') as run:
  
  # Take log of MedHouseVal
  log_train_df = train_df.withColumn('log_MedHouseVal', log(col('MedHouseVal')))
  log_test_df = test_df.withColumn('log_MedHouseVal', log(col('MedHouseVal')))
  
  # Log parameter
  mlflow.log_param('label','log_MedHouseVal')
  mlflow.log_param('features','all_features')
  
  # Create pipeline
  r_formula = RFormula(formula='log_MedHouseVal ~. - MedHouseVal', featuresCol='features', labelCol='log_MedHouseVal', handleInvalid='skip')
  lr = LinearRegression(labelCol='log_MedHouseVal', predictionCol='log_prediction')
  pipeline = Pipeline(stages=[r_formula, lr])
  pipeline_model = pipeline.fit(log_train_df)
  
  # Log model
  mlflow.spark.log_model(pipeline_model, 'log-model', input_example=log_train_df.limit(5).toPandas())
  
  # Make predictions
  pred_df = pipeline_model.transform(log_test_df)
  exp_df = pred_df.withColumn('prediction', exp(col('log_prediction')))
  
  # Evaluate predictions
  rmse = regression_evaluator.setMetricName('rmse').evaluate(exp_df)
  r2 = regression_evaluator.setMetricName('r2').evaluate(exp_df)
  
  # Log metrics
  mlflow.log_metric('rmse', rmse)
  mlflow.log_metric('r2', r2)
  
  # Log artifact
  plt.clf()
  
  log_train_df.toPandas().hist(column='log_MedHouseVal', bins=100)
  fig = plt.gcf()
  mlflow.log_figure(fig, 'username' + '_log_normal.png')
  plt.show()

# COMMAND ----------

# MAGIC %md <i18n value="66785d5e-e1a7-4896-a8a9-5bfcd18acc5c"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC That's it! Now, let's use MLflow to easily look over our work and compare model performance. You can either query past runs programmatically or use the MLflow UI.

# COMMAND ----------

# MAGIC %md <i18n value="0b1a68e1-bd5d-4f78-a452-90c7ebcdef39"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ### Querying Past Runs
# MAGIC 
# MAGIC You can query past runs programmatically in order to use this data back in Python.  The pathway to doing this is an **`MlflowClient`** object.

# COMMAND ----------

from mlflow.tracking import MlflowClient

client = MlflowClient()

# COMMAND ----------

#client.list_experiments()

# COMMAND ----------

experiment_id = run.info.experiment_id
runs_df = mlflow.search_runs(experiment_id)

display(runs_df)

# COMMAND ----------

runs = client.search_runs(experiment_id, order_by=["attributes.start_time desc"], max_results=1)
runs[0].data.metrics

# COMMAND ----------

runs[0].info.run_id

# COMMAND ----------

# MAGIC %md <i18n value="cfbbd060-6380-444f-ba88-248e10a56559"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC Examine the results in the UI.  Look for the following:<br><br>
# MAGIC 
# MAGIC 1. The **`Experiment ID`**
# MAGIC 2. The artifact location.  This is where the artifacts are stored in DBFS.
# MAGIC 3. The time the run was executed.  **Click this to see more information on the run.**
# MAGIC 4. The code that executed the run.
# MAGIC 
# MAGIC 
# MAGIC After clicking on the time of the run, take a look at the following:<br><br>
# MAGIC 
# MAGIC 1. The Run ID will match what we printed above
# MAGIC 2. The model that we saved, included a pickled version of the model as well as the Conda environment and the **`MLmodel`** file.
# MAGIC 
# MAGIC Note that you can add notes under the "Notes" tab to help keep track of important information about your models. 
# MAGIC 
# MAGIC Also, click on the run for the log normal distribution and see that the histogram is saved in "Artifacts".

# COMMAND ----------

# MAGIC %md <i18n value="63ca7584-2a86-421b-a57e-13d48db8a75d"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ### Load Saved Model
# MAGIC 
# MAGIC Let's practice <a href="https://www.mlflow.org/docs/latest/python_api/mlflow.spark.html" target="_blank">loading</a> our logged log-normal model.

# COMMAND ----------

model_path = f"runs:/{run.info.run_id}/log-model"
loaded_model = mlflow.spark.load_model(model_path)

display(loaded_model.transform(test_df))
