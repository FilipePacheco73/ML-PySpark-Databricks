# Databricks notebook source
# DBTITLE 1,Imports
import pandas as pd
import numpy as np

# COMMAND ----------

# DBTITLE 1,Get data and adjust to fit and create pyspark df
from sklearn.datasets import load_iris
df_ = load_iris()
df_input = pd.DataFrame(df_['data'],columns=df_['feature_names'])
df_output = pd.DataFrame(df_['target'],columns=['target'])
df = pd.concat([df_input,df_output],axis=1)
df_spark=spark.createDataFrame(df)
df_spark = df_spark.withColumnRenamed('sepal length (cm)','sepal_length').withColumnRenamed('sepal width (cm)','sepal_width').withColumnRenamed('petal length (cm)','petal_length').withColumnRenamed('petal width (cm)','petal_width')
df_spark = df_spark.withColumn('petal_width',df_spark.target.cast('string')) # Force as an example
display(df_spark)

# COMMAND ----------

# DBTITLE 1,Split train and test data
train_df, test_df = df_spark.randomSplit([.8, .2], seed=42)
print(train_df.cache().count())

# COMMAND ----------

#display(train_df.select('sepal_length','petal_length'))
train_df.select('sepal_length')

# COMMAND ----------

display(train_df.select('sepal_length','petal_length').summary())

# COMMAND ----------

display(train_df)

# COMMAND ----------

# DBTITLE 1,Categorical Variables - One Hot Encoder - OHE
from pyspark.ml.feature import OneHotEncoder, StringIndexer

categorical_cols = [field for (field, dataType) in train_df.dtypes if dataType == "string"]
index_output_cols = [x + "Index" for x in categorical_cols]
ohe_output_cols = [x + "OHE" for x in categorical_cols]

string_indexer = StringIndexer(inputCols=categorical_cols, outputCols=index_output_cols, handleInvalid="skip")
ohe_encoder = OneHotEncoder(inputCols=index_output_cols, outputCols=ohe_output_cols)

# COMMAND ----------

display(train_df)

# COMMAND ----------

# DBTITLE 1,Pipeline
from pyspark.ml import Pipeline

stages = [string_indexer, ohe_encoder, vec_assembler, lr]
pipeline = Pipeline(stages=stages)

pipeline_model = pipeline.fit(train_df)

# COMMAND ----------

# DBTITLE 1,Saving Models
import os
os.getcwd()
dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
#pipeline_model.write().overwrite().save(os.getcwd())

# COMMAND ----------

# DBTITLE 1,Loading Models
from pyspark.ml import PipelineModel

#saved_pipeline_model = PipelineModel.load(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())

# COMMAND ----------

# DBTITLE 1,Apply model to test set
pred_df = pipeline_model.transform(test_df)

display(pred_df.select('features','target','prediction'))

# COMMAND ----------

# DBTITLE 1,Evaluate Model
from pyspark.ml.evaluation import RegressionEvaluator

regression_evaluator = RegressionEvaluator(predictionCol="prediction",labelCol="target", metricName="rmse")

rmse = regression_evaluator.evaluate(pred_df)
r2 = regression_evaluator.setMetricName('r2').evaluate(pred_df)
print(f"RMSE is {rmse}")
print(f"R2 is {r2}")
