# Databricks notebook source
# DBTITLE 1,Imports
import pandas as pd
import numpy as np

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

display(train_df.select("HouseAge","MedHouseVal"))

# COMMAND ----------

display(train_df.select("HouseAge","MedHouseVal").summary())

# COMMAND ----------

display(train_df)

# COMMAND ----------

# DBTITLE 1,Format the input and output as vectors
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler

vec_assembler = VectorAssembler(inputCols=['AveBedrms'], outputCol='features')
vec_train_df = vec_assembler.transform(train_df)

lr = LinearRegression(featuresCol='features',labelCol='MedHouseVal')
lr_model = lr.fit(vec_train_df)

# COMMAND ----------

# DBTITLE 1,Inspect the model
m = lr_model.coefficients[0]
b = lr_model.intercept

print(f"The formula for the linear regression line is y = {m:.2f}x + {b:.2f}")

# COMMAND ----------

# DBTITLE 1,Apply model to test set
vec_test_df = vec_assembler.transform(test_df)

pred_df = lr_model.transform(vec_test_df)

pred_df.select("AveBedrms","features","MedHouseVal","prediction").show()

# COMMAND ----------

# DBTITLE 1,Evaluate Model
from pyspark.ml.evaluation import RegressionEvaluator

regression_evaluator = RegressionEvaluator(predictionCol="prediction",labelCol="MedHouseVal", metricName="rmse")

rmse = regression_evaluator.evaluate(pred_df)
print(f"RMSE is {rmse}")
