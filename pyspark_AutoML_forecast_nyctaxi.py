# Databricks notebook source
# DBTITLE 1,Data loading
# MAGIC %sql
# MAGIC SELECT * FROM samples.nyctaxi.trips

# COMMAND ----------

df_spark = _sqldf
df_spark.summary().display()

# COMMAND ----------

from pyspark.sql.functions import date_format, dayofweek, col, lit

df_spark = df_spark.withColumn('dow', dayofweek(col('tpep_pickup_datetime'))).withColumn('tpep_pickup_date', date_format(col('tpep_pickup_datetime'), 'yyyy-MM-dd'))

# COMMAND ----------

df_spark.display()

# COMMAND ----------

df_spark_grouped = df_spark.groupBy('tpep_pickup_date').sum('trip_distance','fare_amount').sort('tpep_pickup_date').withColumn('tpep_pickup_date',df_spark.tpep_pickup_date.cast('date')).withColumn('dow', dayofweek(col('tpep_pickup_date')))
df_spark_grouped.display()

# COMMAND ----------

train_df = df_spark_grouped.filter(df_spark_grouped.tpep_pickup_date < '2016-02-20')
test_df = df_spark_grouped.filter(df_spark_grouped.tpep_pickup_date >= '2016-02-20')

# COMMAND ----------

from databricks import automl

help(automl.regress)

# COMMAND ----------

help(automl.classify)

# COMMAND ----------

from databricks import automl

summary = automl.forecast(train_df, target_col='sum(fare_amount)', time_col='tpep_pickup_date', frequency='days', horizon=10, primary_metric='smape', timeout_minutes=120)

# COMMAND ----------

print(summary.best_trial)

# COMMAND ----------

# Load the best trial as an MLflow Model
import mlflow

model_uri = f'runs:/{summary.best_trial.mlflow_run_id}/model'

predict = mlflow.pyfunc.spark_udf(spark, model_uri)
pred_df = test_df.withColumn('prediction', predict(*test_df.drop('sum(fare_amount)').columns))
display(pred_df)
