# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md <i18n value="b69335d5-86c7-40c5-b430-509a7444dae7"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC # Feature Store
# MAGIC 
# MAGIC The <a href="https://docs.databricks.com/applications/machine-learning/feature-store.html" target="_blank">Databricks Feature Store</a> is a centralized repository of features. It enables feature sharing and discovery across your organization and also ensures that the same feature computation code is used for model training and inference.
# MAGIC 
# MAGIC Check out Feature Store Python API documentation <a href="https://docs.databricks.com/dev-tools/api/python/latest/index.html#feature-store-python-api-reference" target="_blank">here</a>.
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson you will:<br>
# MAGIC  - Build a feature store with the Databricks Feature Store
# MAGIC  - Update feature tables
# MAGIC  - Perform batch scoring

# COMMAND ----------

# DBTITLE 1,Imports
import pandas as pd
import numpy as np
from pyspark.sql.functions import monotonically_increasing_id, lit, expr, rand
import uuid
from databricks import feature_store
from pyspark.sql.types import StringType, DoubleType
from databricks.feature_store import feature_table, FeatureLookup
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# COMMAND ----------

from sklearn.datasets import load_wine
df_ = load_wine()
df_input = pd.DataFrame(df_['data'],columns=df_['feature_names'])
df_output = pd.DataFrame(df_['target'],columns=['target'])
df = pd.concat([df_input,df_output],axis=1)
df_spark=spark.createDataFrame(df)
df_spark = df_spark.withColumn('ash',df_spark.ash.cast('string')) # Force as an example
df_spark = df_spark.withColumn('target',df_spark.target.cast('int')) # Force as an example
df_spark = df_spark.withColumn("index", monotonically_increasing_id())
display(df_spark)

# COMMAND ----------

# MAGIC %md <i18n value="5dcd3e8e-2553-429f-bbe1-aef0bc1ef0ab"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC Let's load in our data and generate a unique ID for each listing. The **`index`** column will serve as the "key" of the feature table and used to lookup features.

# COMMAND ----------

# MAGIC %md <i18n value="a04b29f6-e7a6-4e6a-875f-945edf938e9e"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC Create a new database and unique table name (in case you re-run the notebook multiple times)

# COMMAND ----------

cleaned_username = 'user_name'
spark.sql(f"CREATE DATABASE IF NOT EXISTS {cleaned_username}")
table_name = f"{cleaned_username}.wine_recognition_" + str(uuid.uuid4())[:6]
print(table_name)

# COMMAND ----------

# MAGIC %md <i18n value="a0712a39-b413-490f-a59e-dbd7f533e9a9"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC Let's start creating a <a href="https://docs.databricks.com/applications/machine-learning/feature-store.html#create-a-feature-table-in-databricks-feature-store" target="_blank">Feature Store Client</a> so we can populate our feature store.

# COMMAND ----------

fs = feature_store.FeatureStoreClient()

# help(fs.create_table)

# COMMAND ----------

# MAGIC %md <i18n value="90998fdb-87ed-4cdd-8844-fbd59ac5631f"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC #### Create Feature Table
# MAGIC 
# MAGIC Next, we can create the Feature Table using the **`create_table`** method.
# MAGIC 
# MAGIC This method takes a few parameters as inputs:
# MAGIC * **`name`**- A feature table name of the form **`<database_name>.<table_name>`**
# MAGIC * **`primary_keys`**- The primary key(s). If multiple columns are required, specify a list of column names.
# MAGIC * **`df`**- Data to insert into this feature table.  The schema of **`features_df`** will be used as the feature table schema.
# MAGIC * **`schema`**- Feature table schema. Note that either **`schema`** or **`features_df`** must be provided.
# MAGIC * **`description`**- Description of the feature table
# MAGIC * **`partition_columns`**- Column(s) used to partition the feature table.

# COMMAND ----------

## select numeric features and aggreagate the review scores, exclude target column 'target'
numeric_cols = [x.name for x in df_spark.schema.fields if (x.dataType == DoubleType()) and (x.name != 'target')]

@feature_table
def select_numeric_features(data):
  return data.select(['index'] + numeric_cols)

numeric_features_df = select_numeric_features(df_spark)
display(numeric_features_df)

# COMMAND ----------

fs.create_table(
  name = table_name,
  primary_keys=['index'],
  df = numeric_features_df,
  schema = numeric_features_df.schema,
  description = 'Numeric features of wine recognition dataset'
  )
