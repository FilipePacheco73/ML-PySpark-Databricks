# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md <i18n value="b9944704-a562-44e0-8ef6-8639f11312ca"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC # XGBoost
# MAGIC 
# MAGIC Up until this point, we have only used SparkML. Let's look a third party library for Gradient Boosted Trees. 
# MAGIC  
# MAGIC Ensure that you are using the <a href="https://docs.microsoft.com/en-us/azure/databricks/runtime/mlruntime" target="_blank">Databricks Runtime for ML</a> because that has Distributed XGBoost already installed. 
# MAGIC 
# MAGIC **Question**: How do gradient boosted trees differ from random forests? Which parts can be parallelized?
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson you:<br>
# MAGIC  - Use 3rd party libraries (XGBoost) to further improve your model

# COMMAND ----------

# DBTITLE 1,Imports
import pandas as pd

# COMMAND ----------

# DBTITLE 1,Get data and adjust to fit and create pyspark df
from sklearn.datasets import fetch_california_housing
df_ = fetch_california_housing()
df_input = pd.DataFrame(df_['data'],columns=df_['feature_names'])
df_output = pd.DataFrame(df_['target'],columns=df_['target_names'])
df = pd.concat([df_input,df_output],axis=1)
df_spark=spark.createDataFrame(df)
df_spark = df_spark.withColumn('HouseAge',df_spark.HouseAge.cast('string')) # Force as an example
display(df_spark)

# COMMAND ----------

# MAGIC %md <i18n value="3e08ca45-9a00-4c6a-ac38-169c7e87d9e4"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## Data Preparation
# MAGIC 
# MAGIC Let's go ahead and index all of our categorical features, and set our label to be **`log(price)`**.

# COMMAND ----------

from pyspark.sql.functions import log, col
from pyspark.ml.feature import StringIndexer, VectorAssembler

train_df, test_df = df_spark.withColumn('label', log(col('MedHouseVal'))).randomSplit([.8, .2], seed=42)

categorical_cols = [field for (field, dataType) in train_df.dtypes if dataType == 'string']
index_output_cols = [x + 'Index' for x in categorical_cols]

string_indexer = StringIndexer(inputCols=categorical_cols, outputCols=index_output_cols, handleInvalid='skip')

numeric_cols = [field for (field, dataType) in train_df.dtypes if ((dataType == 'double') & (field != 'MedHouseVal') & (field != 'label'))]
assembler_inputs = index_output_cols + numeric_cols
vec_assembler = VectorAssembler(inputCols=assembler_inputs, outputCol='features')

# COMMAND ----------

# MAGIC %md <i18n value="733cd880-143d-42c2-9f29-602e48f60efe"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ### Pyspark Distributed XGBoost
# MAGIC 
# MAGIC Let's create our distributed XGBoost model. While technically not part of MLlib, you can integrate <a href="https://databricks.github.io/spark-deep-learning/_modules/sparkdl/xgboost/xgboost.html" target="_blank">XGBoost</a> into your ML Pipelines. 
# MAGIC 
# MAGIC To use the distributed version of Pyspark XGBoost you can specify two additional parameters:
# MAGIC 
# MAGIC * **`num_workers`**: The number of workers to distribute over. Requires MLR 9.0+.
# MAGIC * **`use_gpu`**: Enable to utilize GPU based training for faster performance (optional).
# MAGIC 
# MAGIC **NOTE:** **`use_gpu`** requires an ML GPU runtime. Currently, at most one GPU per worker will be used when doing distributed training.

# COMMAND ----------

display(train_df)

# COMMAND ----------

from sparkdl.xgboost import XgboostRegressor
from pyspark.ml import Pipeline

params = {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 4, 'random_state': 42, 'missing': 0}

xgboost = XgboostRegressor(**params)

pipeline = Pipeline(stages=[string_indexer, vec_assembler, xgboost])
pipeline_model = pipeline.fit(train_df)

# COMMAND ----------

# MAGIC %md <i18n value="8d5f8c24-ee0b-476e-a250-95ce2d73dd28"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## Evaluate
# MAGIC 
# MAGIC Now we can evaluate how well our XGBoost model performed. Don't forget to exponentiate!

# COMMAND ----------

from pyspark.sql.functions import exp, col

log_pred_df = pipeline_model.transform(test_df)

exp_xgboost_df = log_pred_df.withColumn('prediction', exp(col('prediction')))

display(exp_xgboost_df.select('MedHouseVal','prediction'))

# COMMAND ----------

# MAGIC %md <i18n value="364402e1-8073-4b24-8e03-c7e2566f94d2"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC Compute some metrics.

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator

regression_evaluator = RegressionEvaluator(predictionCol='prediction', labelCol='MedHouseVal', metricName='rmse')

rmse = regression_evaluator.evaluate(exp_xgboost_df)
r2 = regression_evaluator.setMetricName('r2').evaluate(exp_xgboost_df)
print(f'RMSE is {rmse}')
print(f'R2 is {r2}')

# COMMAND ----------

# MAGIC %md <i18n value="21cf0d1b-c7a8-43c0-8eea-7677bb0d7847"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## Alternative Gradient Boosted Approaches
# MAGIC 
# MAGIC There are lots of other gradient boosted approaches, such as <a href="https://catboost.ai/" target="_blank">CatBoost</a>, <a href="https://github.com/microsoft/LightGBM" target="_blank">LightGBM</a>, vanilla gradient boosted trees in <a href="https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.classification.GBTClassifier.html?highlight=gbt#pyspark.ml.classification.GBTClassifier" target="_blank">SparkML</a>/<a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html" target="_blank">scikit-learn</a>, etc. Each of these has their respective <a href="https://towardsdatascience.com/catboost-vs-light-gbm-vs-xgboost-5f93620723db" target="_blank">pros and cons</a> that you can read more about.
