# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md <i18n value="2ab084da-06ed-457d-834a-1d19353e5c59"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC # Random Forests and Hyperparameter Tuning
# MAGIC 
# MAGIC Now let's take a look at how to tune random forests using grid search and cross validation in order to find the optimal hyperparameters.  Using the Databricks Runtime for ML, MLflow automatically logs your experiments with the SparkML cross-validator!
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson you:<br>
# MAGIC  - Tune hyperparameters using Grid Search
# MAGIC  - Optimize a SparkML pipeline

# COMMAND ----------

# DBTITLE 1,Imports - Installation
import pandas as pd
import numpy as np
#!pip install mlflow

# COMMAND ----------

# DBTITLE 1,Get data and adjust to fit and create pyspark df
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

from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml import Pipeline

categorical_cols = [field for (field, dataType) in train_df.dtypes if dataType == 'string']
index_output_cols = [x + 'Index' for x in categorical_cols]

string_indexer = StringIndexer(inputCols=categorical_cols, outputCols=index_output_cols, handleInvalid='skip')

numeric_cols = [field for (field, dataType) in train_df.dtypes if ((dataType == 'double') & (field != 'target'))]
assembler_inputs = index_output_cols + numeric_cols
vec_assembler = VectorAssembler(inputCols=assembler_inputs, outputCol='features')

rf = RandomForestRegressor(labelCol='target', maxBins=80)
stages = [string_indexer, vec_assembler, rf]
pipeline = Pipeline(stages=stages)

# COMMAND ----------

# MAGIC %md <i18n value="4561938e-90b5-413c-9e25-ef15ba40e99c"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## ParamGrid
# MAGIC 
# MAGIC First let's take a look at the various hyperparameters we could tune for random forest.
# MAGIC 
# MAGIC **Pop quiz:** what's the difference between a parameter and a hyperparameter?

# COMMAND ----------

print(rf.explainParams())

# COMMAND ----------

# MAGIC %md <i18n value="819de6f9-75d2-45df-beb1-6b59ecd2cfd2"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC There are a lot of hyperparameters we could tune, and it would take a long time to manually configure.
# MAGIC 
# MAGIC Instead of a manual (ad-hoc) approach, let's use Spark's <a href="https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.tuning.ParamGridBuilder.html?highlight=paramgridbuilder#pyspark.ml.tuning.ParamGridBuilder" target="_blank">ParamGridBuilder</a> to find the optimal hyperparameters in a more systematic approach.
# MAGIC 
# MAGIC Let's define a grid of hyperparameters to test:
# MAGIC   - **`maxDepth`**: max depth of each decision tree (Use the values **`2, 5`**)
# MAGIC   - **`numTrees`**: number of decision trees to train (Use the values **`5, 10`**)
# MAGIC 
# MAGIC **`addGrid()`** accepts the name of the parameter (e.g. **`rf.maxDepth`**), and a list of the possible values (e.g. **`[2, 5]`**).

# COMMAND ----------

from pyspark.ml.tuning import ParamGridBuilder

param_grid = (ParamGridBuilder()
              .addGrid(rf.maxDepth, [2, 5])
              .addGrid(rf.numTrees, [5, 10])
              .build())

# COMMAND ----------

# MAGIC %md <i18n value="9f043287-11b8-482d-8501-2f7d8b1458ea"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## Cross Validation
# MAGIC 
# MAGIC We are also going to use 3-fold cross validation to identify the optimal hyperparameters.
# MAGIC 
# MAGIC ![crossValidation](https://files.training.databricks.com/images/301/CrossValidation.png)
# MAGIC 
# MAGIC With 3-fold cross-validation, we train on 2/3 of the data, and evaluate with the remaining (held-out) 1/3. We repeat this process 3 times, so each fold gets the chance to act as the validation set. We then average the results of the three rounds.

# COMMAND ----------

# MAGIC %md <i18n value="ec0440ab-071d-4201-be86-5eeedaf80a4f"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC We pass in the **`estimator`** (pipeline), **`evaluator`**, and **`estimatorParamMaps`** to <a href="https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.tuning.CrossValidator.html?highlight=crossvalidator#pyspark.ml.tuning.CrossValidator" target="_blank">CrossValidator</a> so that it knows:
# MAGIC - Which model to use
# MAGIC - How to evaluate the model
# MAGIC - What hyperparameters to set for the model
# MAGIC 
# MAGIC We can also set the number of folds we want to split our data into (3), as well as setting a seed so we all have the same split in the data.

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator

evaluator = RegressionEvaluator(labelCol='target', predictionCol='prediction')

cv = CrossValidator(estimator=pipeline, evaluator=evaluator, estimatorParamMaps=param_grid, numFolds=3, seed=42)

# COMMAND ----------

# MAGIC %md <i18n value="673c9261-a861-4ace-b008-c04565230a8e"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC **Question**: How many models are we training right now?

# COMMAND ----------

cv_model = cv.fit(train_df)

# COMMAND ----------

# MAGIC %md <i18n value="c9bc1596-7b0f-4595-942c-109cfca51698"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## Parallelism Parameter
# MAGIC 
# MAGIC Hmmm... that took a long time to run. That's because the models were being trained sequentially rather than in parallel!
# MAGIC 
# MAGIC In Spark 2.3, a <a href="https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.tuning.CrossValidator.html?highlight=crossvalidator#pyspark.ml.tuning.CrossValidator.parallelism" target="_blank">parallelism</a> parameter was introduced. From the docs: **`the number of threads to use when running parallel algorithms (>= 1)`**.
# MAGIC 
# MAGIC Let's set this value to 4 and see if we can train any faster. The Spark <a href="https://spark.apache.org/docs/latest/ml-tuning.html" target="_blank">docs</a> recommend a value between 2-10.

# COMMAND ----------

#cv_model = cv.setParallelism(4).fit(train_df)

# COMMAND ----------

# MAGIC %md <i18n value="2d00b40f-c5e7-4089-890b-a50ccced34c6"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC **Question**: Hmmm... that still took a long time to run. Should we put the pipeline in the cross validator, or the cross validator in the pipeline?
# MAGIC 
# MAGIC It depends if there are estimators or transformers in the pipeline. If you have things like StringIndexer (an estimator) in the pipeline, then you have to refit it every time if you put the entire pipeline in the cross validator.
# MAGIC 
# MAGIC However, if there is any concern about data leakage from the earlier steps, the safest thing is to put the pipeline inside the CV, not the other way. CV first splits the data and then .fit() the pipeline. If it is placed at the end of the pipeline, we potentially can leak the info from hold-out set to train set.

# COMMAND ----------

cv = CrossValidator(estimator=rf, evaluator=evaluator, estimatorParamMaps=param_grid, numFolds=3, parallelism=4, seed=42)

stages_with_cv = [string_indexer, vec_assembler, cv]
pipeline = Pipeline(stages=stages_with_cv)

pipeline_model = pipeline.fit(train_df)

# COMMAND ----------

# MAGIC %md <i18n value="dede990c-2551-4c07-8aad-d697ae827e71"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC Let's take a look at the model with the best hyperparameter configuration

# COMMAND ----------

list(zip(cv_model.getEstimatorParamMaps(), cv_model.avgMetrics))

# COMMAND ----------

pred_df = pipeline_model.transform(test_df)

rmse = evaluator.evaluate(pred_df)
r2 = evaluator.setMetricName("r2").evaluate(pred_df)
print(f"RMSE is {rmse}")
print(f"R2 is {r2}")
