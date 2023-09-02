# Databricks notebook source
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

# MAGIC %md <i18n value="9af16c65-168c-4078-985d-c5f8991f171f"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## How to Handle Categorical Features?
# MAGIC 
# MAGIC We saw in the previous notebook that we can use StringIndexer/OneHotEncoder/VectorAssembler or RFormula.
# MAGIC 
# MAGIC **However, for decision trees, and in particular, random forests, we should not OHE our variables.**
# MAGIC 
# MAGIC There is an excellent <a href="https://towardsdatascience.com/one-hot-encoding-is-making-your-tree-based-ensembles-worse-heres-why-d64b282b5769#:~:text=One%2Dhot%20encoding%20categorical%20variables,importance%20resulting%20in%20poorer%20performance" target="_blank">blog</a> on this, and the essence is:
# MAGIC >>> "One-hot encoding categorical variables with high cardinality can cause inefficiency in tree-based methods. Continuous variables will be given more importance than the dummy variables by the algorithm, which will obscure the order of feature importance and can result in poorer performance."

# COMMAND ----------

from pyspark.ml.feature import StringIndexer

categorical_cols = [field for (field, dataType) in train_df.dtypes if dataType == 'string']
index_output_cols = [x + 'Index' for x in categorical_cols]

string_indexer = StringIndexer(inputCols=categorical_cols, outputCols=index_output_cols, handleInvalid='skip')

# COMMAND ----------

# MAGIC %md <i18n value="35e2f231-2ebb-4889-bc55-089200dd1605"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## VectorAssembler
# MAGIC 
# MAGIC Let's use the <a href="https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.VectorAssembler.html?highlight=vectorassembler#pyspark.ml.feature.VectorAssembler" target="_blank">VectorAssembler</a> to combine all of our categorical and numeric inputs.

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

# Filter for just numeric columns (and exclude target, our label)
numeric_cols = [field for (field, dataType) in train_df.dtypes if ((dataType == 'double') & (field != 'target'))]
# Combine output of StringIndexer defined above and numeric columns
assembler_inputs = index_output_cols + numeric_cols
vec_assembler = VectorAssembler(inputCols=assembler_inputs, outputCol='features')

# COMMAND ----------

# MAGIC %md <i18n value="2096f7aa-7fab-4807-b45f-fcbd0424a3e8"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## Decision Tree
# MAGIC 
# MAGIC Now let's build a <a href="https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.regression.DecisionTreeRegressor.html?highlight=decisiontreeregressor#pyspark.ml.regression.DecisionTreeRegressor" target="_blank">DecisionTreeRegressor</a> with the default hyperparameters.

# COMMAND ----------

from pyspark.ml.regression import DecisionTreeRegressor

dt = DecisionTreeRegressor(labelCol='target')

# COMMAND ----------

# MAGIC %md <i18n value="506ab7fa-0952-4c55-ad9b-afefb6469380"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## Fit Pipeline
# MAGIC 
# MAGIC The following cell is expected to error, but we subsequently fix this.

# COMMAND ----------

from pyspark.ml import Pipeline

# Combine stages into pipeline
stages = [string_indexer, vec_assembler, dt]
pipeline = Pipeline(stages=stages)

# Uncomment to perform fit
#pipeline_model = pipeline.fit(train_df)

# COMMAND ----------

# MAGIC %md <i18n value="d0791ff8-8d79-4d32-937d-9fcfbac4e9bd"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## maxBins
# MAGIC 
# MAGIC What is this parameter <a href="https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.regression.DecisionTreeRegressor.html?highlight=decisiontreeregressor#pyspark.ml.regression.DecisionTreeRegressor.maxBins" target="_blank">maxBins</a>? Let's take a look at the PLANET implementation of distributed decision trees to help explain the **`maxBins`** parameter.

# COMMAND ----------

# MAGIC %md <i18n value="1f9c229e-6f8c-4174-9927-c284e64e5753"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/DistDecisionTrees.png" height=500px>

# COMMAND ----------

# MAGIC %md <i18n value="3b7e60c3-22de-4794-9cd4-6713255b79a4"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC In Spark, data is partitioned by row. So when it needs to make a split, each worker has to compute summary statistics for every feature for  each split point. Then these summary statistics have to be aggregated (via tree reduce) for a split to be made. 
# MAGIC 
# MAGIC Think about it: What if worker 1 had the value **`32`** but none of the others had it. How could you communicate how good of a split that would be? So, Spark has a maxBins parameter for discretizing continuous variables into buckets, but the number of buckets has to be as large as the categorical variable with the highest cardinality.
# MAGIC 
# MAGIC Let's go ahead and increase maxBins to **`40`**.

# COMMAND ----------

dt.setMaxBins(100)
pipeline_model = pipeline.fit(train_df)

# COMMAND ----------

# MAGIC %md <i18n value="2426e78b-9bd2-4b7d-a65b-52054906e438"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## Feature Importance
# MAGIC 
# MAGIC Let's go ahead and get the fitted decision tree model, and look at the feature importance scores.

# COMMAND ----------

dt_model = pipeline_model.stages[-1]
display(dt_model)

# COMMAND ----------

dt_model.featureImportances

# COMMAND ----------

# MAGIC %md <i18n value="823c20ff-f20b-4853-beb0-4b324debb2e6"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## Interpreting Feature Importance
# MAGIC 
# MAGIC Hmmm... it's a little hard to know what feature 4 vs 11 is. Given that the feature importance scores are "small data", let's use Pandas to help us recover the original column names.

# COMMAND ----------

import pandas as pd

features_df = pd.DataFrame(list(zip(vec_assembler.getInputCols(), dt_model.featureImportances)), columns=["feature", "importance"])
features_df.sort_values('importance', ascending=False)

# COMMAND ----------

# MAGIC %md <i18n value="1fe0f603-add5-4904-964b-7288ae98b2e8"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## Why so few features are non-zero?
# MAGIC 
# MAGIC With SparkML, the default **`maxDepth`** is 5, so there are only a few features we could consider (we can also split on the same feature many times at different split points).
# MAGIC 
# MAGIC Let's use a Databricks widget to get the top-K features.

# COMMAND ----------

dbutils.widgets.text("top_k", "5")
top_k = int(dbutils.widgets.get("top_k"))

top_features = features_df.sort_values(["importance"], ascending=False)[:top_k]["feature"].values
print(top_features)

# COMMAND ----------

# MAGIC %md <i18n value="d9525bf7-b871-45c8-b0f9-dca5fd7ae825"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## Scale Invariant
# MAGIC 
# MAGIC With decision trees, the scale of the features does not matter. For example, it will split 1/3 of the data if that split point is 100 or if it is normalized to be .33. The only thing that matters is how many data points fall left and right of that split point - not the absolute value of the split point.
# MAGIC 
# MAGIC This is not true for linear regression, and the default in Spark is to standardize first. Think about it: If you measure shoe sizes in American vs European sizing, the corresponding weight of those features will be very different even those those measures represent the same thing: the size of a person's foot!

# COMMAND ----------

# MAGIC %md <i18n value="bad0dd6d-05ba-484b-90d6-cfe16a1bc11e"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## Apply model to test set

# COMMAND ----------

pred_df = pipeline_model.transform(test_df)

display(pred_df.select("features", "target", "prediction").orderBy("target", ascending=False))

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator

regression_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="target", metricName="rmse")

rmse = regression_evaluator.evaluate(pred_df)
r2 = regression_evaluator.setMetricName("r2").evaluate(pred_df)
print(f"RMSE is {rmse}")
print(f"R2 is {r2}")
