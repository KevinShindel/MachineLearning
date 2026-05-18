### Setup Model Registry with UC

```python
import mlflow


# point to UC model registry
mlflow.set_registry_uri('databricks-uc')
client = mlflow.MlflowClient()

# helper function for getting last version of model

def get_latest_model_version(model_name):
    model_version_infos = client.search_model_versions(f"name = '{model_name}'")
    returm max([model_version_infos.version for model_version_infos in model_version_infos])
```


### Register Common MLflow experiment

```python
from sklearn.tree import DecisionTreeClassifier
from mlflow.models import infer_signature

# convert data into pandas
X_train_pdf = features_df.drop(primary_key).toPandas()
Y_train_pdf = response_df.drop(primary_key).toPandas()

clf = DecisionTreeClassifier(max_depth=3, random_state=42)

# use 3-lvl namespace for model name
model_name = f"{catalog_name}.{schema_name}.ml_model"

with mlflow.start_run(run_name="Model-Batch-Deployment-Demo") as mlflow_run:
    
    mlflow.sklearn.autolog(log_input_examples=True,
                           log_models=False,
                           log_post_training_metrics=True, 
                           log_model_signatures=True)
    
    # fit the model and log it to MLflow with signature
    clf.fit(X_train_pdf, Y_train_pdf)
    
    # log model and push to registry with signature
    signature = infer_signature(X_train_pdf, clf.predict(X_train_pdf))
    mlflow.sklearn.log_model(clf, 
                             artifact_path="decision_tree_model",
                             registered_model_name=model_name,
                             signature=signature)

    # ste model alias to "Baseline"
    client.update_model_version(model_name=model_name,
                                version=get_latest_model_version(model_name),
                                description="Baseline model for batch deployment demo",
                                aliases=["Baseline"])
```


### Predictions with MLflow model registry

```python
from mlflow.tracking import MlflowClient
import mlflow
import  pyspark.sql.functions as F


logged_model_path = 'runs:/<run_id>/decision_tree_model'

# load model as a spark_udf
loaded_model = mlflow.pyfunc.spark_udf(spark, model_uri=logged_model_path, result='double')

# use the model for inference
predictions_df = features_df.withColumn("prediction", loaded_model(*features_df.columns))
```

### Same approach by for Pandas UDF

```python
import mlflow

logged_model_path = 'runs:/<run_id>/decision_tree_model'

# load model as a pandas_udf
loaded_model = mlflow.pyfunc.load_model(spark, model_uri=logged_model_path, result='double')

predicted = loaded_model.predict(X_train_pdf)
```

### Load model from registry with version

```python
from mlflow.tracking import MlflowClient
import mlflow

client = MlflowClient()

latest_model_version = client.get_latest_versions_by_alias(model_name, alias="Baseline").version
model_version_uri = f"models:/{model_name}/{latest_model_version}" # should be version 1

predict_func = mlflow.pyfunc.spark_udf(spark, model_uri=model_version_uri, result='double')

prediction_df = features_df.withColumn("prediction", predict_func(*features_df.columns))
```