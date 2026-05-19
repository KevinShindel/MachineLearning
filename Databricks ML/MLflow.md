from pyspark.pandas import mlflow

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


### Fit and Register a Custom Model

1. Declare wrapper class for custom model
2. train base model
3. Instantiate custom model using trained base model & log registry

```python
import pandas as pd
from mlflow.pyfunc import PythonModel

class CustomProbaModel(PythonModel):
    def __init__(self, base_model, labels: list):
        self.base_model = base_model
        self.labels = labels
        
    def predict(self, context, model_input):
        pred_proba = self.base_model.predict_proba(model_input)[:, 1]
        predictions = self.base_model.predict(model_input)
        
        # Organize multiple outputs into a dataframe
        result = pd.DataFrame(pred_proba, columns=[f'proba_{label}' for label in self.labels])
        result['prediction'] = predictions
        
        return result
```


```python
# train base model
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
base_model = DecisionTreeClassifier(max_depth=3, random_state=42)
base_model.fit(X_train, y_train)

````

```python
# log custom model to registry
from mlflow.models import infer_signature


wrapped_model = CustomProbaModel(base_model=base_model, labels=iris.target_names.tolist())

# Define the input and output schema for the model
input_example = X_train.sample(1)
output_example = wrapped_model.predict(None, input_example)
signature = infer_signature(input_example, output_example)

# initiate MLflow run and log the custom model
with mlflow.start_run(run_name="Custom-Model-Registry-Demo") as mlflow_run:
    mlflow.pyfunc.log_model(artifact_path="custom_proba_model",
                            python_model=wrapped_model,
                            input_example=input_example,
                            registered_model_name=f"{catalog_name}.{schema_name}.custom_proba_model",
                            signature=signature)
```


#### Tests Wrapped model

```python
# load model from the run 
run_id = mlflow.last_active_run().info.run_id
logged_model_path = f'runs:/{run_id}/custom_proba_model'
loaded_model = mlflow.pyfunc.load_model(model_uri=logged_model_path)

# use the loaded model for inference
y_test = loaded_model.predict(X_test)
print(y_test.head())
```


### Create endpoint for custom model

```python
from databricks.sdk.service.serving import EndpointCoreConfigInput, EndpointTag
from databricks.sdk import WorkspaceClient

client = WorkspaceClient()
catalog_name = "my_catalog"
schema_name = "my_schema"
endpoint_name = "custom-proba-endpoint"
model_version = '1'
endpoint_config_dict = {
"served_models": [
    {
        "model_name": f"{catalog_name}.{schema_name}.custom_proba_model",
        "model_version": model_version,
        "scale_to_zero_enabled": True,
        "workload_size": "Small"
    }
],
}

endpoint_config_obj = EndpointCoreConfigInput.from_dict(endpoint_config_dict)

try:
    w.serving_endpoints.create_and_wait(name=endpoint_name,
                                        config=endpoint_config_obj,
                                        tags=[EndpointTag.from_dict({
                                            "key": "use_case",
                                            "value": "custom_model_demo" })]
                                        )
except Exception as e:
    if "already exists" in str(e):
        print(f"Endpoint {endpoint_name} already exists. Skipping creation.")
    else:
        raise e
```

### Query endpoint for inference

```python
# hardcore test input
dataframe_records = [
{"sepal length (cm)": 5.1, "sepal width (cm)": 3.5, "petal length (cm)": 1.4, "petal width (cm)": 0.2},
{"sepal length (cm)": 6.2, "sepal width (cm)": 3.4, "petal length (cm)": 5.4, "petal width (cm)": 2.3}
]

print('inference results: ')
query_response = w.serving_endpoints.query(name=endpoint_name,
                                           dataframe_records={"columns": dataframe_records})
print(query_response)
```