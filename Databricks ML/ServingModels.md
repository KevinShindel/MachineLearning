## Endpoint API

Deploy models as APIs within CI\CD system

Create by:
- Rest API
- Python SDK
- GO SDK
- CLI
- Terraform
- Databricks UI

Integrate with:
- CI\CD system
- Terraform


### REST API Endpoint Creation

```python
import requests
import json

def create_inference_endpoin(token: str, workspace_url: str, endpoint_name: str, models: list) -> dict:
    url = f"{workspace_url}/api/2.0/serving-endpoints"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    payload = {
        "name": endpoint_name,
        "served_models": models
    }
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    if response.status_code == 200:
        print("Inference endpoint created successfully.")
    else:
        print(f"Failed to create inference endpoint: {response.text}")  
    return response.json()

name = 'feed-ads'
models = [{
    "model_id": "12345",
    "model_version": "1",
    'workload_size': 'small',
    'scale_to_zero': True
}]

token = "your_access_token"
workspace_url = "https://your-databricks-workspace-url"

endpoint_response = create_inference_endpoin(token, workspace_url, name, models)
```

### Python SDK Endpoint Creation

```python
from mlflow import MlflowClient
import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointTag, EndpointCoreConfigInput

# point to UC model registry
mlflow.set_tracking_uri("databricks-uc")

catalog = "my_catalog"
schema = "my_schema"
model = "my_model"

# init MLflow client
mlflow_client = MlflowClient()
# init workspace client
w = WorkspaceClient()

# define model name
model_name = f"{catalog}.{schema}.{model}"
# parse model name from UC namespace
served_model_name = model_name.split(".")[-1]
# define the endpoint name
endpoint_name = f"{served_model_name}-endpoint"

# get version of our model registered in UC
model_version_champion = mlflow_client.get_model_version(name=model_name, alias='champion', version="latest").version
model_version_challenger = mlflow_client.get_model_version(name=model_name, alias='challenger', version="latest").version

# define endpoint configuration (50% / 50% traffic split between champion and challenger)
endpoint_config_dict = {
    "name": endpoint_name,
    "served_models": [
        {
            "model_name": model_name,
            "model_version": model_version_champion,
            "workload_size": EndpointCoreConfigInput.WorkloadSize.SMALL,
            "scale_to_zero_enabled": True,
        },
        {
            "model_name": model_name,
            "model_version": model_version_challenger,
            "workload_size": EndpointCoreConfigInput.WorkloadSize.SMALL,
            "scale_to_zero_enabled": True,
        }
    ],
    "traffic_config": {
        "routes": [
            {
                "served_model_name": f"{model_name}_{model_version_champion}",
                "traffic_percentage": 50
            },
            {
                "served_model_name": f"{model_name}_{model_version_challenger}",
                "traffic_percentage": 50
            }
        ]
    },
    "auto_capture_config": {
        "catalog_name": catalog,
        "schema_name": schema,
        "table_name_prefix": "serve_models_auto_capture",
    }
}

# create endpoint configuration object
endpoint_config = EndpointCoreConfigInput.from_dict(endpoint_config_dict)

# Serve Endpoint Creation
try:
    endpoint = w.serving.create_and_wait(
        name=endpoint_name,
        config=endpoint_config,
    tags=[
        EndpointTag.from_dict({
            "key": "owner",
            "value": "data_science_team"
        })]
    )
    print(f"Endpoint '{endpoint_name}' created successfully.")
except Exception as e:
    if "already_exists" in str(e):
        print(f"Endpoint '{endpoint_name}' already exists.")
    else:
        raise(e)
```


### Endpoint Verification

```python
from databricks.sdk import WorkspaceClient
import pandas as pd

X_train = pd.read_csv("path_to_your_training_data.csv")

w = WorkspaceClient()
served_model_name = "my_model"
endpoint_name = f"{served_model_name}-endpoint"
endpoint = w.serving.get_serving_endpoint_not_updating(endpoint_name)
assert endpoint.config_update.value == "NOT_UPDATING" and endpoint.state.ready.value == "READY", "Endpoint is not ready for serving"

# Query the endpoint for inference
batch_size = 100
sample_size = 1000
dataframe_records = X_train.sample(sample_size).to_dict(orient="records")
num_batches = (len(dataframe_records) + batch_size - 1) // batch_size

all_predictions = []
all_models = []

for i in range(num_batches):
    batch_records = dataframe_records[i*batch_size:(i+1)*batch_size]
    response = w.serving_ednpoins.query(
        name=endpoint_name,
        dataframe_records=batch_records
    )
    predictions = response.predictions
    model_names = response.served_model_names
    all_predictions.extend(predictions)
    all_models.extend([
        [response.served_model_name] * len(predictions) # assuming all predictions in the batch are from the same model, adjust if needed
    ])
    
# combine predictions and model names into a single dataframe for analysis
results_df = pd.DataFrame({
    "prediction": all_predictions,
    "served_model_name": all_models
})

# count occurrences of each model in the predictions
count_results = results_df['prediction'].value_counts().reset_index()
count_results.columns = ['prediction', 'count']

display(count_results)

# aggregate count of predictions per model
model_count_results = results_df.groupby(['served_model_name', 'prediction']).size().reset_index(name='count')
# display the count of predictions per model
display(model_count_results)
```