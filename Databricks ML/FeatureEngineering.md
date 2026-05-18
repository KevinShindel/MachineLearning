from colorlog import warning

### Simple workflow for creating Features in Databricks ML

```python
from databricks.feature_engineering import FeatureEngineeringClient

# prepare feature set
features_df = telco_df.select(primary_key, *features)

# feature table definition
fe = FeatureEngineeringClient()
feature_table_name = f'{catalog_name}.{database_name}.telco_features'

# santity check 
try:
    fe.drop_table(name=feature_table_name)
except Exception as e:
    print(f"Feature table {feature_table_name} does not exist. Creating a new one.")


# create feature table
fe.create_table(
    name=feature_table_name,
    features=features_df,
    primary_key=primary_kay,
    description="Feature table for Telco customer churn prediction",
    tags={"source": "telco_data", "created_by": "databricks_ml_team"}
)   
```

### Create training set based on feature lookup

```python
from databricks.feature_engineering import FeatureLookup, FeatureEngineeringClient

fe = FeatureEngineeringClient()

primary_key = 'customerID'
label = 'Churn'

fl_handler = FeatureLookup(
table_name=feature_table_name,
lookup_key=primary_key
)
training_set_spec = fe.create_training_set(
    df=response_df,
    label=label,
    feature_lookups=[fl_handler],
    exclude_columns=[primary_key]
)

# load training dataframe based on defined feature lookups specs
training_df = training_set_spec.load_df()

```


### Training with MLflow

```python
import warnings
import mlflow
from sklearn.tree import DecisionTreeClassifier

client = mlflow.tracking.MlflowClient()

# convert data to pandas dataframe for training
X_train = training_df.drop(columns=[primary_key, label]).toPandas()
y_train = training_df.select(label).toPandas().values.ravel()

clf = DecisionTreeClassifier()

with mlflow.start_run(run_name="Telco Churn Prediction") as mlflow_run:
    # Enable auto-logging of input samples, metrics, parameters, and models
    mlflow.sklearn.autolog(
        log_input_examples=True,
        log_models=True,
        log_post_training_metrics=True,
        silent=True
    )

    # Train the model
    clf.fit(X_train, y_train)

    # infer output schema
    try:
        output_schema = _infer_schema(Y_train)
    except Exception as e:
        warnings.warn(f'Could not infer output schema: {e}')
        output_schema = None
        
    # log using feature engineering client and push into registry
    fe.log_model(
        model=clf,
        artifact_path='decission_tree_model',
        flavor='mlflow.sklearn',
        training_set=training_set_spec,
        registered_model_name='TelcoChurnModel',
        output_schema=output_schema
    )
    
    # Set model alias ( aka Champion model )
    client = set_registered_model_alias(
        name='TelcoChurnModel',
        alias='Champion',
        version=mlflow_run.info.run_id
    )

```