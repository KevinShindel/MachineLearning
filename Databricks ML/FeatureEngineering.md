
### Simple workflow for creating Features in Databricks ML

```python
from databricks.feature_engineering import FeatureEngineeringClient
catalog_name = 'my_catalog'
schema = 'my_database'
features = ['gender', 'salary', 'tenure', 'TotalCharges']
primary_key = 'customerID'

telco_df = spark.table(f'{catalog_name}.{schema}.telco_data')

# prepare feature set
features_df = telco_df.select(primary_key, *features)

# feature table definition
fe = FeatureEngineeringClient()
feature_table_name = f'{catalog_name}.{schema}.telco_features'

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

### Use FeatureEngineeringClient to run inference

```python
from databricks.feature_engineering import FeatureEngineeringClient

model_name = 'TelcoChurnModel'
champion_model_uri = f'models:/{model_name}@champion'
primary_key = 'customerID'

fe = FeatureEngineeringClient()
lookup_df = test_df.select(primary_key)

predictions_df = fe.score_batch(
    model_uri=champion_model_uri,
    input_df=lookup_df,
    result_type='string'
)

# before saving ito good to create optimized inference table with liquid clustering on relevant columns ( ex: customer_id, tenure )
predictions_df.write.mode('append').option("mergeSchema", "true").saveAsTable(f'{catalog_name}.{database_name}.batch_inference_table')
```


### Feature Function

- Feaure Function that uses a Python UDF to create on-demand features.

```sql
CREATE OR REPLACE FUNCTION mounthly_charges_avg(total_charges DOUBLE, tenure DOUBLE)
RETURNS DOUBLE
LANGUAGE PYTHON
AS $$
def monthly_charges_avg(total_charges, tenure):
    if tenure == 0:
        return 0.0
    return total_charges / tenure
$$;
```

#### Define combined features

```python
from databricks.feature_engineering import FeatureLookup, FeatureFunction, FeatureEngineeringClient

fe = FeatureEngineeringClient()
catalog_name = 'my_catalog'
schema = 'my_database'
features_for_online_name = f'{catalog_name}.{schema}.features_for_online'
primary_key = 'customerID'
feature_table_name = f'{catalog_name}.{schema}.telco_features'

features = [
    FeatureLookup(
        table_name=feature_table_name,
        lookup_key=primary_key
    ),
    FeatureFunction(
        udf_name="mounthly_charges_avg",
        name='monthly_charges_avg',
        input_bindings={
            'total_charges': f'{feature_table_name}.TotalCharges',
            'tenure': f'{feature_table_name}.tenure'
        }
    )
]

# todo: find out last of code
```

- [Python API](https://docs.databricks.com/aws/en/machine-learning/feature-store/python-api)
- [Databricks FeatureEngineeringClient](https://api-docs.databricks.com/python/feature-engineering/latest/feature_engineering.client.html)
- [Databricks Feature Lookup](https://api-docs.databricks.com/python/feature-engineering/latest/ml_features.feature_lookup.html)