## Pipeline advantages vs limitations

- Advantages:
- - Lower latency ( near real-time )
- - Generate predictions and act sooner
- - Event-driven ( react to changes in data immediately )
- - Better resource utilization ( only run when needed )

- Limitations:
- - More costly than batch solution
- - More complicated to develop and maintain
- - Less throughput ( process smaller batches of data )
- - Resource intensiveness ( comp.power by processing data in real-time )

### Lakeflow Declarative Pipelines

- Accelerate ETL development
- - Declare SQL or Python pipelines
- Auto mange infrastructure ( scaling, recovery, perf. optimization)

```sql
CREATE STREAMING TABLE raw_data
       AS SELECT * 
       FROM cloud_files ('/raw_data', 'json')
```

```sql
CREATE MATERIALIZED VIEW clean_data
       AS SELECT ... 
       FROM raw_data
```

### Delta Live Tables (DLT)

- Defined by a SQL query
- Created and keep up-tp-date by a pipeline

```sql
CREATE OR REFRESH LIVE TABLE clean_data
AS SELECT sum(profit)
FROM prod.sales
```


### Data Quality and Validation
- DEfine data quality and integrity controls within pipeline with data expectations
- Address data quality errors with flexible policies: fail, drop, alert, quarantine, etc.
- All data pipeline runs and quality metrics are captured, tracked and reported in a central dashboard for monitoring and troubleshooting

```sql
CREATE OR REFRESH LIVE TABLE fire_account_bronze AS 
(CONSTRAINT valid_account_open_dt EXPECT (account_open_dt IS NOT NULL AND (account_close_dt > account_open_dt)) ON VIOLATION DROP ROW)
COMMENT 'Bronze table for fire account data'
AS SELECT * FROM cloud_files('/fire_account_data', 'json')
```

### Continuous or Scheduled Data Ingestion
- Incrementally and efficiently process new data files as they arrive in cloud storage with auto-triggered pipelines
- Auto infer schema of incoming files of superimpose what you know with Schema Hints
- Automatic Schema Evolution ( JSON, CSV, AVRO, Parquet)

```sql
CREATE OR REFRESH LIVE TABLE sales_orders_raw 
COMMENT 'Raw sales orders data from cloud storage, ingested from /path/to/sales_orders with JSON format'
TABLEPROPERTIES ('quality' = 'bronze')
AS 
SELECT * FROM cloud_files('/path/to/sales_orders', -- path
                          'json', -- format
                          map('cloudFiles.inferColumnTypes', 'true')) -- Infer schema
```


### DLT workflow

- Create live table in notebook but do not run it
- Create pipeline in Workflows and add notebook with live table
- Click start to run pipeline and create live table

### Inference Pipeline

1. Pipeline Config
2. Inference Config
3. DLT Inference Pipeline

```python
from databricks import dlt
from pyspark.sql import functions as F
import mlflow

# pipeline config
bronze_dataset_path = spark.conf.get("pipeline_config.bronze_dataset_path")
model_name = spark.conf.get("pipeline_config.model_name")

# inference config
mlflow.set_registry_uri('databricks-uc')
model_uri = f'models:/{model_name}@DLT'
model_udf = mlflow.pyfunc.spark_udf(spark, model_uri, result_type='string')
primary_key = 'customer_id'
features = ['feature1', 'feature2', 'feature3']

# DLT inference pipeline

@dlt.table(
    name='raw_inputs',
    comment='Raw input data for inference',
    table_properties={'quality': 'bronze'}  
)
def raw_inputs():
    return (spark.read.format('delta')
            .load(bronze_dataset_path))

@dlt.table(
    name='feature_inputs',
    comment='Feature engineered inputs for inference',
    table_properties={'quality': 'silver'}
)
def feature_inputs():
    return (dlt.read('raw_inputs')
            .select(primary_key, *features)
            .withColumn('feature1_scaled', F.col('feature1') / 100)
            .withColumn('feature2_scaled', F.col('feature2') / 50)
            .withColumn('feature3_scaled', F.col('feature3') / 10)
            .na.drop(how='any'))

@dlt.table(
    name='predictions',
    comment='Model predictions',
    table_properties={'quality': 'gold'}
)
def predictions():
    return (dlt.read('feature_inputs')
            .withColumn('prediction', model_udf(F.struct(features))))
```