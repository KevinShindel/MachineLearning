### Predictive Optimization

- auto identification tables that would benefit from maintenance operations and runs them for the user
- Run: OPTIMIZE ( improve query performance by optimize file size), VACUUM ( truncate history )

```sql
ALTER CATALOG my_catalog_name ENABLE PREDICTIVE OPTIMIZATION;
```

```sql
ALTER SCHEMA my_catalog_name ENABLE PREDICTIVE OPTIMIZATION;
```

### Liquid Clustering

- Replace partitioning and ZORDER

```sql
CREATE TABLE table1 (col0 int, col1 string) 
USING DELTA 
CLUSTER BY (col0);
```

### Manual table optimization

- improve speed of read by colaescing small files into large one
- better use predictive optimization 

```sql
OPTIMIZE table0;
```


### Inference table optimization
- Enable liquid clustering on the inference table to improve query performance and reduce latency for batch inference workloads. By clustering the inference table based on relevant columns, such as customer_id and tenure, we can optimize data retrieval and enhance the efficiency of batch inference processes.

```sql
CREATE OR REPLACE TABLE batch_inference_table
       customer_id STRING,
       churn STRING,
       senior_citizen DOUBLE,
       tenure DOUBLE,
       monthly_charges DOUBLE,
       total_charges DOUBLE
       preiction STRING)
       CLUSTER BY (customer_id, tenure)
```

### Manually Optimize table

```sql
ANALYZE TABLE batch_inference_table COMPUTE STATISTICS FOR ALL COLUMNS;
OPTIMIZE batch_inference_table;
```