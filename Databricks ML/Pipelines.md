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