### Monitoring Features
- Data Key Metricks: avg, std, null cnts, disticncsts etc.
- Time Granularities: over window every day, 5 minutes, over N weeks
- Data Slices: state, product_class, cart_total > 1000
- Tables, Views, ML models: quality / drift monitoring


### Monitoring available by UI and SDK
- [Official documentation page](https://docs.databricks.com/sap/en/create-lakehouse-monitor-api)

```python
# import databricks.lakehouse_monitoring as lm

# set-up monitoring params

lm.create_monitor(
    table_name='my_UC_table_name',
    
)

lm.run_refresh('my_table')
```

### Monitoring point
- Ongoing evaluation
- Logging of key metricks
- Diagnose issues
- Data for monitoring:
- - Input Data
- - Data in feature stores / vector DB
- - Human feedbacks
- - Mid-training checkpoint analysis
- - Model Evaluation metricks


### Table metrics
- create_or_update_monitor -set-up monitoring for a table
- refresh_metrics - at the end of the pipeline, refresh the monitoring metrics for the table