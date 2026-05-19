
![DAB 1](../assets/dab_1.png)
![DAB 2](../assets/dab_2.png)
![DAB 3](../assets/dab_3.png)
![DAB 4](../assets/dab_4.png)


### Example of Batch Inference with DAB

```yaml

resource "databricks_job" "dab_batch_inference" {
  name = "DAB Batch Inference Job"
  new_cluster {
    num_workers = 2
    spark_version = "13.2.x-cpu-ml-scala2.12"
    node_type_id = "Standard_DS3_v2"
    single_user_name = data.databricks_current_user.service_principal.user_name
    data_security_mode = "SINGLE_USER"
  }
  
  notebook_task {
    notebook_path = "notebooks/dab_batch_inference"
    base_parameters = {
      model_name = "my_model"
      input_data_path = "dbfs:/input/data.csv"
      output_data_path = "dbfs:/output/predictions.csv"
      env = local.env
    }

    }

    git_source {
        url = var.git_repo_url
        provider = "azureDevOpsServices"
        branch = "release"
}
  schedule {
    quartz_cron_expression = "0 0 11 * * ?" # Every day at 11:00 AM
    timezone_id = "UTC"
  }
}
```


#### DAB example for train dataset configuration

```yaml
train:
    data_dir_path: &dir tabe_path
    feature_selection_config: feature_selection_config.yml
    model:
        split_config: [0.8, 0.2]
        hyperparams:
            hyperparam_1: 10
            hyperparam_2: 0.01
            
```