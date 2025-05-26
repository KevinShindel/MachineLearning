import click
from mlflow import MlflowClient
from mlflow.entities import ViewType
import mlflow

EXPERIMENT_NAME = "random-forest-best-models"
mlflow.set_tracking_uri("http://127.0.0.1:5000")


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
@click.option(
    "--top_n",
    default=5,
    type=int,
    help="Number of top models that need to be evaluated to decide which one to promote"
)
def run_register_model(data_path: str, top_n: int):

    client = MlflowClient()

    # Retrieve the top_n model runs and log the models
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    best_run = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.validation_rmse ASC"]
    )[0]

    if best_run is not None:
        # Select the model with the lowest test RMSE
        run_id = best_run.info.run_id
        best_test_rmse = best_run.data.metrics['validation_rmse']

        model_uri = f"runs:/{run_id}/model"
        mlflow.register_model(
            model_uri=model_uri,
            name="best-iris-forest-regressor-model"
        )
        print(f"Registered model with run_id: {run_id}")
        print(f"Best test RMSE: {best_test_rmse}")


if __name__ == '__main__':
    run_register_model()
