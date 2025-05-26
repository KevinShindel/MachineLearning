from mlflow.tracking import MlflowClient
import mlflow
from datetime import datetime

mlflow.set_tracking_uri("http://127.0.0.1:5000")
MODEL_NAME = 'best-iris-forest-regressor-model'


def promote_model():
    client = MlflowClient()
    version = '1'  # specify your model version

    # Example 1: Set a status tag
    client.set_model_version_tag(
        name=MODEL_NAME,
        version=version,
        key="status",
        value="staging"
    )

    # Example 2: Add metadata tags
    client.set_model_version_tag(
        name=MODEL_NAME,
        version=version,
        key="promoted_by",
        value="data_science_team"
    )

    client.set_model_version_tag(
        name=MODEL_NAME,
        version=version,
        key="promotion_date",
        value=datetime.now().isoformat()  # Convert to string
    )

    # Example 3: Archive existing production models
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    for mv in versions:
        # Get model version details which includes tags
        model_version = client.get_model_version(MODEL_NAME, mv.version)
        if hasattr(model_version, 'tags') and model_version.tags.get('status') == "production":
            client.set_model_version_tag(
                name=MODEL_NAME,
                version=mv.version,
                key="status",
                value="archived"
            )

    # Set new version as production
    client.set_model_version_tag(
        name=MODEL_NAME,
        version=version,
        key="status",
        value="production"
    )


def get_model_status():
    client = MlflowClient()
    version = '1'

    # Get model version details which includes tags
    model_version = client.get_model_version(MODEL_NAME, version)

    print(f"Model {MODEL_NAME} version {version}:")
    if hasattr(model_version, 'tags'):
        print(f"Status: {model_version.tags.get('status', 'Not set')}")
        print(f"Promoted by: {model_version.tags.get('promoted_by', 'Not set')}")
    else:
        print("No tags found for this model version")


if __name__ == '__main__':
    promote_model()
    get_model_status()