import mlflow
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import numpy as np


EXPERIMENT_NAME = "random-forest-best-models"
RF_PARAMS = ['max_depth', 'n_estimators', 'min_samples_split', 'min_samples_leaf', 'random_state']

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.sklearn.autolog()


def train_and_log_model():
    # lets use Iris Data
    data = load_iris()
    X = data.data
    y = data.target
    print('[+] Data loaded')

    # Split train data 60%
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=0.6)

    # Split train and test data to 20% and 20% of original dataset
    X_test, X_validation, y_test, y_validation = train_test_split(X_temp, y_temp, train_size=0.5)

    # Define the parameter space
    param_distributions = {
        'n_estimators': [int(x) for x in np.linspace(start=100, stop=1000, num=10)],
        'max_depth': [int(x) for x in np.linspace(10, 100, num=10)] + [None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'random_state': [42]
    }
    print('[+] Training model')
    base_model = RandomForestRegressor()

    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=5,
        cv=5,
        verbose=2,
        random_state=42,
        n_jobs=-1
    )

    with mlflow.start_run(run_name="random_search_optimization"):
        # Fit the random search
        random_search.fit(X_train, y_train)

        # Log the best parameters
        mlflow.log_params(random_search.best_params_)

        # Get predictions and log metrics for best model
        val_predictions = random_search.predict(X_validation)
        test_predictions = random_search.predict(X_test)

        # Log validation and test metrics
        val_rmse = mean_squared_error(y_validation, val_predictions, squared=False)
        test_rmse = mean_squared_error(y_test, test_predictions, squared=False)

        mlflow.log_metric("validation_rmse", val_rmse)
        mlflow.log_metric("test_rmse", test_rmse)

        # Log best score from CV
        mlflow.log_metric("best_cv_score", random_search.best_score_)

        # Log the best model
        mlflow.sklearn.log_model(random_search.best_estimator_, "best_model")

        print('[+] Model trained')
        return random_search.best_estimator_, random_search.best_params_


if __name__ == '__main__':
    model, params = train_and_log_model()
