import json
import logging
import os
import signal
import subprocess
import sys
import time

import joblib
import mlflow
import requests
from mlflow.models import ModelSignature

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TaxiDurationPredictor(mlflow.pyfunc.PythonModel):
    """Model for predicting taxi ride duration."""

    def __init__(self, model_path: str = "model.pkl"):
        self.model_path = model_path
        self.model = None

    def load_context(self, context):
        """Load model when MLflow loads the model."""
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
            else:
                raise FileNotFoundError(f"Model file not found at {self.model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def predict(self, context, model_input):
        """Make predictions with the model."""
        if self.model is None:
            raise ValueError("Model not loaded. Please ensure load_context() is called.")

        try:
            return self.model.predict(model_input)
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise

def create_model_signature():
    """Create model signature for input/output specifications."""
    model_input = json.dumps([{'name': 'total_amount', 'type': 'float'}])
    model_output = json.dumps([{'name': 'duration', 'type': 'float'}])
    return ModelSignature.from_dict({'inputs': model_input, 'outputs': model_output})

def wait_for_server(url: str, timeout: int = 60):
    """Wait for server to be ready."""
    start_time = time.time()
    while True:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                logger.info("Server is ready!")
                return True
        except requests.exceptions.ConnectionError:
            if time.time() - start_time > timeout:
                raise TimeoutError("Server failed to start within timeout period")
            time.sleep(1)

class ModelServer:
    def __init__(self, run_id: str, port: int = 5001):
        self.run_id = run_id
        self.port = port
        self.process = None

    def start(self):
        """Start the MLflow model server."""
        command = [
            "mlflow", "models", "serve",
            "-m", f"runs:/{self.run_id}/model",
            "--port", str(self.port),
            "--no-conda"
        ]

        self.process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # Wait for server to start
        try:
            wait_for_server(f"http://localhost:{self.port}/health")
            logger.info(f"Model server started successfully on port {self.port}")
        except TimeoutError:
            self.stop()
            raise

    def stop(self):
        """Stop the MLflow model server."""
        if self.process:
            self.process.terminate()
            self.process.wait()
            logger.info("Model server stopped")

def log_model_to_mlflow(model_path: str, tracking_uri: str = "http://127.0.0.1:5000") -> str:
    """Log model to MLflow with proper configuration."""
    try:
        mlflow.set_tracking_uri(tracking_uri)

        conda_env = {
            'channels': ['defaults', 'conda-forge'],
            'dependencies': [
                'python=3.11',
                'scikit-learn',
                'pandas',
                'joblib'
            ],
            'name': 'taxi_duration_env'
        }

        with mlflow.start_run(run_name="duration-predictor") as run:
            model = TaxiDurationPredictor(model_path)

            mlflow.pyfunc.log_model(
                artifact_path='model',
                python_model=model,
                conda_env=conda_env,
                registered_model_name="duration-predictor",
                signature=create_model_signature()
            )

            logger.info(f"Model logged with run_id: {run.info.run_id}")
            return run.info.run_id

    except Exception as e:
        logger.error(f"Error logging model to MLflow: {str(e)}")
        raise

def handle_shutdown(server):
    """Handle graceful shutdown on SIGINT/SIGTERM."""
    def signal_handler(signum, frame):
        logger.info("Shutdown signal received. Stopping server...")
        server.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

def main():
    MODEL_PATH = "model.pkl"  # Update this path as needed
    TRACKING_URI = "http://127.0.0.1:5000"
    SERVE_PORT = 5001

    try:
        # Log model to MLflow
        run_id = log_model_to_mlflow(MODEL_PATH, TRACKING_URI)
        logger.info(f"Model successfully logged to MLflow with run_id: {run_id}")

        # Start model server
        server = ModelServer(run_id, SERVE_PORT)
        server.start()

        # Setup graceful shutdown
        handle_shutdown(server)

        # Keep the script running
        logger.info(f"Model is being served at http://localhost:{SERVE_PORT}/invocations")
        logger.info("Press Ctrl+C to stop the server")

        # Example of how to make a prediction
        logger.info("\nExample of how to make a prediction using curl:")
        logger.info("""curl -X POST -H "Content-Type: application/json" -d "{\"inputs\": [20.0]}" http://127.0.0.1:""" + str(SERVE_PORT) + """/invocations""")

        # Keep the main thread alive
        while True:
            time.sleep(1)

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()