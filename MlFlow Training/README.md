### Using MLFlow

1. Run server ```mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000```
2. Load and serve model ```python taxi_prediction.py```
3. Make request ```curl -X POST -H "Content-Type: application/json" -d "{\"inputs\": [2.0, 30.5, 500]}" http://127.0.0.1:5001/invocations```