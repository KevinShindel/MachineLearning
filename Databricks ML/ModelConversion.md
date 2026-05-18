### ONNX Conversion

- Convert and log model ONNX 
```python
import mlflow
import onnxmltools

onnx_model = onnxmltools.convert_keras(model)
mlflow.onnx.log_model(onnx_model, "onnx-model")
```

- Read and score model by ONNX Runtime
```python
import onnxruntime

session = onnxruntime.InferenceSession(model.SerializeToString())
input_name = session.get_inputs()[0].name
predictions = session.run(None, {input_name:data_np.astype(np.float32)})[0]
```