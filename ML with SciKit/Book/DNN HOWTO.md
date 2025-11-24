## Simple DNN model for image classification using TensorFlow/Keras ( 10 classes )

```python
from tensorflow import keras
from tensorflow.keras import layers, losses, activations, models, optimizers, metrics

model = models.Sequential([                # sequential model
    layers.InputLayer(shape=[28, 28]),     # input layer for 28x28 images
    layers.Flatten(),                      # input layer flattening (converts 2D to 1D, x.reshape(-1, 1))
    layers.Dense(300, activation=activations.relu),  # hidden layer 1 (ReLU activation)
    layers.Dense(100, activation=activations.relu),  # hidden layer 2 (ReLU activation)
    layers.Dense(10, activation=activations.softmax) # output layer (10 classes, softmax activation)
])

# Compile the model

model.compile(loss=losses.sparse_categorical_crossentropy, # loss function
              optimizer=optimizers.SGD(),                        # optimizer
              metrics=[metrics.binary_accuracy])                   # metrics to monitor

history = model.fit(X_train,
                    y_train,
                    epochs=30,
                    validation_data=(X_valid, y_valid),
                    verbose=2)
```
# Recommendation table for Regression DNN models


| Params                   | Value                                        |
|--------------------------|----------------------------------------------|
| Input size               | 1 per feature                                |
| Hidden layers            | fom 1 to 5                                   |
| Neurons per layer        | from 10 to 100                               |
| Activation Hidden Layers | ReLU, SeLU                                   |
| Activation Output Layer  | Linear -> -Inf, +Inf, ReLU >=0, Softplus > 0 |
| Loss functions           | MSE, MAE, Huber ( combined )                 |
| Optimizers               | SGD, Adam, RMSprop                           |


# Recommendation table for Binary Classification DNN models

| Params                   | Value                                                                             |
|--------------------------|-----------------------------------------------------------------------------------|
| Input size               | 1 per feature                                                                     |
| Hidden layers            | fom 1 to 5                                                                        |
| Output Neurons           | 1                                                                                 |
| Neurons per layer        | from 10 to 100                                                                    |
| Activation Hidden Layers | ReLU, SeLU                                                                        |
| Activation Output Layer  | Sigmoid ( 0 to 1 )                                                                |
| Loss functions           | Binary Crossentropy                                                               |
| Optimizers               | SGD(large data, low overfitting), Adam (fast training), RMSprop (for noised data) | 

# Multi-Binary Classification DNN models

| Params                   | Value                   |
|--------------------------|-------------------------|
| Input size               | 1 per feature           |
| Hidden layers            | fom 1 to 5              |
| Output Neurons           | N ( number of classes ) |
| Neurons per layer        | from 10 to 100          |
| Activation Hidden Layers | ReLU, SeLU              |
| Activation Output Layer  | Sigmoid ( 0 to 1 )      |
| Loss functions           | Binary Crossentropy     |
| Optimizers               | SGD, Adam, RMSprop      |

# Recommendation table for Multi-Class Classification DNN models

| Params                   | Value                           |
|--------------------------|---------------------------------|
| Input size               | 1 per feature                   |
| Hidden layers            | fom 1 to 5                      |
| Output Neurons           | N ( number of classes )         |
| Neurons per layer        | from 10 to 100                  |
| Activation Hidden Layers | ReLU, SeLU                      |
| Activation Output Layer  | Softmax ( sum = 1 )             |
| Loss functions           | Sparse Categorical Crossentropy |
| Optimizers               | SGD, Adam, RMSprop              |

