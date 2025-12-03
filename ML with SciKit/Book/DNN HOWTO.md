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

# Optimizers Explanation

| Optimizer                        | Speed | Quality           | Description                                                                |
|----------------------------------|-------|-------------------|----------------------------------------------------------------------------|
| SGD                              | *     | ***               | Stochastic Gradient Descent, good for large datasets with low overfitting. |
| SGD(lr=0.001, momentum=0.9)      | **    | ***               | SGD with momentum, faster convergence while maintaining quality.           |
| SGD(momentum=0.9, nesterov=True) | **    | ***               | Nesterov Accelerated Gradient, improves convergence speed.                 |
| AdaGrad                          | ***   | *(early stopping) | Adaptive Gradient Algorithm, good for sparse data.                         |
| RMSprop                          | ***   | ** or ***         | Root Mean Square Propagation, effective for non-stationary objectives.     |
| Adam                             | ***   | ** or ***         | Adaptive Moment Estimation, combines benefits of AdaGrad and RMSprop.      |
| Nadam                            | ***   | ** or ***         | Nesterov-accelerated Adam, faster convergence with good quality.           |
| AdaMax                           | ***   | ** or ***         | Variant of Adam based on the infinity norm, stable for large datasets.     |

# Regularization Techniques

| Technique            | Description                                                                         | Examples                                                      |
|----------------------|-------------------------------------------------------------------------------------|---------------------------------------------------------------|
| Dropout              | Randomly drops neurons during training to prevent overfitting.                      | layers.Dropout(0.5)                                           |
| L1 Regularization    | Adds a penalty equal to the absolute value of the magnitude of coefficients.        | regularizers.L1(0.01)                                         | 
| L2 Regularization    | Adds a penalty equal to the square of the magnitude of coefficients.                | regularizers.L2(0.01)                                         |
| L1_L2 Regularization | Combines L1 and L2 penalties to leverage benefits of both methods.                  | regularizers.L1L2(l1=0.01, l2=0.01)                           |
| Batch Normalization  | Normalizes the inputs of each layer to stabilize learning and speed up convergence. | layers.BatchNormalization()                                   |
| Early Stopping       | Stops training when the validation performance starts to degrade.                   | keras.callbacks.EarlyStopping(monitor='val_loss', patience=5) |

# Example of DNN model with L2 regularization

```python

from functools import partial
from tensorflow.keras import layers, activations, regularizers, models

RegularizedDense = partial(layers.Dense,
                         activation=activations.relu,
                         kernel_initializer='he_normal',
                         kernel_regularizer=regularizers.L2(0.01))


model = models.Sequential([
    layers.InputLayer(shape=[28, 28]),
    layers.Flatten(),
    RegularizedDense(300),
    RegularizedDense(100),
    layers.Dense(10, activation=activations.softmax)
])
```

# Tuned Dropout Example

```python
from tensorflow.keras import layers, activations, models

class MCDropout(layers.Dropout):
    """
     Monte Carlo Dropout layer that is active during training and inference.
    """
    def call(self, inputs):
        return super().call(inputs, training=True)

model = models.Sequential([
    layers.InputLayer(shape=[28, 28]),
    layers.Flatten(),
    layers.Dense(300, activation=activations.relu),
    MCDropout(0.5),
    layers.Dense(100, activation=activations.relu),
    MCDropout(0.5),
    layers.Dense(10, activation=activations.softmax)
])
```

# Max-Norm regularization Example

```python
from tensorflow.keras import layers, activations, regularizers, models, initializers, constraints


layers.Dense(
    n_units=100, 
    activation=activations.relu,
    kernel_initializer=initializers.he_normal,
    kernel_constraint=constraints.max_norm(1.0) # max-norm constraint 
)
```

# Standard DNN configuration

| Hyperparameter         | Recommended Value  |
|------------------------|--------------------|
| Kernel Initializaztion | Xe initialization  |
| Activation Function    | ELU                |
| Normalization          | BatchNorm          |
| Optimizer              | RMSProp / Nadam    |
| Regularization         | EarlyStopping + L2 |
| LR Plot                | One cycle          |

# DNN with self-normalizing layers configuration

| Hyperparameter         | Recommended Value |
|------------------------|-------------------|
| Kernel Initializaztion | LeCun normal      |
| Activation Function    | SeLU              |
| Regularization         | AlphaDropout      |
| Optimizer              | RMSProp / Nadam   |
| LR Plot                | One cycle         |