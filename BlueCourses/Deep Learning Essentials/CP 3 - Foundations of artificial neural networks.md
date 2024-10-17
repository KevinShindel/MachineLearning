## The perceptron
- Very loosely based on how the human brain works:
- - A collection of artificial neurons that are connected, allowing to send messages to each other.
- - We can start by modelling a small piece of this as a mathematical construct: the perceptron.
- A perceptron works over several numerical inputs: the input neurons or input units
- Each neuron has an output: its activation
- - For input neurons, activation is simply given by input data instance
- Every input unit's output is multiplied with a weight and summed by the perceptron unit
- Most neural networks use a linear combination over the inputs, like the weighted sum just seen 
- - Though other approaches exist as well, as we will see later on
- The difference between combination/transfer function and activation function is often not explicitly stated 
- - Most references simply talk about the activation function
- Both operations can be described as separate layers
- Very often, an additional weight is added as a bias
- - Basically: an additional input unit with its output fixed to 1 and with own weight
- - Comparable with intercept in a statistical regression model

| Concept              | Description                                                             |
|----------------------|-------------------------------------------------------------------------|
| Inputs               | The input features                                                      |
| Weights              | Parameters that are learned during training                             |
| Combination function | Or: transfer func, i.e. weighted sum of inputs                          |
| Net input            | Result of combination function                                          |
| Activation function  | Often used to describe the combination function and activation together |
| Bias                 | Or: threshold                                                           |
| Output               | Or: activation, the prediction                                          |

## Concept


## Activation and transfer functions

## Bias

## Training a perceptron
- For simple stat. models (linear regression) closed-form analytical formulas to determine the optimum parameter estimates exist
- For nonlinear models, such as neural networks, the parameter estimates must be determined numerically, using an iterative algorithm
- Feed-forward: push one instance through the network given current weight values
- Determine output y and compare to true output to determine error (cost, loss)
- - E.g. error = (expected - predicted)^2 = 0.25
- - Use this to update the weights
- Errors must be closer and closer to zero

## A simple iterative approach
- Each training instance is shown to the model one at a time
- - The model makes a prediction
- - The error is calculated
- - The model is updated in order to reduce the error for the next prediction
- For out simple perceptron, the weights can be updated using the following formula:
  - w = w + learning_rate * (expected - predicted) * x
- For the sake of notation, let:
- - o = w1 + w2 + b
- Gradient descent: the process of updating the weights in order to minimize the error
- - w = w + learning_rate * (expected - predicted) * x

## A simple perceptron in Python
```python
import math

# First, we define our data set. X defines a set of two-dimensional data points, and y defines our vector of outcomes (targets).
# Note: typically, we'd want to normalize, standardize, or minmax scale our instances first to make the network train better.
# Given that our samples aren't too extreme here, we can safely skip this step.

X = [[0, 1], [1, 0], [2, 2], [3, 4], [4, 2], [5, 2], [4, 1], [5, 0]]
y = [0, 0, 0, 0, 1, 1, 1, 1]

# Next up, we define our activation function, and can immediately define a function to predict an outcome given an instance.
# Since we use a sigmoid activation function, the output of the perceptron will be bounded between 0 and 1 and can be directly interpreted as a probability.

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def predict(instance, weights):
  # We need a weight for each input, plus a bias weight
  assert len(weights) == len(instance) + 1
  # Assume that the first weight given is our bias
  output = weights[0]
  for i in range(len(weights)-1):
    output += weights[i+1] * instance[i]
  return sigmoid(output)

# We can now already see what happens if we let an untrained perceptron make some predictions, e.g. by setting all the weights to 0.

# Setting these initial values for the weights is typically called "initialization", and is an important topic on its own we'll discuss later on.
# For now, we'll just set them to 0.
weights = [0, 0, 0]

for i in range(len(X)):
  prediction = predict(X[i], weights)
  print(X[i], y[i], '->', prediction)

def train(instance, weights, y_true, l_rate):
  prediction = predict(instance, weights)
  abserror   = y_true - prediction
  weights[0] = weights[0] + l_rate * 2 * abserror * prediction * (1-prediction)
  for i in range(len(weights)-1):
      weights[i+1] = weights[i+1] + l_rate * 2 * abserror * prediction * (1-prediction) * instance[i]
  return weights

# ext, we can set our learning rate, and train for one pass over our instances.

l_rate = 0.01

for i in range(len(X)):
  weights = train(X[i], weights, y[i], l_rate)

print(weights)

for i in range(len(X)):
  prediction = predict(X[i], weights)
  print(X[i], y[i], '->', prediction)

# It looks like nothing has happened. So let's try this for a couple more "epochs" (passes over the training data) and see what happens:

l_rate = 0.01
epochs = 2000

for n_epoch in range(epochs):
  for i in range(len(X)):
    weights = train(X[i], weights, y[i], l_rate)

print(weights)

for i in range(len(X)):
  prediction = predict(X[i], weights)
  print(X[i], y[i], '->', prediction)
```

## Gradient descent
- Error is function of the weights given a piece of training data
- - Minimize error using gradient of the error (loss) function
- Gradient descent - is a process of minimizing a function by following gradients of the cost functions
- - The involves the first partial derivatives of the lost function so that at given point, the gradient gives the direction of the steepest ascent
- One iteration: one instance fed-forward , weights are updated.
- One epoch: one full pass over all instances in the training set
- To properly train our perceptron, need to perform multiple passes over the training dset: multiple epochs
- - Typically, the training set is reshuffled after every epoch

## The XOR problem
- Assume the input data set given by the table 
- - No matter how long we train, the perceptron does a poor job
- - A perceptron cannot learn non-linear problems such as the XOR problem

## Multilayer perceptron (MLP) networks
- What happens if we would build a network with more then one perceptron?
- This is the idea of a multilayer perceptron (MLP)
- A Multi-Layer Perceptron Network:
- - All neurons in one layer are fully connected to those in the next layer
- - Three layers is typical (input, hidden, output) but more layers are possible (deep learning)
- - The questions is how to train this network: how to update many weights at once

## Backpropagation
- We can't use the same approach as we did for a single perceptron
- - We don't know what the 'true outcome' should be for the lowe layers
- - The issue took quite some time to solve
- - Eventually, a method called backpropagation was developed
- - Basically the chain rule applied on this partial derivative applied in step-by-step manner.
- Feed forward is till easy: we just move from left to right.
- We can also still easily compare th predicted output with the true which gives us an error value.
- The idea of backpropagation is to 'back propagate' the error through the network (stepwise use of chain rule and partial derivatives)
- Whe weights can be updated according to the gradient of the error function.

## Automatic differentiation

## Automatic differentiation in TensorFlow

## Summary so far

## ☞ Handwritten digits recognition with an MLP

## Further aspects

## Activation functions

## ReLU

## Initialization

## ☞ The importance of initialization

## Loss functions

## Stochastic gradient descent

## Backpropagation alternatives

## Optimizers

## Learning rate

## Preventing overfitting

## Hyperparameter optimization
