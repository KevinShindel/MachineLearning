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
- Loss functions involving absolute errors or the ReLU activation function
- TensorFlow and many other deep learning frameworks utilize automatic differentiation
- - Not the same as symbolic or numeric differentiation
- - Inspection of computational graph which links operations over higher rank metrics together
- See notebook 'dle_fnd_automaticdifferentiation.ipynb'

## Summary so far
- Let's recap the basic ideas using: https://playground.tensorflow.org/
- The deep in deep learning isn't reference to any kind of deeper understanding achieved by the approach
- - It stands for this idea of successive layers of neurons
- Other appropriate names for the field could have been:
- - Layered representations learning
- - Hierarchical representations learning
- - Differential function learning
- Modern deep learning often involves tens or even hundreds of layers
- - Trained automatically from exposure to training data

## Handwritten digits recognition with an MLP
- Scope: recognize handwritten digits
- Step by using Keras:
- 1. Define the model
- 2. Compile the model
- 3. Fit the model
- 4. Evaluate the model
- 5. Make predictions
- See notebook 'dle_mlp_imagedigits.ipynb'

## Further aspects
- Activation functions: ReLU, sigmoid, tanh, softmax
- - Logistic (sigmoid) function: 1 / (1 + exp(-x)) -> Output between 0 and 1
- - Hyperbolic tangent (tanh) function: (exp(x) - exp(-x)) / (exp(x) + exp(-x)) -> Output between -1 and 1
- - Rectified linear unit (ReLU) function: max(0, x) -> Output between 0 and infinity
- - Softmax function: exp(x) / sum(exp(x)) -> Output between 0 and 1, summing to 1
- - Linear function: f(x) = x -> Output is the input
- - Radial basis function (RBF): exp(-x^2) -> Output between 0 and 1
- It was found that such activation functions limit training for deeper neural networks with more layers
- - Due to problems with "exploding" or "vanishing" gradients
- Two key was to solve this:
- - Rectified linear unit (ReLU) function: f(x) = max(0, x) -> Output between 0 and infinity
- - Better initialization of the weights

## ReLU
- ReLU reduces the likelihood of the vanishing gradient problem
- - The problem of vanishing gradients happens when the gradient of the loss function becomes very small
- - The opposite problem is called the exploding gradient problem
- - The constant gradient of ReLU results in faster training
- - Why not linear activation function? The derivative of the ReLU function is 0 for x < 0
- Variants such as noisy or leaky ReLU's also commonly used

## Initialization
- In our simple example, we have initialized the weights to 0 (that's is not ideal)
- Good starting values for the weights are essential for the training of a neural network
- - Prevent layer activation outputs from exploding or vanishing
- - If either occurs, loss gradients will either be too large or too small to flow backwards beneficially and the network will take longer to converge, if it is even able to do so
- An older approach is "preliminary training" or "pre-training" of the network
- - Use a random starting weights (uniform/gaussian) and train the network for a few epochs
- - Use the best of the final values as the new starting point and continue training with those

## The importance of initialization
- For more details, see notebook 'dle_fnd_initialization.ipynb'
- The matrix product of our inputs and weight matrix that we initialized from a standard normal distribution will, on average, have a standard deviation very close to the 
  square root of the number of input units
- What we'd like is each layer's output to have a standard deviation of 1
- Diving gaussian weights with square root of number of inputs of the layer
- Using ReLU and Kaiming HE initialization

## Loss functions
- Neural networks can be easily adapted to multiclass and multi-output problems
- - For multiclass problem, k-output neurons are typically used with a softmax activation function to make sure outputs sum to 1
- - Note that neurons is hidden layers commonly use different activation functions that output neurons
- For perceptron model, a squared error loss function is used
- - Many other loss functions are available
- For regression -> Mean squared error
- For binary classification -> Binary cross-entropy
- For multiclass classification -> Categorical cross-entropy

## Stochastic gradient descent
- Recall: the error is a function of the wights given a piece of training a data
- - Minimize error using the gradient of the error (loss) function
- - Gradient descent is the process of minimizing a function by following the gradients of the cost function
- Normal gradient descent presents all training instances to the network
- - One update of the weights follows based on averaged error over the whole training set
- - This is more precise, though time-consuming approach
- Stochastic gradient descent (SGD) is a more efficient approach
- - Updates weights after every example
- - But is more sensitive to particular examples
- - Looks like a more "drunk walk" towards the minimum
- Most implementations hence present a "mini-batch" of examples
- - Shuffle the training set , present in a smaller batches
- - Update weights after each mini-batch

## Backpropagation alternatives
- Other approaches to train artificial neural networks exist
- - Advanced non-linear optimization techniques
- - Hessian based optimization
- - Newton based optimization
- - Conjugate gradient
- - Levenberg-Marquardt
- - Genetic algorithms
- - Particle swarm optimization
- - Simulated annealing
- - Ant colony optimization
- - Firefly algorithm
- - Grey wolf optimizer
- - Harmony search
- - No training at all: transfer learning
- These are rarely used in practice

## Optimizers
- Even when using backpropagation, there are many ways to update the weights 
- All based on finding a fast and stable convergence to the minimum
- Keras comes with many optimizers already implemented
- See: https://keras.io/optimizers/

## Learning rate
- The learning rate determines the speed of the convergence
- - Higher: quicker towards the minimum, but risk of overshooting
- - Lower: slower, but more precise. Get risk trapped in local minimum
- - Adaptive learning rates: change the learning rate during training
- - Momentum based: also prevents overshooting
- Finding a good initial learning rate is a topic on its own
- - See: https://github.com/psklight/keras_one_cycle_clr
- - Plot loss function over a small batch for different learning rates, use best one to continue training
- Cyclical training rates: start with a low learning rate, increase it, and then decrease it again
- One cycle policy: a disciplined approach to NN hyperparameter tuning: learning rate, batch size, momentum and weight decay
- 
## Preventing overfitting
- Continious training will continue to lower the error on the training set, but will eventually lead to overfitting (memorizing the training set)
- As such, keeping track of a validation set is crucial
- - Stop training when validation error has reached its minimum (early stopping)
- Regularization: adding a penalty term to the loss function
- Dropout: randomly set a fraction of the input units to 0 at each update during training
- Plot the graph of the Epochs vs. Error

## Hyperparameter optimization
- Even for MLP, expect to tune:
- - Learning rate
- - Error function
- - Number of hidden layers
- - Number of neurons in each layer
- - Activation functions
- This trend of having to tune lots of hyperparameters will continue
- - Best practices and rules of thumb are hard to find and change rapidly: very empirical domain
