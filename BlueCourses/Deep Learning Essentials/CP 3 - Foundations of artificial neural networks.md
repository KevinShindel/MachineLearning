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

## A simple iterative approach

## ☞ A simple perceptron in Python

## Gradient descent

## The XOR problem

## Multilayer perceptron (MLP) networks

## Backpropagation

## Automatic differentiation

## ☞ Automatic differentiation in TensorFlow

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
