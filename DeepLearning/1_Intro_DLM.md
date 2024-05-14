
## Applications of Deep Learning
- NLP
- Speech Recognition
- Image Recognition
- Video Analysis
- Self Driving Cars
- Customer experience, health care, finance, etc.

## Prerequisites
- ML concepts and tech.
- Python programming, notebooks
- Keras and TensorFlow
- Scikit-learn, Pandas, Numpy
- Text processing with NLTK


## What is Deep Learning?
- Subset of ML
- Neural Networks with many layers
- Imitates how human process data and make decisions
- Exponential growth in data and computing power in the last few years
- Powered by advances in large-scale data processing and interface

### Linear Regression
- A linear model
- Relationship between dependent and independent variables
- Predicts the value of dependent variable based on independent variable
- Slope and intercept models the relationship
- Simple linear regression
    - y = ax + b
    - y = a1x1 + a2x2 + a3x3 + ... + b
    - y - dependent variable
    - x - independent variable
    - a - slope
    - b - intercept

### Build a Linear Regression Model
- Find values for slope and intercept # 5=2a+b, 9=4a+b
- Use known values of x and y ( multiple values ) # 9 = 4a + 5 - 2a
- Multiple independent variables make it complex # a = (9-5)/(4-2), b = 5-2a 

### Logistic Regression
- A binary classification model
- Relationship between dependent and independent variables
- Output is a probability (0 or 1)
- Simple logistic regression
    - y = f*(ax+b)d
    - y - dependent variable
    - x - independent variable
    - a - slope
    - b - intercept
    - f - logistic function

### An analogy for Deep Learning

- Deep learning is a complex and iterative process
- Starts with random initialization and works towards a solution

### Perceptron 
- A single layer neural network
- Algorithms for supervised learning of binary classifiers
- Resembles a biological neuron
- Based on logic regression
- Perceptron formula
    - y = f*(ax+a2x2+...+b)d
    - y = f*(w1x1+w2x2+...+b)
    - w - weights
    - b - bias
    - f - activation function ( 1 if value > 0, 0 otherwise )


### Artificial Neural Networks
- A network of perceptrons, modeled after the human brain
- Perceptrons are called nodes in the neural network
- Nodes organized as layers
- Each node has a weight and bias and activation function
- Each node is connected to all nodes in the next layer

### ANN working process
1. Inputs (independent variables) are send from input layer to hidden layer
2. Inputs passed on to the nodes in the next hidden layer
3. Each node computes its output based on the weights and bias and activation function
4. Node output is then passed on as inputs to the next layer

### Training ANN 
- Model is represented by parameters and hyperparameters
  - Weights, biases, activation functions
- Training a model means determining the best values for these parameters and hyperparameters that maximize the accuracy of the model.
- Inputs, weights and biases might by n-dimensional arrays.

### Training process
- Use training data ( known values of inputs and outputs )
- Create network architecture with intuition
- Start with random weights and biases
- Minimize error in predicting known outputs from inputs
- Adjust weights and biases to minimize error
- Improve model by adjusting layers, node counts, and other hyperparameters