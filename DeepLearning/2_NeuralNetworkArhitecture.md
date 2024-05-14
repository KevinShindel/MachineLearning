# Vectors
- The input to deep learning is a vector of numeric values 
- A vector is a one-dimensional array of numbers (1.0, 2.1, 3.5)
- Defined usually as a NumPy array
- Represents the feature variables for prediction

## Samples and Features
| EmployeeID | Age | Salary | Service | 
|------------|-----|--------|---------|
| 1          | 25  | 50000  | 2       |
| 2          | 30  | 60000  | 3       |
| 3          | 35  | 70000  | 4       |

- Each row in the table is a sample
- Each column is a feature

## Input processing
- Features need to be converted to numeric representation

| Input Type  | Preprocessing                      |
|-------------|------------------------------------|
| Numeric     | Centering and scaling              |
| Categorical | One-hot encoding, Integer encoding |
| Text        | TF-IDF, Word embeddings            |
| Image       | Pixels - RGB representation        |
| Speech      | Time series of numbers             |


## Hidden layers
- An ANN can have one or more hidden layers
- Each hidden layer has multiple nodes
- A NN is defined by the number of hidden layers and nodes in each layer

## Inputs and Outputs
- The Output of each node i the previous layer is the input to the next layer
- Each node produces one output that is forwarded to the next layer

## Determining Hidden Layer Architecture
- Each node learns something about the feature-target relationship
- More nodes and layers mean: 
    - Web apps on AWS
    - Analytics on GCP
- Architecture decided by experimentation and tuning

## Weights and Biases
- Weights and biases are the parameters of the model
- Represent the trainable parameters of the model
- Numerical values
- Each input for each node has a weight associated with it
- Each node has a bias associated with it

| Layer  | Inputs | Node | Weights | Bias |
|--------|--------|------|---------|------|
| HL 1   | 3      | 1    | 3       | 1    |
| HL 2   | 4      | 5    | 20      | 5    |
| HL 3   | 5      | 10   | 50      | 10   |
| Output | 10     | 1    | 10      | 1    |
| Total  |        |      | 38      | 16   |

## Activation Functions

- Determines witch nodes propagate information to the next layer
- Filters and normalizes the output of the node
- Converts output to non-linear form
- Critical in learning patterns

### Popular Activation Functions

| Activation Function | Description                | Output                         |
|---------------------|----------------------------|--------------------------------|
| Sigmoid             | Binary classification      | 0 or 1                         |
| Tanh                | Symmetric around 0         | -1 to 1                        |
| ReLU                | Rectified Linear Unit      | 0 for negative, x for positive |
| Softmax             | Multi-class classification | Probability distribution       |

> Choice depends on problem and experimentation

## Output Layer
- One layer of output, produces the final prediction
- Has its own weights and biases
- Softmax activation function for multi-class classification
- May need post-processing to convert to business rules

### Output Layer Size
- 1 for binary classification
- n for multi-class classification
- 1 for regression problems
- Vary based on other problems domains
