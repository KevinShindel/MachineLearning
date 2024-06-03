# Neural networks
- Is predictive analytics technique: Classification, Regression
- Biological view: inspired by the brain
- Statistical view: non-linear regression, generalization of statistical models
- Only 1 hidden layer: Universal approximation, Shallow neural networks
- Weights determined using optimization procedure: Minimize MSE, Done by analytical software 
- Number of hidden neurons: Depends on the complexity of the problem, Cross-validation 
- Hidden neurons tuned using iterative methods:
- - Split data into training, validation, and test sets
- - Vary hidden neurons from 1 to 10
- - Training NN on training set and measure performance on the validation set
- - Choose the number of hidden neurons with maximum validation set performance
- - Measure performance on the test set

# Neural Networks in Python

```python
import pandas as pd
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report

hmeq = pd.read_csv('c:/temp/hmeq.csv')

# Remove all missing values
hmeq = hmeq.dropna()

# Create dummies for JOB and REASON
cat_vars = ['REASON', 'JOB']

for var in cat_vars:
    cat_list = pd.get_dummies(hmeq[var], prefix=var, drop_first=True)
    hmeq = hmeq.join(cat_list)
    hmeq.drop(columns=var, inplace=True)

Y = hmeq.loc[: , 'BAD']
X = hmeq.drop(columns='BAD')

# Normalize the inputs
normalized_X = preprocessing.normalize(X)

# Create NN with 1 hidden layer, 3 neurons and logistic
# transformation functions

mynet = MLPClassifier(solver='adam', # Optimization algorithm 
                      activation='tanh', # Activation function
                      hidden_layer_sizes=(3), # 1 hidden layer with 3 neurons
                      early_stopping=True, # Stop when validation score is not improving
                      validation_fraction=0.3,  # 30% of the data is used for validation
                      random_state=12345) # Seed for random number generator

mynet = mynet.fit(normalized_X, Y) # Train the model

predictions = mynet.predict(normalized_X) # Make predictions

print(confusion_matrix(Y, predictions)) # Confusion matrix
print(classification_report(Y, predictions)) # Classification report
```

# Deep Learning Neural Networks
- Deep Learning triggers: GPU/TPU, Parallel Computing, Cloud Computing, NoSQL databases, TensorFlow, Keras, PyTorch
- Hidden layer extracts features from data
- Features automatically learned by the model
- Requires large amounts of data and computational power

# Opening Neural Networks Black Box
- NN commonly referred to as black box techniques because of complexity, mathematical relationship between outputs and inputs is not easily understood
- In credit risk modeling, analytical models need to be white box and interpretable
- Techniques to open up NN black box: variable selection, rule extraction, two-stage models.

# Variable Selection
- Statistical methods: magnitude of coefficients of variables (weights)
- More complicated in neural networks because of non-linear relationships
- If all variable-to-hidden layer weights are small, then the variable is not important
- Variable selection algorithm:
- - Train NN with all variables
- - Remove variable where variable-to-hidden layer weights are small
- - Retrain NN with remaining variables
- - If predictive power increases (or stays same), repeat process, if not, reconnect variable and stop.

# Rule Extraction
- Extract propositional IF-THEN rules from NN mimicking its behavior
- Decomposition rule extraction: Intertwined with internal workings of NN, Analyze weights, biases, and activation functions
- Pedagogical rule extraction: Consider NN as black box, use NN as oracle to label and generate additional training observations

# Decompositional Rule Extraction
Starts with:
1. Train a NN from original data
2. Remove redundant variables (aka features)
3. Categories hidden unit activations of pruned network by clustering
4. Extract rules that describe network outputs in terms of discretized hidden unit activations values
5. Generate rules that describe discretized hidden unit activation values in terms of network inputs
6. Merge 2 sets of rules generated in steps 4 and 5 to obtain a set of rules that relate network outputs to network inputs

# Decompositional Rule Extraction Example

```python
import pandas as pd
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.cluster import KMeans

# load data
hmeq = pd.read_csv('c:/temp/hmeq.csv')

# Remove all missing values
hmeq = hmeq.dropna()

# Create dummies for JOB and REASON
cat_vars = ['REASON', 'JOB']

for var in cat_vars:
    cat_list = pd.get_dummies(hmeq[var], prefix=var, drop_first=True)
    hmeq = hmeq.join(cat_list)
    hmeq.drop(columns=var, inplace=True)
    
Y = hmeq.loc[: , 'BAD']
X = hmeq.drop(columns='BAD')

# Normalize the inputs
normalized_X = preprocessing.normalize(X)

# Create NN with 1 hidden layer, 3 neurons and logistic
model = MLPClassifier(solver='adam', # Optimization algorithm 
                      activation='tanh', # Activation function
                      hidden_layer_sizes=(3), # 1 hidden layer with 3 neurons
                      early_stopping=True, # Stop when validation score is not improving
                      validation_fraction=0.3,  # 30% of the data is used for validation
                      random_state=12345) # Seed for random number generator

model = model.fit(normalized_X, Y) # Train the model

# Extract hidden layer activations
hidden_layer = model.hidden_layer_sizes[0]
hidden_layer_activations = model.predict(normalized_X)

# Cluster hidden layer activations
kmeans = KMeans(n_clusters=3, random_state=12345)
kmeans.fit(hidden_layer_activations.reshape(-1, 1))

# Extract rules
rules = []

for i in range(hidden_layer):
    cluster = kmeans.labels_[i]
    rule = 'IF cluster = ' + str(cluster) + ' THEN '
    for j in range(X.shape[1]):
        rule += X.columns[j] + ' = ' + str(X.iloc[i, j]) + ' AND '
    rule = rule[:-5]
    rules.append(rule)
    
for rule in rules:
    print(rule)
    
predictions = model.predict(normalized_X) # Make predictions
```

# Pedagogical Rule Extraction
- Use NN to relabel training data
- Build a decision tree using C4.5/CART/CHAID and relabeled data
- Use NN as oracle to generate additional training data when data becomes too partioned
- When generating new observations, take into account: previous tree splits, distribution of training data.
- Trepan algorithm: Combines a decision tree and NN to generate rules

# Quality of Extracted Rules
- Accuracy of an extracted rule set
- Fidelity: measures how well extracted a rule set mimics NN
- Conciseness of an extracted rule set

# Rule Extraction Example
- Trained and pruned NN using regularization
- Extracted rules using decomposition approach
- Transform propositional IF-THEN to decision table

# Two-Stage Model
- Start with original data
- Build simple model (Linear or Logistic Regression)
- Calculate errors from a simple model
- Build NN that predicts errors from a simple model
- Score new observations by adding output from simple and NN models
- Ideal balance between model interpretability and performance
- Do not estimate in one multivariate setup

# Self-Organizing Maps
- Unsupervised learning technique
- Feedforward neural network with 2 layers: input and output
- Neurons from output layer are organized in 2D grid
- Each input connected to all neurons in output layer
- Weight randomly initialized
- When training vector x is presented, weight vector for ech neuron is compared with x using 
  Euclidian distance metric.the winning neuron
- Decreasing learning rate and radius give stable
- Neuron with smallest distance is called  map after certain amount of training 
- Training stopped when BMUs remain stable or after number of iterations (500 times number of SOM neurons)
- Neurons move toward input neurons
- SOMs can be visualized using U-matrix, component planes, and hit maps
- For a small num of variables (4 or 5) the codebook vectors can be visualized in 2D or 3D and 
  is excellent visualization of the contribution to the SOM for each variable
- A codebook vector graph combines basically different component planes, in a single visualization.

# Self-Organizing Maps Evaluated

Advantages:
- Exploratory data analysis
- Can be combined with decision trees to further characterize clusters
Disadvantages:
- Difficult to compare and interpret
- Experimental evaluation needed