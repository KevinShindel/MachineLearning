# Bayesian Networks
- Bayesian network represents a joint probability distribution over a set of categorical, 
  stochastic variables.
- Ideal for combining domain knowledge with data patterns.
- Bayesian networks are directed acyclic graphs (DAGs) where nodes represent random variables 
  and edges represent probabilistic dependencies between the variables.
- Learning a Bayesian network is a two-step process:
  1. Structure learning: Determine the graph structure of the Bayesian network.
  2. Parameter learning: Determine the parameters of the Bayesian network.
- Domain expert vs Data Driven Approach
- Domain expert approach:
  1. Interview credit experts
  2. list important variables
  3. Draw dependencies between variables
  4. Manage knowledge conflicts
  5. Estimate the conditional probabilities from the data
- Data driven approach:
- - Correlation-based methods
- - Markov Chain Monte Carlo (MCMC) methods
- - Independence test methods (Chi-square test, Fisher's exact test)
- - Integer linear programming methods
- Hybrid methods

# Example Bayesian Network Classifier
- Bayesian network classifier is a probabilistic model that represents a joint probability distribution 
  over a set of categorical, stochastic variables.
- The classifier is a directed acyclic graph (DAG) where nodes represent random variables and edges 
  represent probabilistic dependencies between the variables.
- The classifier is a generative model that can be used to predict the class label of a new instance 
  by computing the posterior probability of the class label given the instance.

# Naive Bayes Classifier
- Is most simple Bayesian network classifier. It learns the class conditional probabilities.
- New case classified by using Bayes rule to compute posterior probability of each class given the case.
- Simplifying assumption assumes that variables are conditionally independent given the class label.
- No need to estimate denominator instead, normalize probabilities to sum to 1.
- Probabilities estimated using frequency counts for categorical variables and normal or kenel 
  density-based methods for continuous variables.

# Naive Bayes Classifier Python Example
```python
import pandas as pd
from sklearn.naive_bayes import GaussianNB

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

gnb = GaussianNB()

y_pred = gnb.fit(X,Y).predict(X)

print("Number of mislabeled points out of a total %d points : %d"
      % (X.shape[0],(Y != y_pred).sum()))
```

# Tree Augmented Naive Bayes Classifier (TAN)
- Introduced in 1998 by Friedman et al.
- Relax independence assumption by allowing arcs between variables
- Arc from x to y if x and y are conditionally independent given the class label
- Each variable has, as parents the class variable and at most one other variable.
- Experimental results indicated that TANs outperform Naive Bayes with the same computational 
  complexity and robustness.

# Bayesian Networks Examples
- Predict spending evolution (sign of regression slope) for customers to detect churn for DYI chain.
- Predict the probability of a customer to buy a product based on the customer's profile.
- 
