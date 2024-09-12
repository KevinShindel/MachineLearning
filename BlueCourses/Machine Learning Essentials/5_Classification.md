
# Classification problem:
- Given observation with predictor determine class from a set.
- Classes known beforehand
- Binary versus multiclass

# Linear Regression
- Linear regression: estimates linear relationship between target variable and predictor variables (features)
- Ordinary Least-Squares (OLS) - is a method for estimating the unknown parameters in a linear regression model. OLS chooses the parameters of a linear function of a set of explanatory variables by minimizing the sum of the squares of the differences between the target-dependent variable and those predicted by the linear function.
- Linear regression with transformation such that output is always between 0 and 1, and can be interpreted as probability
- After model has been estimated using historical data, can use it to score or assign probabilities to new observations
- Doubling Amount:
- - Amount of change required for doubling primary outcome odds

## Logistic Regression in Python

```python
import pandas as pd
import statsmodels.api as sm

hmeq = pd.read_csv('c:/temp/hmeq.csv')

# Remove all missing values
hmeq = hmeq.dropna()

# Create dummies for JOB and REASON
cat_vars = ['REASON', 'JOB']

for var in cat_vars:
    cat_list = pd.get_dummies(hmeq[var], prefix=var, drop_first=True)
    hmeq = hmeq.join(cat_list)
    hmeq.drop(columns=var, inplace=True)

Y = hmeq[['BAD']]
X = hmeq.drop(columns='BAD')

logit_model = sm.Logit(Y, X)

result = logit_model.fit()

print(result.summary())
```

## Nomograms

- Nomograms are a graphical representation of a mathematical equation
- It Can be used to estimate the probability of an event
- It Can be used to estimate the effect of a predictor on the outcome

## Logistic Regression and WOE

- Weight of Evidence (WOE) is a measure of the "strength" of a grouping for separating good and bad
- No dummy variables needed
- More robust to outliers

Table of example WOE values: custID, age, age_group, age_woe

| custID | age | age_group | age_woe |
|--------|-----|-----------|---------|
| 1      | 25  | 20-30     | 0.5     |
| 2      | 35  | 30-40     | 0.7     |
| 3      | 45  | 40-50     | 0.9     |
| 4      | 55  | 50-60     | 1.1     |

# Decision Trees

- Split based on the lowest impurity
- Different ways to calculate impurity: Gini, Entropy, Misclassification
- - Gini: measure of how often a randomly chosen element would be incorrectly classified
- - Entropy: measure of uncertainty
- - Misclassification: measure of how often a randomly chosen element would be incorrectly classified
- Recursively partition training data
- Recursive Partitioning Algorithms (RPAs)
- Tree induction algorithms: CHAID, CART, ID3, C4.5, C5.0
- - CHAID: Chi-squared Automatic Interaction Detection (Kass, 1980)
- - CART: Classification and Regression Trees (Breiman et al., 1984)
- - ID3: Iterative Dichotomiser 3 (Quinlan, 1986)
- - C4.5: Successor of ID3 (Quinlan, 1993)
- - C5.0: Successor of C4.5 (Quinlan, 1997)
- Classification trees: predict class, target is categorical (e.g., good, bad) (PD)
- Regression trees: predict value, target is continuous (e.g. price, temperature) (LGD, EAD)
- Splitting decision: Which variable to split on and where to split
- Stopping criteria: When to stop adding nodes?
- Assigment decision: How to assign a class to a leaf node? 
- If a tree continues to split, it will:
- - Overfit: model will be too complex and will not generalize well
- - Over-detail: model will be too detailed and will not generalize well
- - Over-complex: model will be too complex and will not generalize well
- - Fit noize in data: model will be too detailed and will not generalize well
- - Generalize poorly: model will not generalize well

Advantages of Decision Trees:
- Easy to understand and interpret
- Non-parametric (no assumptions about the shape of the data)
- Robust to outliers

Disadvantages of Decision Trees:
- Sensitive to changes in training data
- Unstable (small changes in data can lead to different trees)
- Prone to overfitting

# Decision Trees in Python

```python
import pandas as pd
from sklearn import tree
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

Y = hmeq[['BAD']]
X = hmeq.drop(columns='BAD')

mytree = tree.DecisionTreeClassifier(max_depth=4)
mytree = mytree.fit(X, Y)
predictions = mytree.predict(X)
print(confusion_matrix(Y, predictions))
print(classification_report(Y, predictions))
```

# K-Nearest Neighbors (KNN) Classification

- KNN is a non-parametric method used for classification and regression
- Classify new observation by looking at the K closest observations in the training set
- The Neighbors are deemed if it has the smallest distance to the new observation 
- Beware of standardize variables before using KNN

- Advantages of KNN:
- - Intuitive simple
- - Easy to understand

- Disadvantages of KNN:
- - How to choose K?
- - large computing power requirement, because, for each new observation, the distance to all other observations must be calculated
- - Impact of irrelevant features

# K-Nearest Neighbors in Python

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

hmeq = pd.read_csv('hmeq.csv')

# Remove all missing values
hmeq = hmeq.dropna()

# Create dummies for JOB and REASON
cat_vars = ['REASON', 'JOB']

for var in cat_vars:
    cat_list = pd.get_dummies(hmeq[var], prefix=var, drop_first=True)
    hmeq = hmeq.join(cat_list)
    hmeq.drop(columns=var, inplace=True)

X = hmeq.drop(columns='BAD')
y = hmeq.BAD
X = StandardScaler().fit_transform(X)

clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(X, y)
```

# Multiclass Classification

- Examples: prediction corporate ratings, classifying a set of images, classifying a set of documents
- Map multiclass classification to set of binary classifications
- One v One:
- - Contrast every class against every other class
- - For k classes, build k(k-1)/2 classifiers
- - Assign new observation to class with most votes
- One v All:
- - Contrast every class against all other classes
- - For k classes, build k classifiers
- - Assign new observation to class with the highest probability

# One vs One Coding


# One vs All Coding

# Multiclass Decision Trees

- It Can be easily generalized to multiclass problems
- Impurity criteria: Gini, Entropy, Misclassification