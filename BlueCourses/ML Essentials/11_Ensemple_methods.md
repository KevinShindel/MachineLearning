# Ensemble methods
- Instead of using a single decision tree, use different decision trees and combine outputs.
- Different decision trees cover different parts of input space.
- Bagging, Random Forest, XGBoost are some of the ensemble methods.
- These are meta-learning schemes that can be applied for every unstable classifier, but are thr 
  most popular for decision trees.

# Bootstrapping
- Very popular technique for ensemble methods.
- Given a dataset of size N, create a new dataset of size N by sampling with replacement.
- This new dataset is called a bootstrap sample.
- The original dataset is called the population.
- The bootstrap sample is used to train a model.
- This process is repeated multiple times to create multiple models.
- The final output is the average of the outputs of all models.

# Bagging
- Bagging stands for Bootstrap Aggregating.
- It is a technique to reduce variance of a model.
- Take N bootstrap samples from the population.
- Build a classifier (decision tree) on each bootstrap sample.
- For classification: New observation called by letting N classifiers vote majority voting 
  scheme, ties resolved arbitrarily.
- For regression: New observation called by averaging the outputs of N classifiers.

# Bagging in Python
```python
import pandas as pd
import numpy as np

from sklearn.ensemble import BaggingClassifier

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

# BaggingClassifier uses a decision tree by default
clf = BaggingClassifier(n_estimators=100)
clf.fit(X, y)
```
# Boosting
- Build multiple classifier using a weighted sample of the training data
- Iteratively re-weight the training distribution according to classification error: 
  Misclassified cases get a higher weight, Correctly classified cases get a lower weight, 
  Difficult observations get more attention.
- Either classification algorithm can directly work with weighted training examples or if not 
  sample new training set according to weight distribution.
- Final model is a weighted combination of all individual classifiers.
- AdaBoost (Adaptive Boosting) is a popular implementation of boosting.
- AdaBoost variants:
- - AdaBoost.M1: (Freund and Schapire, 1996) - Binary classification
- - AdaBoost.M2: (Schapire and Singer, 1999) - Multi-class classification
- - AdaBoost.R: (Drucker et al., 1997) - Regression
- - AdaBoost.RT: (Freund and Schapire, 1997) - Regression with decision trees
- - Gradient Boosting: (Friedman, 2001) - Generalization of AdaBoost
- Advantages of AdaBoost:
- - T is the only parameter to tune
- - Fast, simple and easy to program
- Disadvantages of AdaBoost:
- - Boosting can have a risk of overfitting to the hard examples in the data.

# AdaBoost in Python
```python
import pandas as pd
import numpy as np

from sklearn.ensemble import AdaBoostClassifier

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

# AdaBoostClassifier uses a decision tree by default
clf = AdaBoostClassifier()
clf.fit(X, y)
```

# Random Forest
- First introduced by Breiman in 2001.
- Data set with N observations and M features.
- Can be used for classification (PD) and regression (LGD, CCF).
- Random Forest is an ensemble of decision trees.
- Each tree is built on a bootstrap sample of the data.
- Advantages of Random Forest:
- - Only 2 parameters to determine: T and M
- - Empirical evaluation shows that AdaBoost is something even better than AdaBoost.
- - Able to cope with very large number of features.
- - Faster than begging or boosting because of the random feature selection.
- Disadvantages of Random Forest:
- - Decreased interpretation because of multiple trees.
- Gini importance: mean Gini gain produced by variable over all trees.
- Permutation importance:
- - Each tree has it own out-of-bag sample of data that was not used during construction
- - Calculate variable importance as follows: Measure prediction accuracy on out-of-bag sample, 
    values of variable in out-of-bag are randomly shuffled, measure decrease in prediction 
    accuracy on shuffled data.

# Random Forest in Python
```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

hmeq = pd.read_csv('hmeq.csv')

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

myforest = RandomForestClassifier(n_estimators=100, 
                   max_depth=4, random_state=0)

myforest = myforest.fit(X, Y)
predictions = myforest.predict(X)

print(confusion_matrix(Y, predictions))
print(classification_report(Y, predictions))
```

# XGBoost
- First introduced by Chen and Guestrin in 2016.
- Combine weak learners to create a strong learner.
- Series of decision trees created which together form single predictive model.
- New learners trained on errors of previous learners.
- Tree boosting: tree in ensemble if fit to residual of prediction from earlier trees 
- Residual defined in terms of derivative of loss function.
- Optimized distributed gradient boosting library designed to be highly efficient, flexible and 
  portable.
- Supports various objective functions, including regression, classification and ranking.
- Open-source implementation of gradient boosting framework (C++, Java, Python, R and Julia).
- Push the limit of computations resources for boosting algorithms.
- Uses a more regularized model and attention as the algorithm of choice for many winning teams 
  of machine learning competitions (Kaggle, KDD cup).

# XGBoost in Python
```python
import pandas as pd
from xgboost.sklearn import XGBClassifier 

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

xgboostmodel = XGBClassifier()
xgboostmodel.fit(X,Y)

y_pred=xgboostmodel.predict(X)

print("Number of mislabeled points out of a total %d points : %d"
      % (X.shape[0],(Y != y_pred).sum()))
```