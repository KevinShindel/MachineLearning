# Model Interpretation
- Depends on application and business analyst: logistic regression, decision trees, random forests, gradient boosting, neural networks, etc.
- White-box models: credit risk modeling, medical diagnosis, etc.
- ZZZBlack-box models: response prediction, recommendation systems, fraud detection etc.

# Feature Importance
- Which features are important according to the model?
- Different techniques exist: 
- - Based on mean decrease in impurity in tree based ensembles
- - Based on position in trees
- - Drop feature importance
- - Permutation importance

# Permutation based feature importance
- Start with trained model on given data set and performance measure (accuracy, AUC, profit, etc.)
- Randomly permutate values for feature under study.
- Use trained model to predict observations again.
- Importance = baseline performance measure - performance measure on permuted data set.
- Note that this make take interactions into account (shuffling breaks interaction effects).
- Indicates which features are important, but not to which extent they affect outcome
- Can also be used for feature selection: retrain model on top-N features
- Note that correlated features likely to share importance which might lead to misinterpretation
- Possible to investigate multiple features at once to zoom in on interaction effects.
- Makes sense to assess drop in predictive power on both training and test set:
- - Training set: learn and understand how the model has actually relied on each feature
- - Test set: verify whether feature rankings are similar and there was no overfitting.

# Permutation based feature importance in Python
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost.sklearn import XGBClassifier

# pip install rfpimp
from rfpimp import importances, plot_importances

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

feature_names = X.columns
feature_names

# XGBClassifier works a bit differently from normal sklearn classifiers
# As such, we'll need to provide a raw numpy array to XGBClassifier

X_np = X.values
xgboostmodel = XGBClassifier()
xgboostmodel.fit(X_np, y)

y_pred = xgboostmodel.predict(X_np)

print("Number of mislabeled points out of a total %d points : %d"
     % (X.shape[0],(y != y_pred).sum()))

# XGBClassifier changes the feature names to f0, f1, ..., so we need to do the same

X_fnames = X.copy()
X_fnames.columns = ['f{}'.format(i) for i in range(X.shape[1])]

imp = importances(xgboostmodel, X_fnames, y, n_samples=-1)

# Put the correct feature names back in
imp.index = hmeq.drop(columns='BAD').columns[[np.where(X_fnames.columns==idx)[0][0] for idx in imp.index]]

viz = plot_importances(imp)
viz.view()
```

# Partial dependence plots
- Understand how feature impacts outcome of model:
- - Keep feature under study as-is and impute other with median and mode over all observations
- - Trained model predict new dataset and plots prediction results for values of feature under study
- Absence of evidence does not mean evidence of absence (e.g. interaction effects)
- Compare data and partial dependence plot to detect interaction effects: 
- - Categorize feature (age, income, etc.) and look at percentage of target (e.g. bad loans) in each category
- - E.g. "churn rate drops for customer between 30 and 40 years old" vs "churn probability stays 
    constant customers between 30 and 49 years old" indicates presence of interaction effect: 
    age alone not a sufficient explanation.
- Need to inspect both training and test set to avoid overfitting.
- Possible to keep more than one feature as-is whilst replacing others with their median or mode.
- Harder to visualize (e.g. contour plots, 3D plots, etc.).

# Partial dependence plots in Python
```python
import pandas as pd
import matplotlib.pyplot as plt
from xgboost.sklearn import XGBClassifier
from sklearn.inspection import plot_partial_dependence

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

feature_names = X.columns
feature_names

# XGBClassifier works a bit differently from normal sklearn classifiers
# As such, we'll need to provide a raw numpy array to XGBClassifier

X_np = X.values
xgboostmodel = XGBClassifier()
xgboostmodel.fit(X_np, y)

y_pred = xgboostmodel.predict(X_np)

print("Number of mislabeled points out of a total %d points : %d"
     % (X.shape[0],(y != y_pred).sum()))

# XGBClassifier changes the feature names to f0, f1, ..., so we need to do the same

X_fnames = X.copy()
X_fnames.columns = ['f{}'.format(i) for i in range(X.shape[1])]

features = [0, 1, 2, 3, 4, (1,2)]
fig = plt.figure(figsize=(10,10))
plot_partial_dependence(xgboostmodel, X_fnames, features, feature_names=feature_names,
                        n_jobs=3, grid_resolution=50, fig=fig)
plt.show()
```

# Individual conditional expectation (ICE) plots
- Very similar idea to partial dependence plots
- Key idea is not replace features with median/mode, keep feature as is, but create a new 
  observations based on values of feature under study.
- Also, possible to define grid-based range over feature under study between its observed minimum 
  and maximum.

# Visual Analytics

# Decision Tables

# LIME

# Shapley Value
