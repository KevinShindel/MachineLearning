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
- Every original observations leads to multiple rows in modified data set.
- Let model predict over all these observations and plot results.
- For each distinct value for feature under study, we now have multiple predictions and can plot 
  these.
- ICE plots well suited to show behaviour of feature across data set.
- Recommended practice is to inspect ICE plots for important features.
- Partial dependence and ICE plots should be of key tool to any machine learner workeing with 
  black box models.
- Essential tools to bridge gap between complex models with great predictive accuracy and 
  interpretability: validate the model, bridge gap towards business stake holders, provide 
  explanations to end users.

# ICE plots in Python
```python
import pandas as pd
import matplotlib.pyplot as plt
from xgboost.sklearn import XGBClassifier

# pip install pdpbox
from pdpbox import pdp

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

pdp_value = pdp.pdp_isolate(
    model=xgboostmodel, dataset=X_fnames, feature='f1', model_features=X_fnames.columns
)

fig, axes = pdp.pdp_plot(pdp_value, 'MORTDUE', center=True, plot_lines=True, frac_to_plot=100)
plt.show()
```

# Visual Analytics
- Reduce cognitive overload by having users interact with data and/or analytical models using 
  visual tools.
- Help data scientists + business users to explore and better understand data + models
- "A picture is worth a thousand words"

# Decision Tables
- Condition entry describes relevant subset of values (state) for given condition subject 
- Every column in entry part of DT is a classification rule.
- Decision tables useful for:
- - visualizing rules in intuitive and user-friendly way
- - checking rules for completeness and anomalies

# LIME
- Local Interpretable Model-agnostic Explanations (introduced by Ribeiro et al. in 2016)
- LIME implements a "local surrogate model" to explain predictions of black box models
- Works on any type of black box: model-agnostic
- Works with tabular and other types of data (text, images, etc.)
- LIME generates a new data set by perturbing the original data set

# LIME in Python
```python
import pandas as pd
import numpy as np
import sklearn.preprocessing
from sklearn.svm import SVC

hmeq = pd.read_csv('hmeq.csv')

# pip install lime 
import lime
import lime.lime_tabular

# Remove all missing values
hmeq = hmeq.dropna()

# From pandas DF to Numpy array
hmeq = hmeq.values

# One Hot Encoding
labels = hmeq[:,0]
le= sklearn.preprocessing.LabelEncoder()
le.fit(labels)
labels = le.transform(labels)
class_names = le.classes_
hmeq = hmeq[:,1:]

categorical_features = [3,4]

categorical_names = {}
for feature in categorical_features:
    le = sklearn.preprocessing.LabelEncoder()
    le.fit(hmeq[:, feature])
    hmeq[:, feature] = le.transform(hmeq[:, feature])
    categorical_names[feature] = le.classes_

hmeq = hmeq.astype(float)

encoder = sklearn.preprocessing.OneHotEncoder(categorical_features=categorical_features)

np.random.seed(1)
train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(hmeq, labels, train_size=0.75)

encoder.fit(hmeq)
encoded_train = encoder.transform(train)

# We use SVMs

clf = SVC(probability=True)
clf.fit(encoded_train, labels_train)

sklearn.metrics.accuracy_score(labels_test, clf.predict(encoder.transform(test)))

predict_fn = lambda x: clf.predict_proba(encoder.transform(x)).astype(float)

# Explaining predictions

# We now create our explainer. The categorical_features parameter lets it 
# know which features are categorical (in this case, all of them). The 
# categorical names parameter gives a string representation of each 
# categorical feature's numerical value, as we saw before.

# Our explainer (and most classifiers) takes in numerical data, even if 
# the features are categorical. We thus transform all of the string 
# attributes into integers, using sklearn's LabelEncoder. We use a dict 
# to save the correspondence between the integer values and the original 
# strings, so that we can present this later in the explanations.

feature_names = ["LOAN","MORTDUE","VALUE","REASON","JOB","YOJ","DEROG","DELINQ","CLAGE","NINQ","CLNO","DEBTINC"]

explainer = lime.lime_tabular.LimeTabularExplainer(train, class_names =  ['BAD', 'GOOD'], feature_names = feature_names, categorical_features = categorical_features, categorical_names = categorical_names, kernel_width=3)

# We now show a few explanations. These are just a mix of the continuous and categorical examples we showed before. 
# For categorical features, the feature contribution is always the same as the linear model weight.

i = 100
exp = explainer.explain_instance(test[i], predict_fn, num_features=5)
# exp.show_in_notebook(show_all=False)
exp.show_in_notebook(show_all=True)

# Another observation explained
i = 101
exp = explainer.explain_instance(test[i], predict_fn, num_features=5)
# exp.show_in_notebook(show_all=False)
exp.show_in_notebook(show_all=True)

# And another...
i = 102
exp = explainer.explain_instance(test[i], predict_fn, num_features=5)
# exp.show_in_notebook(show_all=False)
exp.show_in_notebook(show_all=True)


# And another...
i = 3
exp = explainer.explain_instance(test[i], predict_fn, num_features=5)
# exp.show_in_notebook(show_all=False)
exp.show_in_notebook(show_table = True, show_all=True)
```

# Shapley Value
- Represents the "payout" of a player in a cooperative game
- Can be used the contribution of a feature to a model prediction
- Not always feasible to consider all possible subsets of features
- Shapley value is the average of the all these samples

# Shapley Value in Python
```python
#pip install shap

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import shap

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
print("Accuracy = {0}%".format(100*np.sum(linear_lr.predict(X_test) == y_test)/len(y_test)))

explainer = shap.TreeExplainer(clf, data=X_train) 
# There is also DeepExplainer, KernelExplainer and GradientExplainer

shap_values = explainer.shap_values(X_test)

shap.initjs()
shap.force_plot(explainer.expected_value[0], shap_values[1][0,:], X_test.iloc[0,:])

shap.initjs()
shap.force_plot(explainer.expected_value[1], shap_values[1][1,:], X_test.iloc[1,:])

shap.summary_plot(shap_values, X_test, plot_type="bar")
```
