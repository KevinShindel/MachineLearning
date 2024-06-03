# Problems with NN
- Multimodal objective function: multiple local minima
- Higly paramaterized: choose number of hidden layers, number of neurons in each layer, activation function, learning rate, etc.

# Linear Programming
- Linear programming is a method to achieve the best outcome in a mathematical model whose requirements are represented by linear relationships.
- Minimize the sum of the absolute values of the deviations from the target values. (MSD)
- Popular in early credit scoring (Fair, Isaac and Company

# Linear Separable Case
- The goal is to find the hyperplane that separates the two classes.
- Consider hyperplane, which minimize the distance between the two classes.
- Large margin separating hyperplane
- Given a set of training data, the SVM algorithm outputs an optimal hyperplane which categorizes new examples.
- Assume that the data is linearly separable.
- Maximize or minimize the margin between the two classes.
- Optimization problem
- The classifier then becomes a linear combination of the support vectors.
- using Lagrangian optimization, a quadratic programming problem is obtained.
- Solution of QP problem is global: convex optimization problem.
- Training points that lie on one of the hyperplanes are called support vectors.

# Linear Non-Separable Case
- Allow for errors by introducing slack variables in the inequality constraints.
- The optimization problem then becomes a quadratic programming problem.
- The solution is a soft margin classifier.

# Non-Linear SVM Classifier
- Is a linear classifier in a higher dimensional space.

# RBF SVM in Python
```python
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

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

clf = SVC()
clf.fit(X, y)
```


# Kernel Functions
- The kernel function is used to map the input space into a higher dimensional space.
- The kernel function is a similarity function.
- The kernel function is a dot product in the higher dimensional space.
- Kernel functions types: linear, polynomial, radial basis function (RBF), sigmoid.

# NN interpretation of SVM Classifier
- Number of hidden neurons determined automatically.

# Tuning the hyperparameters
1. Set aside two-thirds of the data for training/validation set and the remaining one third for 
     testing.
2. Starting from i=0, perform 10-fold cross-validation on the training/validation set for ech 
     combination from the initial candidate tuning sets.
3. Choose optimal from tuning sets and look at best cross-validation performance for ech 
     combination.
4. if i=i_max go to step 5, otherwise increment i by 1 and go to step 2.
5. Construct the SVM classifier using the total training/validation set the optimal choise of 
     the tuned hyperparameters.
6. Assess the test set accuracy by means of the independent test set.

# Tuning hyperparameters of RBF SVM in Python
```python
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

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

# Create a train and test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Default tuning procedure based on a cross-validated grid search

param_grid = {
    'C': [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500],
    'gamma': [0.5, 5, 10, 15, 25, 50, 100, 250, 500]
}

clf = GridSearchCV(SVC(), param_grid, cv=10)
clf.fit(X_train, y_train)

print("Best parameters set found on train set:")
print(clf.best_params_)

print("CV score:")
print(clf.best_score_)

# Further fine-tuning by iteratively fine-graining grid

def get_fine_tuned_range(param_list, current_value, n_new_params=10):
    idx = param_list.index(current_value)
    start = param_list[idx - 1] if idx - 1 >= 0 else param_list[idx] / 2
    end = param_list[idx + 1] if idx + 1 < len(param_list) else param_list[idx] * 1.5
    return list(np.linspace(start, end, n_new_params))
    
for i in range(1,5):
    print("Iteration", i)
    new_c = get_fine_tuned_range(param_grid['C'], clf.best_params_['C'])
    new_gamma = get_fine_tuned_range(param_grid['gamma'], clf.best_params_['gamma'])
    param_grid = { 'C': new_c, 'gamma': new_gamma }
    print("New parameter grid:", param_grid)
    clf = GridSearchCV(SVC(), param_grid, cv=10)
    clf.fit(X_train, y_train)
    print("Best parameters are now:", clf.best_params_)
    print("Score is now:", clf.best_score_)

# Retrain on complete training set

final_clf = SVC(**clf.best_params_)
final_clf.fit(X_train, y_train)

print("Classification report on test set:")
print(classification_report(y_test, final_clf.predict(X_test)))
```

# Benchmarking Study
- 10 public available binary classification datasets and 10 public available multiclass 
  classification data sets.
- Various domains: medicine, artificial, credit scoring, sociology and others.
- Studied both SVMs and variants thereof, linear discriminant analysis, quadratic discriminant 
  analysis, logistic regression, C4.5, oneR, k-nearest neighbors, neural networks, and random
- RBF kernels, linear kernels and polynominal kernels
- Used One-vs-One coding for multiclass problems and contracted with others
- Performance was compared using a paired t-test.
- Average ranking were computed and compared using a sign test.
- Six months of computer time on a 16-node cluster.
- RBF SVMs and RBM LS-SVMs were the best classifiers compared to the other algorithms.
- For the multiclass case, One-vs-One coding scheme yielded better performance that the Minimum 
  Output Coding Scheme.
- Simple classification algorithms (e.g. logistic regression) also yield very good results.
- Most data sets are only weakly non-linear.
- However, consider the importance of marginal performance benefits
- Similar to classification case, construct Lagrangian and solve dual formulation, which has 
  global minimum.
- Notice that SVM algorithm depends only on dot product between various observations

# One-CLass SVMs
- One-class SVMs are used for anomaly detection.
- The goal is to find a hyperplane that separates the normal data from the abnormal data.
- The hyperplane is chosen to maximize the margin between the normal data and the hyperplane.
- The hyperplane is chosen to minimize the number of abnormal data points that lie on the normal side of the hyperplane.

# One-class SVM in Python
```python
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.svm import OneClassSVM

# Create a simple 2d data set
X, y = make_blobs(centers=[[0, 0], [0, 0]], cluster_std=[5, 0.5])
plt.scatter(X[:,0], X[:,1])

# One class SVM

clf = OneClassSVM(gamma='scale')
clf.fit(X)

xx, yy = np.meshgrid(np.linspace(-15, 15, 250),
                     np.linspace(-15, 15, 250))
pred = clf.predict(X)

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.scatter(X[:,0], X[:,1], c=pred+1)
plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')
```

# Extension to SVMs
- SVMs can be extended to regression problems.
- The goal is to find a hyperplane that minimizes the sum of the deviations from the target values.
- The Lagrangian optimization is used find the optimal hyperplane.
- LS-SVMs solve a set of linear equations instead of a convex QP problem.
- LS-SVMs are faster than SVMs.
- Relevance Vector Machines (RVMs) are a Bayesian extension of SVMs.
- Transductive Support Vector Machine (TSVM) is used for semi-supervised learning.
- Proximal Support Vector Machine (PSVM) is used for large-scale learning.
- 

# Opening SMV Black box
- Use ideas from NN rule extraction: Decompositional approach, Padagogical approach, 
  Knowledge-based approach.
- Two-stage models: Estimate a simple model first and predict errors using SVMs