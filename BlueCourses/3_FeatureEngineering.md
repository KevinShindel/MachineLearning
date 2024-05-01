
# Feature Engineering Definition

- Transforming dataset variables to help analytical models achieve better performance it terms 
  of: predictive power and interpretability.
- Simple example: 
  - Date of birth -> Age
- Manual feature engineering versus automated feature engineering.
- The best way to improve the performance of an analytical model is by designing smart features.


## RFM features
- Recency, Frequency, Monetary Value
- Recency: How recently a customer has made a purchase
- Frequency: How often a customer makes a purchase
- Monetary Value: How much money a customer spends on purchases
- Can be operationalized in many ways, e.g.:
  - Recency: Number of days since last purchase
  - Frequency: Number of purchases in the last year
  - Monetary Value: Total spend in the last year
- Useful for behavioral credit scoring
- Very popular in marketing and fraud detection

## Trend Features
- Trends summarize historical evolution
- Examples:
  - Average purchase value in the last 3 months
  - Average number of purchases in the last 6 months
  - Average time between purchases in the last year
- Can be useful for size variables (e.g. number of employees) or time series data
- Beware of denominators equal to zero
- Can put higher weight on recent values
- Extension: time series analysis

## Logarithmic Transformation

- Transform: x-> f(x) = log(x)
- Applied to size variables: loan amount, income, etc.
- Only for positive values

### Power Transformation

- Simple power transform: x-> f(x) = x^p
- p > 1: makes distribution more right-skewed
- 0 < p < 1: makes distribution more left-skewed
- set p by: 
  - Experimentation
  - Visual inspection
  - Performance metrics (AUC, profit, etc.)

### Box Cox Transformation

- x -> f(x) = (x^p - 1) / p
- Only for positive x: shift x by a constant
- Does not guarantee normality
- Set p by:
  - Experimentation
  - Visual inspection
  - Performance metrics (AUC, profit, etc.)

### Box Cox Transformation in Python

```python
import pandas as pd
import math

def boxcox(x, lam):
   if lam!=0: return (x**lam-1)/lam
   if lam==0: return math.log(x)

hmeq = pd.read_csv('c:/temp/hmeq.csv')

hmeq["BOXCOXLOAN"] = boxcox(hmeq['LOAN'],0.5)

hmeq.hist(column="LOAN", bins=20, grid=False)
hmeq.hist(column="BOXCOXLOAN", bins=20, grid=False)
```

### Yeo Johnson Transformation

- x -> f(x) = ((x+1)^p - 1) / p
- For positive and negative x: shift x by a constant
- Set p by:
  - Experimentation
  - Visual inspection
  - Performance metrics (AUC, profit, etc.)

### Yeo Johnson Transformation in Python

```python
import pandas as pd
from sklearn.preprocessing import PowerTransformer

hmeq = pd.read_csv('c:/temp/hmeq.csv')

pt = PowerTransformer(method='yeo-johnson')
hmeq['YEOJOHNSONLOAN'] = pt.fit_transform(hmeq[['LOAN']])
hmeq.hist(column="LOAN", bins=20, grid=False)
hmeq.hist(column="YEOJOHNSONLOAN", bins=20, grid=False)
```

### Performance Optimization

- Split data into training and testing sets
- Specify a range for p, e.g p = [-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3]
- For each p:
  - Apply transformation to training set
  - Train model
  - Apply transformation to testing set
  - Evaluate model
- Build a model on training set and measure AUC on validation set
- Pick the p that maximizes AUC
- Build final model on combined training and validation sets and measure AUC on test set

### Performance Optimization in Python

```python
# Load packages
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import yeojohnson
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# Suppress SettingWithCopyWarning from Pandas
pd.options.mode.chained_assignment = None
# Suppress FutureWarnings from Sklearn
import warnings

warnings.filterwarnings("ignore")

# Read data
hmeq = pd.read_csv("hmeq.csv")

# Preprocess
# Missing values
# Remove observation if it has more than 3 variables missing
hmeq.dropna(axis=0, thresh=hmeq.shape[1] - 3, inplace=True)

# Remove DEBTINC
hmeq.drop(['DEBTINC'], axis=1, inplace=True)

# Replace remaining missing values with variable mean
hmeq.MORTDUE.fillna(hmeq.MORTDUE.mean(skipna=True), inplace=True)
hmeq.VALUE.fillna(hmeq.VALUE.mean(skipna=True), inplace=True)
hmeq.YOJ.fillna(hmeq.YOJ.mean(skipna=True), inplace=True)
hmeq.CLAGE.fillna(hmeq.CLAGE.mean(skipna=True), inplace=True)
hmeq.BAD = hmeq.BAD.astype('bool')

hmeq.DEROG.fillna(0, inplace=True)
hmeq.DELINQ.fillna(0, inplace=True)
hmeq.NINQ.fillna(0, inplace=True)

hmeq.REASON.fillna('DebtCon', inplace=True)
hmeq.JOB.fillna('Other', inplace=True)

# Make factors
hmeq.REASON = hmeq.REASON.astype('category')
hmeq.JOB = hmeq.JOB.astype('category')

# Model

# Split in train and test
train, test = train_test_split(hmeq, train_size=(2 / 3), random_state=44)

# Make a backup of train for later use
trainBU = train.copy()

# Make a split to divide into train and validation sets
includeTraining = train.sample(frac=(2 / 3), replace=False, random_state=44).index

# Initialize train to the clean copy
train = trainBU.copy()

# Specify values for lambda
lambd = np.arange(-3, 3.1, 0.5)
# Create output vector
perf = [np.nan for x in lambd]

# Grid search for the best lambda using the train and validation set.
for idx, i in enumerate(lambd):
  # Transform VALUE with yeo-johnson using lambda[i]
  train.VALUE = yeojohnson(train.VALUE, lmbda=i)

  # Split train into a training and validation sets using the split includeTraining
  training = train.loc[includeTraining]
  validation = train.loc[train.index.difference(includeTraining)]

  # Make logistic regression with training and evaluate it on validattion

  endog = pd.get_dummies(training.BAD)
  endog.columns = ['False', 'True']
  endog = endog[['True', 'False']].values

  exog = training.drop('BAD', axis=1)
  exog = pd.get_dummies(exog, drop_first=True)

  glm_binom = sm.GLM(endog, exog, family=sm.families.Binomial()).fit()
  # print(glm_binom.summary())

  val = validation.drop('BAD', axis=1)
  val = pd.get_dummies(val, drop_first=True)

  logPrediction = glm_binom.predict(val)
  logPrediction

  # Store the performance
  perf[idx] = roc_auc_score(validation.BAD, logPrediction)
  # Retrieve the train data with VALUE column unchanged for the next iteration
  train = trainBU.copy()

# Find the best lambda
bestLambda = lambd[np.argmax(perf)]
bestLambda

endog = pd.get_dummies(train.BAD)
# Change order of True/False column for BAD.
# GLM expects True column to appear before False.
endog.columns = ['False', 'True']
endog = endog[['True', 'False']].values

exog = train.drop('BAD', axis=1)
exog = pd.get_dummies(exog, drop_first=True)

benchmarkLogModel = sm.GLM(endog, exog, family=sm.families.Binomial()).fit()

tst = test.drop('BAD', axis=1)
tst = pd.get_dummies(tst, drop_first=True)

logPrediction = benchmarkLogModel.predict(tst)

print(roc_auc_score(test.BAD, logPrediction))

# Transform VALUE with yeo-johnson using the best lambda
train.VALUE = yeojohnson(train.VALUE, bestLambda)
test.VALUE = yeojohnson(test.VALUE, bestLambda)

# Build logistic regression on train set and evlauate it in test set
endog = pd.get_dummies(train.BAD)
endog.columns = ['False', 'True']
endog = endog[['True', 'False']].values

exog = train.drop('BAD', axis=1)
exog = pd.get_dummies(exog, drop_first=True)

finalLogModel = sm.GLM(endog, exog, family=sm.families.Binomial()).fit()

tst = test.drop('BAD', axis=1)
tst = pd.get_dummies(tst, drop_first=True)

logPrediction = finalLogModel.predict(tst)

print(roc_auc_score(test.BAD, logPrediction))
```

### Principal Component Analysis

- Technique to reduce the dimensionality of the data
- Principal components can be calculated by making use of the eigenvectors of the covariance matrix
- New variables are linear combinations of the original variables
- Principal components are orthogonal to each other
- Principal components are uncorrelated
- Pro: 
  - Reduces the number of variables
  - Removes multi-collinearity
  - Improves model performance
- Cons:
  - Loss of interpretability
  - Loss of information

### t-SNE

- Technique to reduce the dimensionality of the data
- t-Distributed Stochastic Neighbor Embedding
- t-SNE - is non-linear dimensionality reduction technique
- t-SNE stands for:
  - Comparable to PCA
  - t-SNE seeks to preserver local similarities

> t-SNE works in 2 steps:
> 1. Probability distribution representing similarity measure over pairs of high-dimensional data points is constructed.
> 2. Similar probability distribution over data points in low-dimensional map is constructed

1. Measure similarities between data points in high dimensional space
2. Measure similarities between data points in low dimensional space

> Most implementations only allow for 2 or 3 dimensions
> t-SNE learns non-parametric mapping:
>   - No explicit function to map from high to low dimensions
>   - No possible to embed test points in existing map
>   - t-SNE less suitable as dimensionality reduction technique in predictive setup
>   - Extensions exist that learn multivariate regressor to predict map location from input data or construct regressor that minimizes regressor tha\t minimizes t-SNE loss directly
> Can provide your own pairwise similarity matrix and do KL-minimization instead using built-in conditional probability based similarity measure
>   - Diagonal elements should be 0 and should be normalized to sum to 1
>   - Distance matrix can also be used: similarity = 1 / (1 + distance)
>   - Avoids having to tune the perplexity parameter (decide on similarity of points)
> As t-SNE uses a gradient descent algorithm based approach, remarks regarding learning rates and initialization of mapped points apply
>   - E.g initialization sometimes done using PCA
>   - Defaults usually work well
> Most important hyperparameter is perplexity:
>   - Knob that sets number of effective nearest neighbors (similar to k in k-NN)
>   - Perplexity values depends on density of data
>   - Denser dataset requires higher perplexity
>   - Typical values range from 5 to 50
> Impact of perplexity: neighborhood effectively considered
> Different perplexity values can lead to different results:
>   - Size of clusters has no meaning
>   - Neither does distance between clusters

# TODO: Read more about t-SNE at : [t-SNE](https://distill.pub/2016/misread-tsne/)
# TODO: Implementations of t-SNE in Python [t-SNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)