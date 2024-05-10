
# Classification problem:
- Given observation with predictor determine class from set.
- Classes known beforehand
- Binary versus multiclass

# Linear Regression
- Linear regression: estimates linear relationship between target variable and predictor variables (features)
- Ordinary Least-Squares (OLS) - is a method for estimating the unknown parameters in a linear regression model. OLS chooses the parameters of a linear function of a set of explanatory variables by minimizing the sum of the squares of the differences between the target dependent variable and those predicted by the linear function.
- Linear regression with transformation such that output is always between 0 and 1, and can be interpreted as probability
- After model has been estimated using historical data, can use it to score or assign probabilities to new observations
- Doubling Amount:
- - Amount of change required for doubling primary outcome odds

## Logistic Regression in Python

```python
import pandas as pd
import numpy as np
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