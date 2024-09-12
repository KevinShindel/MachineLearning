# Variable selection
- Variables = Attributes = Characteristics = Inputs = Features
- - If n-variables, 2n-1 possible variable subsets
- Heuristic search needed to find the best subset
- Good variable subsets contain variables highly correlated with the target variable and uncorrelated with each other
- Benefits: compact model, reduce estimation time, improve prediction accuracy, improve interpretability
- Curse of dimensionality: as the number of variables increases, the number of possible variable subsets increases exponentially
- Difference between redundant variable and relevant variable: correlation implies redundancy, 
  variable can be redundant but relevant, correlation between "age" and "time on "books""
- Variable selection methods: Filter, Wrapper, Embedded
- Filter: select variables before model building, based on statistical measures
- Wrapper: select variables during model building, based on model performance
- Embedded: select variables during model building, based on model performance, but more efficient than wrapper methods
- Variable selection methodology:
- - Step 1: filter -> quick screening of variables, independent of analytical technique
- - Step 2: Forward/Backward/Stepwise Regression -> wrapper methods, dependent on analytical 
    technique, based upon p-value of regression model
- - Step 3: AUC-based variable selection -> iterative procedure based on AUC

# Filter methods

|                       | Continuous target   | Categorical target                        |
|-----------------------|---------------------|-------------------------------------------|
| Continuous variables  | Pearson correlation | Fisher score                              |
| Categorical variables | ANOVA, Fisher score | Chi-square, Cramer's V, Information value |

- Pearson correlation: measures the linear relationship between two continuous variables
- Fisher score: measures the difference between the means of two groups divided by the sum of the variances of the two groups
- ANOVA: Analysis of Variance, measures the difference between the means of two or more groups
- Chi-square: measures the difference between the observed and expected frequencies of two or more groups
- Cramer's V: measures the strength of association between two categorical variables
- Information value: measures the predictive power of a categorical variable
- Compute Pearson correlation between each continuous variable and the target variable
- Always between -1 and 1
- Keep only variables for which p > 0.50 e.g. top 10% of variables (where p is the absolute 
  value of the correlation coefficient)
- Only linear relationships
- Compute Fisher score between each continuous variable and the target variable
- Compute ANOVA between each categorical variable and the target variable
- IV = sum((% of target = 1 - % of target = 0) * log(% of target = 1 / % of target = 0))
- Gain measures decrease in impurity when splitting on variable
- Cramer's V, Information Value and Gain give similar results.

|                  | IV     | Gain   | Cramer's V |
|------------------|--------|--------|------------|
| Checking account | 0.66   | 0.0947 | 0.35       |
| Credit history   | 0.2932 | 0.04   | 0.24       |
| Purpose          | 0.27   | 0.03   | 0.22       |
| Housing          | 0.25   | 0.03   | 0.21       |
| Age              | 0.24   | 0.03   | 0.20       |

# Cramer's V in Python
```python
import pandas as pd
import numpy as np
import scipy.stats as ss

hmeq = pd.read_csv('c:/temp/hmeq.csv')
df1 = hmeq[['JOB','BAD']]
confusion_matrix = pd.crosstab(hmeq['JOB'], hmeq['BAD'])
chi2 = ss.chi2_contingency(confusion_matrix)[0]
CramerV = np.sqrt(chi2/df1.size)
```

# Information Value in Python
```python
import pandas as pd
import numpy as np

hmeq = pd.read_csv('c:/temp/hmeq.csv')

def calc_iv(df, feature, target):
    # https://www.kaggle.com/puremath86/iv-woe-starter-for-python
    lst = []
    df[feature] = df[feature].fillna('NULL')
    for i in range(df[feature].nunique()):
        val = list(df[feature].unique())[i]
        lst.append([feature, val, df[df[feature] == val].count()[feature], df[(df[feature] == val) & (df[target] == 1)].count()[feature]])
    data = pd.DataFrame(lst, columns=['Variable', 'Value', 'All', 'Bad'])
    data = data[data['Bad'] > 0]
    data['Share'] = data['All'] / data['All'].sum()
    data['Bad Rate'] = data['Bad'] / data['All']
    data['Distribution Good'] = (data['All'] - data['Bad']) / (data['All'].sum() - data['Bad'].sum())
    data['Distribution Bad'] = data['Bad'] / data['Bad'].sum()
    data['WoE'] = np.log(data['Distribution Good'] / data['Distribution Bad'])
    data['IV'] = (data['WoE'] * (data['Distribution Good'] - data['Distribution Bad'])).sum()
    data = data.sort_values(by=['Variable', 'Value'], ascending=[True, True])
    return data['IV'].values[0]
    
calc_iv(hmeq, 'JOB', 'BAD')
```

# Forward/Backward/Stepwise Regression
- Hypothesis test (Wald test) for each variable in the model to determine if it should be included.
- Linear regression model: y = b0 + b1x1 + b2x2 + ... + bnxn
- Logistic regression model: y = 1 / (1 + exp(-z)), z = b0 + b1x1 + b2x2 + ... + bnxn
- Low (high) p-value represents  significant variable
- Rule of thumb: 
- - p-value < 0.01 - is highly significant
- - p-value < 0.05 - is significant
- - p-value < 0.1 - is weakly significant
- Can be used in different ways: forward, backward, stepwise
- Forward - starts from empty model and always adds variables based on low p-values
- Backward - starts from full model and always removes variables based on high p-values
- Stepwise - starts from empty model and adds or removes variables based on p-values

### TODO: need refactoring (all three methods are the same)

# Forward selection in Python
```python
import pandas as pd
import statsmodels.api as sm

hmeq = pd.read_csv('c:/temp/hmeq.csv')
X = hmeq[['LOAN','MORTDUE','VALUE','YOJ','DEROG','DELINQ','CLAGE','NINQ','CLNO','DEBTINC']]
y = hmeq['BAD']
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
model.summary()
```

# Backward selection in Python
```python
import pandas as pd
import statsmodels.api as sm

hmeq = pd.read_csv('c:/temp/hmeq.csv')
X = hmeq[['LOAN','MORTDUE','VALUE','YOJ','DEROG','DELINQ','CLAGE','NINQ','CLNO','DEBTINC']]
y = hmeq['BAD']
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
model.summary()
```

# Stepwise selection in Python
```python
import pandas as pd
import statsmodels.api as sm

hmeq = pd.read_csv('c:/temp/hmeq.csv')
X = hmeq[['LOAN','MORTDUE','VALUE','YOJ','DEROG','DELINQ','CLAGE','NINQ','CLNO','DEBTINC']]
y = hmeq['BAD']
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
model.summary()
```


# BART: Backward Regression Trimming
- BART is a backward regression method that uses a trimming technique to remove variables
- BART methodology:
- - Step 1: Split the data in training, validation and test set
- - Step 2: Estimate a logistic regression model with all n-variables on the training set and 
    measure the performance (AUC, profit, etc.) on the validation set
- - Step 3: Estimate n-logistic regression models with n-1 variables on the training set and 
    proceed with best one in terms of performance on the validation test.
- - Step 4: Estimate all n-1 logistic regression models with n-2 variables on the training set 
    and proceed with the best one in terms of performance on the validation set. Continue doing 
    this until no variables left in model.
- - Step 5: Choose the best model based on the validation set performance.
- - Step 6: Estimate a model on the combined training and validation set and measure the 
    performance on the independent test set.

# Criteria for variable selection
- Interpretability: the model should be easy to understand
- Computation cost of variable selection: the method should be computationally efficient
- Legal concerns: some variables are not allowed to be used in the model