# Performance Measurement

- Performance: How well does the estimated model perform in predicting the behavior of the data?
- Decide on data set split-up: Split sample method, N-fold cross-validation, single sample method.
- Decide on performance measure: R-squared, Mean Squared Error, Mean Absolute Error, Root Mean Squared Error, etc.

# Split Sample Method

- For large data sets: Large => 1000 observations
- Set aside a portion of the data for testing (1/3 usually) that is not used during the estimation process.
- Calculate the performance of classifier on testing data set.
- Stratification: Same class distribution (good/bad odds) in training and testing set

# Cross-Validation Method

- Small to medium-sized data sets (between 100 and 1000 observations)
- Split date in K folds (K=5 or 10)
- Train the model on K-1 folds and test on the remaining fold
- Repeat K times and compute mean of performance measures: Can also get standard deviation and/o9r confidence intervals
- Leave-one-out cross-validation: K = N, where N is the number of observations
- Stratified cross-validation: Same class distribution (good/bad odds) in training and testing set
- Practical advice: Use 10-fold cross-validation, leave-one-out cross-validation is computationally expensive

# Single Sample Method

- For very small data sets (less than 100 observations)
- Performance = f(training error, model complexity)
- Penalize for complexity of the model
- Akaike Information Criterion (AIC), Bayesian Information Criterion (BIC), Mallows Cp, etc.
- Model complexity can be measured by number of estimated parameters
- Based on statistical learning theory
- Akaike Information Criterion (AIC): AIC = -2 * log-likelihood + 2 * number of parameters
- Bayesian Information Criterion (BIC): BIC = -2 * log-likelihood + log(N) * number of parameters
- Mallows Cp: Cp = (SSE / MSE) - (N - 2 * p), where p is the number of parameters
- Good model has low AIC or BIC or Cp
- Only meaningful to compare between models with same data set

# Performance measures for classification

Let’s now continue and discuss various performance measures for classification.  The most popular ones are:

- the confusion matrix with the classification accuracy, classification error, sensitivity and specificity
- the Receiver Operating Characteristic curve and the area under the curve
- the Cumulative Accuracy Profile curve and Accuracy Ratio
- the lift curve
- the Kolmogorov-Smirnov distance
- the Mahalanobis distance. 

# Confusion Matrix
- Confusion matrix: Table that describes the performance of a classification model, and gives a class-by-class decomposition of the errors.

- True Positive (TP): Correctly classified as positive
- True Negative (TN): Correctly classified as negative
- False Positive (FP): Incorrectly classified as positive
- False Negative (FN): Incorrectly classified as negative

+ Classification Accuracy = (TP + TN) / (TP + TN + FP + FN)
+ Error Rate = (FP + FN) / (TP + TN + FP + FN)
+ Sensitivity = TP / (TP + FN)
+ Specificity = TN / (TN + FP)
+ Precision = TP / (TP + FP)
+ Recall = TP / (TP + FN)
+ F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
+ False Positive Rate = FP / (FP + TN)
+ False Negative Rate = FN / (TP + FN)

All these vary when classification cutoff is varied:
- Predict all customers as good: sensitivity = 100%, specificity = 0%
- Predict all customers as bad: sensitivity = 0%, specificity = 100%

| Name | Good/Bad | Score | Predicted |
|------|----------|-------|-----------|
| John | Bad      | 0.2   | Bad       |
| Mary | Good     | 0.8   | Good      |
| Tom  | Bad      | 0.6   | Good      |
| Emma | Good     | 0.4   | Bad       |
| Bob | Good     | 0.7   | Good      |

- Classification Accuracy = TP + TN / (TP + TN + FP + FN) = 2 / 4 = 50%
- Error Rate = FP + FN / (TP + TN + FP + FN) = 2 / 4 = 50%
- Sensitivity = TP / (TP + FN) = 1 / 2 = 50%
- Specificity = TN / (TN + FP) = 1 / 2 = 50%

# Receiver Operating Characteristic (ROC) Curve

+ Make table with sensitivity and 1-specificity for each possible cutoff
+ ROC curve: Plot of sensitivity vs. 1-specificity for all possible cutoffs
+ In credit scoring, sensitivity is percentage  of goods predicted to be good and 1-specificity is percentage of bads predicted to be good
+ the perfect model has sensitivity = 100% and 1-specificity = 0%
+ ROC curve can be summarized by area underneath the curve (AUC)
+ AUC provides an estimate of probability that the model will rank a randomly chosen good customer higher than a randomly chosen bad customer
+ Diagonal represents classifier found by randomly guessing class and serves as a benchmark
+ AUC > 0.5: Model is better than random guessing

# ROC Curve in Python
    
```python
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
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

logit_model = sm.Logit(Y,X)

result = logit_model.fit()

Ypred = result.predict(X)

fp_rate, tp_rate, thresholds = roc_curve(Y, Ypred)
roc_auc = auc(fp_rate, tp_rate)

plt.title('Receiver Operating Characteristic')
plt.plot(fp_rate, tp_rate, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
```

# Cumulative Accuracy Profile (CAP) Curve
- Sort population from lowest to highest probability of being good
- measure cumulative percentage of goods and bads
- Lorenz curve, Power curve, Captured Event Plot (SAS)
- Accuracy Ratio: Area between CAP curve and random model / Area between perfect model and random model 
- Accuracy Ratio (AR) = (Area below power curve current model - Area below random model) / (Area below perfect model - Area below random model)
- AR sometimes also called as Gini coefficient
- AR = 2 * AUC - 1

# Lift Curve
- Sort population from lowest to highest probability of being good
- Suppose in the top 10% the lowest score, 60% are bads. If in total population 10% are bad, then lift = 60 / 10 = 6
- Lift value is the cumulative percentage of bads per decile, divided by the overall percentage of bads
- Using no model or random sorting, lift would always be 1.
- Lift can also be expressed in a non-cumulative way.

# Kolmogorov-Smirnov Distance

- Separation measure
- Distance between cumulative distribution of goods and bads
- KS = max |Fg - Fb|, where Fg is cumulative distribution of goods and Fb is cumulative distribution of bads
- KS distance metric max vertical distance between ROC curve and diagonal

# Mahalanobis Distance

- Measure the Mahalanobis distance between the two mean scores
- Better than Euclidean distance because it takes into account the covariance between the variables
- Closely related is the divergence measure
- Mahalanobis distance is defined as the difference between the mean scores of the two groups divided by the standard deviation of the scores

# Performance measures for regression

- Person correlation coefficient: Correlation between predicted and actual values
- Mean Squared Error: Average of squared differences between predicted and actual values
- Mean Absolute Deviation: Average of absolute differences between predicted and actual values
- Pearson correlation varies between -1 and 1