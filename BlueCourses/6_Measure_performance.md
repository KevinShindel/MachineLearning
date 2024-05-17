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

# Receiver Operating Characteristic (ROC) Curve

# Cumulative Accuracy Profile (CAP) Curve

# Lift Curve

# Kolmogorov-Smirnov Distance

# Mahalanobis Distance

# Performance measures for regression
