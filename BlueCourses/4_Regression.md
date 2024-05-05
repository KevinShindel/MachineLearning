# Regression

## Linear Regression

- It Is a predictable technique used to model the relationship between two variables by fitting a linear equation to observed data.
- Best for continuous target variable.
- Examples:
- - Customer Lifetime Value (CLV)
- - Sales Forecasting
- - Loss Given Default (LGD)
- - Sales
- - Blood Pressure
- - Number of votes
- - Lifetime of a battery
- - Air pollution

| ID | Age | Recency | Frequency | Monetary | CLV  |
|----|-----|---------|-----------|----------|------|
| C1 | 20  | 26      | 4.2       | 126      | 3817 |
| C2 | 58  | 37      | 2.1       | 59       | 4310 |
| C3 | 33  | 12      | 3.3       | 231      | 7623 |
| C4 | 41  | 20      | 1.3       | 78       | 1014 |
| C5 | 29  | 33      | 2.9       | 189      | 5481 |

CVL - is target variable

- OLS regression: Ordinary Least Squares will always yield a solution.
- Check the validity of model assumptions
- If the model is correct, then residuals resemble true errors
- Frequently occurring deviations from linear regression assumptions:
- - Error terms not normally distributed: Transform target or applied Generalized Linear Model
- - Variance of error terms is not constant: Transform target or use Weighted Least Squares regression
- - Error terms correlated: Noise can be modeled as ARMA process or apply spurious or dynamic regression
- - Non-linear relationship between target and features: Transform target and/or predictor variables
- Linearity assumption restricted to coefficients, not to predictors
- Obtain non-linear associations by transforming predictors
- Following relation still considered linear regression model
- Also, other transformations and iterations between predictors may be included
- Graphical tools are helpful to decide whether interaction terms, power terms or transformations might be necessary

## High Dimensional Data

- A popular measure for quality of estimator is the Mean Squared Error (MSE)
- MSE includes both squared bias and variance of estimator
- Both bias and variance of estimator should be minimized
- LS estimator desired property of being unbiased
- When there are many variables, variance may come very high
- LS regression needs to be adapted to cope with high-dimensional data
- Lower variance at cost of making estimator biased
- Technologies have changed and typically large number of variables are measured
- In genomics, imaging or chemometrics, data are often ultra-high dimensional: dimensions exceed sample size
- Using feature engineering, many extra features added to dataset leading to an extremely large number of features
- Traditional regression techniques intended for low-dimensional setting and not appropriate for high-dimensional data:
- - If sample size is much larger than dimension: LS estimates typically have low variance
- - If sample size is no much larger than dimension: LS estimates have high variance
- - If sample size is smaller than dimension: LS estimates are not uniquely defined (variance is infinite, and method cannot be used)
- Popular strategies for subset selection:
- - Forward stepwise selection
- - Backward stepwise selection
- - Hybrid stepwise selection

## Ridge Regression

- Instead of subset selection, apply shrinkage or regularization to reduce the size of coefficient estimates 
- Remember the definition of the Last Squares estimator
- Ridge Regression (Hoerl and Kennard, 1970) is a shrinkage method

## Lasso Regression

- Assumption of sparsity in high-dimensional data: small number of features contribute to the model
- Ridge regression cannot force estimates to become exactly zero
- Alternative penalty methods: SCAD penalty, minimax concave penalty, adaptive Lasso, Dantzig selector
- Cross-validation to select tuning params
- - Define grid of values for tuning parameter
- - Compute cross-validation error for each value
- - Select value with the smallest cross-validation error
- - Refit model using all available observations and selected value of tuning parameter

- Computational efficiency
- - Forward stage-wise regression
- - Least Angle Regression (LARS)

- Goal is to increase a little bit of bias while reducing variance a lot

## Elastic Net

- Drawback of LASSO regression:
- - If n < p, LASSO selects at most n variables before it saturates
- - LASSO fails to do grouped selection

## Principal Component Regression (PCR)

- PCR - is a technique to cope with many, possibly correlated, predictors
- First, PCA is applied to predictors
- The Number of principal components is selected using cross-validation
- Instead of applying the regression model on the original predictors, components are used as predictors
- PCR can be seen as a special case of linear regression where the dimension reduction leads to a constraint on estimated coefficients
- PCR depends on scaling of variables

## Partial Least Squares Regression (PLS)

- PCR involves identifying linear combinations that best represent predictors but no guarantee that these directions will be the best ones for predicting response
- Alternative is PLS regression, where both predictors and response are taken into account to create the components
- Similar as with PCR, number of components is selected using cross-validation 
- Standardize predictors and response
- Popular algorithms are NIPALS, SIMPLS, Bidiag to compute the components and apply the LS regression on them
- Variance aspects often dominate and then PLS behaves closer to PCR, although in many examples PLS needs less components than PCR

## Generalized Linear Models

- Assumptions of a linear regression model often violated:
- - Normally distributed errors
- - Constant variability
- - Mean is a linear function of variables
- Generalized Linear Models (GLM) extend a linear regression model and relax assumptions
- Parameters are typically estimated by maximum likelihood, but in general there is no analytical solution and one has to rely on numerical solutions using iterative procedures
- GLMs consist of three important parts:
- - Random component (describes distribution of target variable)
- - Systematic component (specified by linear combination of predictors to obtain linear predictor)
- - Link function (links random and systematic components and describes how the value for target variable is related to linear predictor)
- LS regression assumes normally distributed errors, but GLMs allow every distribution belonging to the exponential family as error distribution
- Popular distributions belong to the exponential family are:
- - Count data: Poisson distribution (number of claims, number of defaults, number of failures)
- - Positive continuous data: Gamma distribution (claim sizes, loss amounts, hospital stays)
- - Proportions: Binomial distribution (probability of default, fraud, etc.)
- Even more distributions can be obtained by taking transformations of these distributions

## Generalized Additive Models (GAMs)

- Generalized Additive Models (GAMs) are an extension of GLMs
- Goal of GAMs is to allow non-linear relationships between predictors and target variable
- GAMs still contain a sum of features but extended by GLMs by replacing linear functions with smooth functions
- Popular approach for functions is to use splines
- Definition of a spline: piecewise polynomial function that is smooth at the knots
- Because of the smoothness, splines are often used in regression models