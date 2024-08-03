## Model based collaborative filtering (MBCF)
- Use of ML to extract patterns from data
- Popular techniques: clustering, regression, association rules, Bayesian networks, network community mining

## MBCF: Clustering
- Build clusters of users or items using k-means, SOMs, etc., to their rating vectors
- Identify nearest cluster for active user
- Missing ratings predicted using cluster averages.
- All users within cluster receive same recommendations
- When compared to k-nearest neighbor, clustering is more scalable as clusters can be computed 
  off-line: k-nearest neighbor methods shown to be more accurate.
- Note: can also cluster items

|          | Item 1 | Item 2 | Item 3 | Item 4 | Item 5 | Item 6 |
|----------|--------|--------|--------|--------|--------|--------|
| **Bart** | 8      | 7      |        | 6      | 1      |        |
| _Laura_  |        | 2      | 8      |        | 4      | 7      | 
| _Sophie_ |        | 2      |        |        |        | 9      |
| _Victor_ |        | 1      |        |        | 3      |        | 
| **Tim**  | 9      | 7      |        | 5      | 2      |        |

- Two clusters: {Bart, Tim}, {Laura, Sophie, Victor}
- Recommend Item 3 to Sophie, followed by Item 5 and Item 6

## MBCF: Regression
- Linear vs non-linear regression
- Fit regression models, which each time relate item to another item.

## MBCF: Regression in Python

```python
import statsmodels.api as sm
import numpy as np
# Prepare data

users = ['Bart', 'Michael', 'Tim', 'Sophie', 'Victor', 'Laura', 'Maria']
items = ['Rambo II', 'Rocky IV', 'Nothing Hill', 'Pulp Fiction', 'Jurassic Park', 'Moulin Rouge', 'The Big Lebowski']

ratings = np.array([
    [4,      5,      1,      np.nan,      np.nan,      0,      np.nan],
    [1,      1,      5,      np.nan,      np.nan,      np.nan, np.nan],
    [5,      4,      1,      0,           5,           0,      np.nan],
    [np.nan, np.nan, np.nan, 1,           5,           np.nan, np.nan],
    [0,      np.nan, np.nan, 5,           1,           np.nan, np.nan],
    [np.nan, np.nan, 5,      np.nan,      np.nan,      5,      np.nan],
    [5,      3,      2,      np.nan,      4,           1,      np.nan],
])
# Model
# We're going to fit m x (m-1) regression models, each time using the ratings of expert i to 
# predict the ratings for j, model[i][j]. Here, the experts are item-based, but user-based can be 
done as well

# Fit m (m-1) regression models

models = {}

for i in range(len(items)):
    models[i] = {}
    for j in range(len(items)):
        models[i][j] = None
        if i == j: continue
        X = ratings[:,i]
        X = sm.add_constant(X)
        Y = ratings[:,j]
        try:
            model   = sm.OLS(Y, X, missing='drop')
            results = model.fit()
        except ValueError:
            # This model could not be fit
            continue
        models[i][j] = (model, results)
# Kick out the models with low R-squared values
for i in models:
    for j, val in models[i].items():
        if val is None: continue
        if np.isnan(val[1].rsquared) or val[1].rsquared < .75: models[i][j] = None
        else: print(i, '->', j, ':', val[1].rsquared)
        
# The warning is due to some of the R2 being nan

# Next, we need to give weights to the remaining models, summing to 1. The paper goes into some 
# depth on how this can be done using the model errors. However, we're just going to give each 
# the same weighht here

weight = 0
for i in models:
    for j, val in models[i].items():
        if val is None: continue
        weight += 1
weight = 1/weight
       
# Finally, we can get the recommendations for a user. To rate an item for a user, we need to 
# query all the expert models, iterating over the rated items

user_to_recommend_for = 0
def rating(user, item):
    nom, denom = 0, 0
    rated_items = np.argwhere(~np.isnan(ratings[user, :]))[:, 0]
    for rated_item in rated_items:
        if models[rated_item][item] is None: continue
        X = ratings[:,rated_item]
        X = sm.add_constant(X)
        prediction = models[rated_item][item][1].predict(X)
        nom   += weight * prediction[user]
        denom += weight
    if denom == 0: return 0
    return nom / denom

for itm_idx in np.argwhere(np.isnan(ratings[user_to_recommend_for, :]))[:, 0]:
    prediction = rating(user_to_recommend_for, itm_idx)
    print('Item index {} ({}) rating = {}'.format(itm_idx, items[itm_idx], prediction))
```

## MBCF: Association Rules
- Examples:
- - If customer has car loan and car insurance, then customer has checking account in 80% of cases.
- - If customer buys spaghetti, then customer buys red whine in 70% of cases.
- Support of item set is percentage of total transactions that contains item set.
- Frequent item set: item set which support is higher than pre-specified threshold.
- Support = number of transactions supporting / number of transactions.
- Association rule has confidence if 100% of transactions that contain item A also contain item B

| Transaction | Items                           |
|-------------|---------------------------------|
| 1           | beer, milk, diapers, baby food  |
| 2           | beer, milk, diapers             |
| 3           | beer, milk                      |
| 4           | cigarettes, diapers, baby food  |
| 5           | tomatoes, water, apples, limon  | 
| 6           | water, beer, milk, diapers      |
| 7           | water, beer, milk               |
| 8           | spaghetti, red wine, beer, milk |
| 9           | spaghetti, red wine, beer       |
| 10          | spaghetti, red wine             |

- Item set {baby food, diapers, beer } has support 2/10 = 20%
- Association rule: {baby food, diapers} -> {beer} has confidence 100%
- Use association rules for recommendations as follows:
- - look at what items user already purchased.
- - recommend item according to association rules with high confidence.

## MBCF: Bayesian Networks
- A Bayesian network represents a joint probability distribution over a set of categorical, stochastic variables.
- - qualitative part specifying the conditional dependencies between variables.
- - quantitative part specifying conditional probabilities of variables.
- - probabilistic white-box model.
- Ideal for combining domain knowledge with data patterns.
- Every item is a node.
- Focus on binary rating (like/dislike, purchase/no purchase).
- Learn Bayesian network from historical data:
- - correlation-based methods.
- - Markow Chain Monte Carlo (MCMC) methods.
- Use probabilistic inference for prediction.
- Consider customer who likes items A and E and does not like items B and D.
- What is probability that customer likes item C?
- - Similarity to item A and E = 0.8.
- - Hence probability that customer likes item C = 0.8.
- Winner-takes-all rule says customer likes item C.


## MBCF: Network Community Mining
- Use of social network data to improve recommendations.
- - Social network data: who is connected to whom.
- - Social network data can be explicit or implicit.
- - Explicit: Facebook, LinkedIn. 
- - Implicit: who sends email to whom.

## MBCF: Evaluation
- Analytical models can be built off-line
- At run-time, learned analytical model used to make predictions.
- Models updated periodically.
- Model building and updating can be computationally expensive.
