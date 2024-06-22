## Motivation
- Matrix factorization = Matrix decomposition
- Factorize matrix into a product of other matrices
- Simplest form is 2 matrices: user feature matrix and item feature matrix
- Features = concepts = latent factors

## UV Decomposition
- Full rating matrix is very high dimensional in terms of users and items.
- Relevance often dependents on small number of latent factors.
- Matrix factorization learns latent factors underlying user-item relevance.
- RMSE - often used to quantify approximation:
- - Sum of squared differences between non-blank entries in the original matrix and the approximated matrix.
- - Divide by number of non-blank entries and take square root.
- Usually start with randomly chosen U and V and adjust them to minimize RMSE.
- Optimization process with multiple local minima 1 global minimum: choose multiple starting 
  value for U and V to et close to global minimum.
- UV decomposition algorithm:
- - Preprocessing step: for each non-blank rating, subtract item and user avg.
- - Initialize step: choose starting U and V with random values.
- - Optimization step (stochastic): gradient descent to minimize RMSE.
- - Convergence step: stop when RMSE converges.
- UV decomposition is a form of collaborative filtering.
- Latent dimensions:
- - Number of latent dimensions is a hyperparameter.
- - Each row in U represents condensed information about 'type if item' user likes.
- - Each column in V represents condensed information about 'type of user' that likes item.
- Making predictions using matrix factorization becomes simple.
- Predicted interest of Seppe in Rambo II

|       | i_lat_1 | i_lat_2 | i_lat_3 |
|-------|---------|---------|---------|
| Seppe | 1.15    | 0.33    | 0.5     |

|         | Rambo II |
|---------|----------|
| u_lat_1 | 2        | 
| u_lat_2 | 1.4      |
| u_lat_3 | 0.01     |

- Predicted interest = (1.15 * 2) + (0.33 * 1.4) + (0.5 * 0.01) = 2.8

## Non-Negative UV Decomposition
- R = U * V (decomposed matrix)
- Subject to R, U, V >= 0
- Non-negativity leads to easier interpretation of latent factors.
- Can be done using (adapted) gradient descent.
- 

## Non-Negative UV Decomposition in Python

```python
import pandas as pd
import numpy as np

# Install surprise with
# conda install -c conda-forge scikit-surprise

from surprise.reader import Reader
from surprise import Dataset
from surprise import accuracy

from surprise.prediction_algorithms.matrix_factorization import NMF
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
# surprise works with data frames with three columns: user ids, item ids and the rating, so let's 
# make on

df = pd.DataFrame([
    [u, i, ratings[u,i]]
    for u in range(len(users)) for i in range(len(items))
], columns=['user', 'item', 'rating'])
df.head()

# surprise doesn't like missing values, so we impute with 2.5 (you can use other strategies as 
# well, as discussed in the course)

df.replace(to_replace=np.nan, value=2.5, inplace=True)
reader = Reader(rating_scale=(0,5))
data   = Dataset.load_from_df(df, reader)

# NMF
# Train on the complete data set. surprise also has support for cross-validation and train/test 
# splitting, see docs at https://surprise.readthedocs.io/en/stable/getting_started.html

trainset = data.build_full_trainset()

# Build an algorithm, and train it.
recommender = NMF(n_factors=2, biased=True)
recommender.fit(trainset)

# Make some predictions

user_to_recommend_for = 0

for itm_idx in np.argwhere(np.isnan(ratings[user_to_recommend_for, :]))[:, 0]:
    prediction = recommender.predict(user_to_recommend_for, itm_idx)
    print('Item index {} ({}) rating = {}'.format(itm_idx, items[itm_idx], prediction))

for user in users:
    u = users.index(user)
    print('User {} bias: {}'.format(user, recommender.bi[u]))

for item in items:
    i = items.index(item)
    print('Movie {} bias: {}'.format(item, recommender.bi[i]))

# User factors (non-negative)
print(recommender.pu)

# Item factors (non-negative)
print(recommender.qi)
```

## Singular Value Decomposition
- 

## Singular Value Decomposition in Python
- 

## Tensor Decomposition
- 

## Closing Thoughts
- 
