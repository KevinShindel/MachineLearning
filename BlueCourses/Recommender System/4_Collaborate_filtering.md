## Definition
- Use rating matrix to make a prediction.
- K-nearest neighbor based collaborate filtering:
- - User-User collaborate filtering
- - Item-Item collaborate filtering
- Model based collaborate filtering:
- - ML methods for recommendation.
- Key assumption:
- - Users who had similar testes in the past, will have similar tastes in the future.

## User-User Collaborative Filtering: Basic Idea
- Intuition: users who are interested in similar items in the past, will probably be interested 
  in similar items in the future.
- Pearson correlation coefficient is used to measure similarity between users, its range is [-1, 
  1], where -1 means negative correlation, 1 means positive correlation, 0 means no correlation.
- Cosine measure: 
- - between 0 (low similarity) and 1 (high similarity).
- - users are vectors in the item space.
- - only consider common items.
- - differences in avg rating behavior are not considered.
- Adjusted cosine similarity:
- - Takes avg user rating into account.
- - values range from -1 to 1.
- Jaccard similarity:
- - Frequently used in binary data.
- - range from 0 to 1 (0 means no similarity, 1 means perfect similarity).
- Problem with similarity measures:
- - computation cost.
- - dimensionality reduction.

## User-User Collaborative Filtering: Similarity Measures
- ?

## User-User Collaborative Filtering: Prediction
- How the similar users are selected?
- Take into account rating bias.
- Note: analysis of MovieLens data says: "in most real-world situations, a neighborhood size of 
  20 to 50 neighbors seems reasonable".

## User-User Collaborative Filtering in Python

```python
# # Imports and Toy Data
import numpy as np

from scipy.stats import pearsonr
from scipy.spatial.distance import cosine
from scipy.spatial.distance import jaccard

from sklearn.metrics import pairwise_distances


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

# Pearson Correlation (example from the slide)
pearsonr([7,2,9,4,2], [8,4,7,6,4])

# Cosine Similarity (example from the slide)
1 - cosine([7,2,9,4,2], [8,4,7,6,4])

# 1 - as scipy return the distance, not similarity 

# Adjusted Cosine Similarity (example from the slide)
a = [6,7,0,2,0,9,4,1,2]
b = [0,8,5,4,6,7,6,0,4]
1 - cosine(a - np.mean(a), b - np.mean(b))


# Jaccard Distance (example from the slide)
jaccard([1, 0, 1, 0, 1, 1, 0, 0], [1, 0, 0, 1, 1, 1, 0, 1])


# # User-User Filtering (without Bias)
# We first define helper functions to extract common ratings from two rating vectors and to calculate the common cosine similarity
common_ratings = lambda x, y : np.argwhere(
    (~np.isnan(x)) & (~np.isnan(y)) & (x >= 0) & (y >= 0)
)[:, 0]

def common_cosine(v1, v2):
    common = common_ratings(v1, v2)
    v1c, v2c = v1[common], v2[common]
    if len(v1c) <= 1: return 0
    return 1 - cosine(v1c, v2c)


# Get the pairwise similarities between users using `pairwise_distances` and our custom function. The passed array needs to have a shape of `[n_samples, n_features]`. Since we have `[users, items]`, we'll calculate distance between users given their ratings as feature vectors, which is what we want here
distances = pairwise_distances(np.nan_to_num(ratings, nan=-1), metric=common_cosine)

# Now let's try to recommend some items for a user

# Which user are we going to provide recommendations for?
user_to_recommend_for = users.index('Bart')

# Similarities for user_to_recommend_for to others
distances[user_to_recommend_for, :]

# Get the k-nearest neighbors for the user
k = 2
nb_idx = np.argsort(distances[0, :])[::-1][1:1+k] # 1+k as an element is always nearest to itself
nb_idx

# Now we simply need to the ratings of the neighbors per item
neighbor_ratings = ratings[nb_idx, :]
recommendations = np.nanmean(neighbor_ratings, axis=0)

# Note that we get a warning (and a `nan` result) for the last item, as none of the neighbors has rated this one

# Show the recommended values for movies the user hasn't seen yet
for itm_idx in np.argwhere(np.isnan(ratings[user_to_recommend_for, :]))[:, 0]:
    print('Item index {} ({}) rating = {}'.format(itm_idx, items[itm_idx], recommendations[itm_idx]))


# # User-User Filtering (with Bias)

# Which user are we going to provide recommendations for?
user_to_recommend_for = users.index('Bart')

# Get the k-nearest neighbors for the user
k = 2
nb_idx = np.argsort(distances[0, :])[::-1][1:1+k] # 1+k as an element is always nearest to itself


# First, get the average ratings for each user and stack it to a vertical array (this will make broadcasting easier later on)
rating_bias = np.expand_dims(np.nanmean(ratings, axis=1), axis=1)

# Calculate the nominator item-wise. For each item: sum the (ratings of the neighbors - their bias) x the similarity
neighbor_ratings_unbiased         = ratings[nb_idx, :] - rating_bias[nb_idx]
neighbor_vertical_similarities    = np.expand_dims(distances[user_to_recommend_for, nb_idx], axis=1)
neighbor_ratings_times_similarity = neighbor_ratings_unbiased * neighbor_vertical_similarities
neighbor_ratings_summed           = np.nansum(neighbor_ratings_times_similarity, axis=0)


# Calculate the denominator: sum of similarities of neighbors
summed_similarities = np.sum(distances[user_to_recommend_for, nb_idx])

# We can now calculate the results
recommendations = rating_bias[user_to_recommend_for] + neighbor_ratings_summed / summed_similarities

# Show the recommended values for movies the user hasn't seen yet
for itm_idx in np.argwhere(np.isnan(ratings[user_to_recommend_for, :]))[:, 0]:
    print('Item index {} ({}) rating = {}'.format(itm_idx, items[itm_idx], recommendations[itm_idx]))
```

## Item-Item Collaborative Filtering
- Intuition: "Items which were previously liked by same users, will continue to be liked by same 
  users." 
- 2 steps: 
- - Calculate Item-Item similarity matrix.
- - Recommend items similar to the items a user has shown an interest in.
- Item-Item vs User-User collaborative filtering:
- - number of items vs number of users.
- - item are simpler whereas users typically have multiple tastes.
- - users may like similar items but each also have their own-specific differences.
- Amazon.com uses item-item collaborative filtering.

## Item-Item Collaborative Filtering in Python
```python

import numpy as np

from scipy.stats import pearsonr
from scipy.spatial.distance import cosine
from scipy.spatial.distance import jaccard

from sklearn.metrics import pairwise_distances
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

# Item-Item Filtering (without Bias)
# Same helper functions as for User-User

common_ratings = lambda x, y : np.argwhere(
    (~np.isnan(x)) & (~np.isnan(y)) & (x >= 0) & (y >= 0)
)[:, 0]

def common_cosine(v1, v2):
    common = common_ratings(v1, v2)
    v1c, v2c = v1[common], v2[common]
    if len(v1c) <= 1: return 0
    return 1 - cosine(v1c, v2c)

#Get the pairwise similarities between users using pairwise_distances and our custom function. 
# The passed array needs to have a shape of [n_samples, n_features]. Since we have [users, items],
# we need to transpose the matrix here

distances = pairwise_distances(np.nan_to_num(ratings.T, nan=-1), metric=common_cosine)

# Example: let's get the similarities between a movie to the others
# Note the relatively high similarity to Nothing Hill. This is because we're only considering 
# common ratings. An alternative approach would be to impute with 0 values, the mean rating, or 
# mean rating for the movie, as discussed in the course

distances[items.index('Jurassic Park'), :]

# Which user are we going to provide recommendations for?
user_to_recommend_for = users.index('Bart')


# Get the k-nearest neighbors for the item
k = 2

# Note that we now need the nearest neighbors for each item! Not just for the user we want to recommend for
nb_idx = np.argsort(distances)[:, ::-1][:, 1:1+k]
nb_idx, nb_idx.shape # 7 movies which each k=2 neighbors

# Get the neighboring similarities per item
neigbor_similarities = distances[np.expand_dims(range(nb_idx.shape[0]), axis=1), nb_idx]

# Get the neigboring ratings per item
neigbor_ratings = ratings[user_to_recommend_for , nb_idx]

# Multiply both and sum
multiplied = neigbor_similarities * neigbor_ratings

neighbors_summed = np.nansum(multiplied, axis=1)

# Divide by sum of similarities
recommendations = neighbors_summed / np.sum(neigbor_similarities, axis=1)


# Show the recommended values for movies the user hasn't seen yet
for itm_idx in np.argwhere(np.isnan(ratings[user_to_recommend_for, :]))[:, 0]:
    print('Item index {} ({}) rating = {}'.format(itm_idx, items[itm_idx], recommendations[itm_idx]))

# Item-Item Filtering (with Bias)
# Which user are we going to provide recommendations for?
user_to_recommend_for = users.index('Bart')
user_to_recommend_for

# Get the k-nearest neighbors for the item
k = 2

# Note that we now need the nearest neighbors for each item! Not just for the user we want to recommend for
nb_idx = np.argsort(distances)[:, ::-1][:, 1:1+k]
nb_idx, nb_idx.shape # 7 movies which each k=2 neighbors

# Get the neighboring similarities per item
neigbor_similarities = distances[np.expand_dims(range(nb_idx.shape[0]), axis=1), nb_idx]

# Get the average ratings per item, again as a vertical vector
rating_bias = np.expand_dims(np.nanmean(ratings, axis=0), axis=1)


# Again, the warning here is expected: the last movie has no ratings

# Get the nominator: first get the ratings minus the rating bias for each item
neighbor_ratings_unbiased = ratings[user_to_recommend_for, nb_idx] - rating_bias
neighbor_ratings_unbiased, neighbor_ratings_unbiased.shape
```

## K-nearest neighbor based Filtering: Extensions
- Spreading activation:
- - exploit transition of customer preferences
- - assume we are looking for recommendations for User 1
- - in collaborative filtering, User 2 is neighbor for User 1 because both bought Item 2 and Item 4
- - recommend Item 3 to User 1, because User 2 bought it.
- - User-based or item-based CF approaches assume paths of length 3: Item 3 is relevant for User 
    1 because there exists a 3-step path (User1-Item2-User2-Item3) between them.
- - Since number of such paths of length 3 is small in parse rating matrices, idea is to also 
    consider longer paths.
- Slope One Predictors
- Simple item based collaborative filtering approach for non-binary ratings.
- Comparable performance to more complex approaches.

|         | Item 1 | Item 2 | Item 3 |
|---------|--------|--------|--------|
| Bart    | 5      | 3      | 4      |
| Michael | 4      | 1      | ?      |
| Tim     | ?      | 2      | 4      |

- Avg diff in rating between Item 1 an Item 2 is ((5-3) + (4-1))/2 = 2.5
- Avg diff between Item 1 and Item 3 is 1
- Options to predict Tim's rating for Item 1:
- - Only use Item 2: 2 + 2.5 = 4.5
- - Only use Item 3: 4 + 1 = 5
- - Weighted average: (4.5 + 5) / (1+2) = 4.67
- Recursive Collaborate Filtering
- Assume there is a very close neighbor Michael of Bart who not rated the target item 5 as well.
- Apply collaborative filtering recursively and predict rating for item 5 for Michael.
- Use this predicted rating instead of rating of more distant direct neighbor.

## Slope One predictors in Python
```python
import pandas as pd
import numpy as np

# Install surprise with
# conda install -c conda-forge scikit-surprise

from surprise.reader import Reader
from surprise import Dataset
from surprise import accuracy

from surprise.prediction_algorithms.slope_one import SlopeOne
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
# make one

df = pd.DataFrame([
    [u, i, ratings[u,i]]
    for u in range(len(users)) for i in range(len(items))
], columns=['user', 'item', 'rating'])
df.head()

#surprise doesn't like missing values, so we impute with 2.5 (you can use other strategies as 
# well, as discussed in the course)

df.replace(to_replace=np.nan, value=2.5, inplace=True)
reader = Reader(rating_scale=(0,5))
data   = Dataset.load_from_df(df, reader)

# Slope One
# Train on the complete data set. surprise also has support for cross-validation and train/test 
# splitting, see docs at https://surprise.readthedocs.io/en/stable/getting_started.html

trainset = data.build_full_trainset()

# Build an algorithm, and train it.
recommender = SlopeOne()
recommender.fit(trainset)

# Make some predictions

user_to_recommend_for = 0

for itm_idx in np.argwhere(np.isnan(ratings[user_to_recommend_for, :]))[:, 0]:
    prediction = recommender.predict(user_to_recommend_for, itm_idx)
    print('Item index {} ({}) rating = {}'.format(itm_idx, items[itm_idx], prediction))
```

## K-nearest neighbor based filtering: Advantages
- Easy to develop
- Easy to explain recommendations
- Good performance
- Widely studied and applied

## K-nearest neighbor based filtering: Disadvantages
- Cold start problem for new users and items: what about users\items with no\very few ratings?
- Performance drops when sparsity increases
- Popularity bias: items liked by more users have higher chance of being recommended
- Scaling: do not scale well for most real-world scenarios, E-business sites have tens of 
  millions of customers and millions of items.

## K-nearest neighbor based Filtering : Scientific Perspective

|                       | Binary purchase char.controlled | Reduction as preprocessing | CF method  | Similarity measure | Evaluation metric |
|-----------------------|---------------------------------|----------------------------|------------|--------------------|-------------------|
| Breese et al. 1998    | None                            | None                       | User-based | Cosine             | Accuracy          |
| Li et al. 2011        | None                            | None                       | User-based | Cosine             | Accuracy          |
| Deshpande et al. 2004 | None                            | None                       | Item-based | Cosine             | Accuracy          |
| Linden et al. 2003    | None                            | None                       | Item-based | Cosine             | Accuracy          |
| Pradel et al. 2012    | None                            | None                       | Item-based | Cosine             | Accuracy          |
| Sarwar et al. 2001    | Sparsity                        | None SVD                   | User-based | Cosine             | Accuracy          |
| Geuens et al. 2018    | Sparsity                        | None SVD                   | User-based | Cosine             | Accuracy          |
