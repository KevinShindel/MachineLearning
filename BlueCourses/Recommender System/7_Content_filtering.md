## Content filtering: Basic Idea
- Intuition: Recommend items that are similar to those that a user liked in the past.
- Recommended items to customer similar to previous items rated highly by the customer.
- Recommendations based on content of items, instead of other users opinions.
- Description of item content based on attributes or features (author, genre, etc.).
- Examples: Movie recommendations (based on genre, actors, etc.), music recommendations (based 
  on artist, genre, etc.)., web page recommendations (based on content of pages visited).
- Positive opinion on:
- - The Shining, Thriller, Jack Nicholson, Stanley Kubrick, 144 minutes, 1980.
- - Full Metal Jacket, War, Matthew Modine, Stanley Kubrick, 116 minutes, 1987.
- Negative opinion on:
- - The bucket list, Comedy, Jack Nicholson, Rob Reiner, 97 minutes, 2007.
- - Saving Private Ryan, War, Tom Hanks, Steven Spielberg, 169 minutes, 1998.
- What can we say about:
- 2001: A Space Odyssey, Science Fiction, Stanley Kubrick, 160 minutes, 1968.

## Item and User Profiles
- Build item profile for each item
- Profile is set (vector) of item features:
- - Movie: title, actors, director, genre, etc.
- - Text: words in document, author, etc.
- - Web page: words in page, links, etc.
- How to pick important features?
- - Heuristic: pick most frequent words. (e.g. TF-IDF) Term = Feature, Document = Item
- User profile: average of rated item profiles. variation: weight by difference from average 
  rating on item.
- Can use classification algorithm (e.g decision tree) to build profiles or make recommendations.
- Example: 
- - Items: new articles
- - Features: high TD-IDF words, author, date, etc.

## Content Filtering in Python

```python
import pandas as pd
import numpy as np

import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt


# We're going to use the Wikiedia "movie plots" data set, which can be downloaded from https://www.kaggle.com/jrobischon/wikipedia-movie-plots/data

df = pd.read_csv('wiki_movie_plots_deduped.csv')
df.head(3)

# Content Filtering
# Preprocessing
# Drop Wiki Page as we won't need it
df.drop(columns=['Wiki Page'], inplace=True)
# Make TF-IDF features for Plot
vectorizer = TfidfVectorizer(stop_words='english', min_df=0.1, max_df=0.9)
X = vectorizer.fit_transform(df.Plot)

print(X.shape)
print(list(vectorizer.vocabulary_.keys()))

# Make sure that the columns are in the right order
cols = [[x for x,c in vectorizer.vocabulary_.items() if c == i][0] for i in range(X.shape[1])]

plot_features = pd.DataFrame(data=X.todense(), columns=cols)
plot_features.head(3)

# Replace low-occurring Origins with a single level
for u in zip(*np.unique(df['Origin/Ethnicity'], return_counts=True)):
    if u[1] / len(df) < 0.01:
        df.loc[df['Origin/Ethnicity'] == u[0], 'Origin/Ethnicity'] = 'OTHER'
        
for u in zip(*np.unique(df['Origin/Ethnicity'], return_counts=True)): print(u)

# Convert to dummies
origin = pd.get_dummies(df['Origin/Ethnicity'])
print(origin)

# Helper function to convert a column of text to a new feature matrix based on slitting the text up into words and creating dummy variables. Columns with only a couple of occurrences are dropped

def make_feature_matrix(col, splitter=r'\s*,\s*', thresh=20):
    colm = col.apply(lambda x : list(
        set([w for w in re.split(splitter, str(x).strip()) if w and w.strip()])
    ))
    mlb = MultiLabelBinarizer() # works faster than pd.get_dummies
    fm = pd.DataFrame(mlb.fit_transform(colm), columns=mlb.classes_, index=df.index)
    cols_to_remove = np.argwhere(np.sum(fm.to_numpy(), axis=0) < thresh)[:, 0]
    fm.drop(columns=fm.columns[cols_to_remove], inplace=True)
    return fm

# Create feature matrices for directors, genre and cast
directors = make_feature_matrix(df['Director'])
genres    = make_feature_matrix(df['Genre'], splitter=r',|—|-|/|;|\s+') # use a liberal split here
cast      = make_feature_matrix(df['Cast'])
# Also drop the unknown/nan columns
directors.drop(columns=['Unknown'], inplace=True)
genres.drop(columns=['unknown'], inplace=True)
cast.drop(columns=['nan'], inplace=True)
# Which genres do we have now?
print(genres.columns)


# # Now concat everything back together to a single feature matrix, use the titles as index
stitched = pd.concat([
    df[['Title', 'Release Year']], 
    origin,
    plot_features, 
    directors,
    genres, 
    cast
], sort=False, axis=1).set_index('Title')

stitched.head(3)


# Modelling
# Let's say we have a user which has liked the following movies (e.g. a random bunch of recent action movies)
mask         = (stitched['Release Year'] > 2005) & (stitched['Origin/Ethnicity'] == 'American') & (stitched['action'] > 0)
liked_movies = list(stitched.loc[mask, :].index)
liked_movies = np.random.choice(liked_movies, 50, replace=False)
# One hot encode the feature matrix and create binary target vector
X = stitched
y = stitched.index.isin(liked_movies).astype(int)
# We'll use a random forest classifier here as an example. For the sake of brevity, we don't 
# create a train/test split in this example

# Also note that you could still use a distance metric on the constructed (normalized) feature 
# matrix and predict similar items that way, this doesn't require building a model per user (as 
# we do here)

classifier = RandomForestClassifier(class_weight='balanced')
classifier.fit(X, y)

probs           = classifier.predict_proba(X)[:, 1]
recommendations = np.argsort(probs)[::-1]
# Print top list of unwatched movies
i = 0
for recommendation in stitched.index[recommendations]:
    if recommendation in liked_movies: continue
    i += 1
    if i > 10: break
    print(recommendation)

# "Harry Potter and the Deathly Hallows: Part I" is an interesting recommendation... wasn't it in the selected genres already?

df.loc[df.Title=="Harry Potter and the Deathly Hallows: Part I", :]

# Nope, turns out to be fantasy

# Which features has the random forest picked up on?

importances = classifier.feature_importances_
std = np.std([tree.feature_importances_ for tree in classifier.estimators_], axis=0)
indices = np.argsort(importances)[::-1][:15]

plt.figure(figsize=(15,5))
plt.title("Feature Importances")
plt.bar(range(len(indices)), importances[indices], color="r", yerr=std[indices], align="center")
plt.xticks(range(len(indices)), [X.columns[i] for i in indices])
plt.xlim([-1, len(indices)])
plt.show()


```

## Content filtering: Advantages
- No cold start problem for new items (but still problem for new users).
- No sparsity problem.
- Easy to explain recommendations.
- Recommend to users with unique tastes.
- Recommend new and unpopular items.

## Content filtering: Disadvantages
- Tagging is expensive (e.g. images, movies, music).
- No info from other users.
- Recommendations for new users are not accurate.
- Over-specialization.
