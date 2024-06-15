## User Interest
- Value recommender system try to predict user interest.
- Can be difficult to define and measure.
- Highly dependent on the context: movies, jobs, facebook.
- **Explicit user interest**: more robust, user effort, less data, biased, sparse data
- **Implicit user interest**: noisy, no user effort, non-intrusive, lost of data.
- Both explicit and implicit user interest do not necessarily match.
- **Explicit user interest types**:
- - Amazon used starts to rate products.
- - MovieLens used stars to rate movies.
- - Facebook used likes, smiles to rate posts.
- - LinkedIn used endorsements to rate skills.
- - Tinder used swipes to rate people.
- **Implicit user interest types**:
- - Clicking on a link.
- - Time spend on page.
- - Demo downloads.
- - Read time.
- - Scrolling.
- - Mouse movement.
- - Bookmarking.
- - Printing.
- - Online apps.
- - Message boards.
- Case study on jon recommendation
- Looked at jobs users clicked on
- Compared with job-types users said they are interested in
- Big discrepancy.

## Rating Matrix
- Constructed by aggregating interest data into a single scope per user-item pair: rows 
  represent users, columns represent items.
- Rating (U, I) = f(observed interest data): One feedback/combination of multiple 
  feedback indicators.

|         | Rambo 2 | Rocky 6 | Silent Hill | The Godfather | The Matrix | The Notebook |
|---------|---------|---------|-------------|---------------|------------|--------------|
| Alice   | 5       | 4       | 0           | 0             | 0          | 0            |
| Bob     | 0       | 0       | 0           | 5             | 0          | 0            |
| Charlie | 0       | 0       | 3           | 0             | 5          | 0            |
| David   | 0       | 0       | 0           | 0             | 0          | 5            |
| Eve     | 0       | 2       | 5           | 0             | 0          | 1            |

- Scalability problem: 
- - E.g. 1m users, 100k items, 1b ratings.
- - 10 billion rating combinations (+-40Gb with 32-bit integers).
- - Full matrix expansion is time/memory inefficient
- Sparsity problem:
- - not all users are going to give an opinion on all items.
- - catalogue of 10m books, probability 2 users who bought 50 books each, have book in common is 0.0005.
- - Sparsity of <<1% are common.
- Rating bias problem:
- - not everyone is equality generous with their ratings.
- - data normalization.

## Recommender System Workings
- Collaborative filtering
- Content filtering
- Knowledge based filtering
- Hybrid recommender systems

## Goal of Recommender System
- Prediction: predict missing ratings
- Ranking: predict top k users for an item, or top k items for a user.
- 

## Evaluating Recommender Systems
- Evaluation is key to success of analytical model
- Two decisions: 
- - Decide what items to generate predictions for
- - Decide how to measure performance 
- Prediction rating
- Prediction conversion
- Ranking
- Other criteria

## Evaluating Recommender Systems: Prediction
- Prediction accuracy measuring:
- - Root Mean Squared Error (RMSE)
- - Mean Absolute Deviation (MAD)
- - Pearson correlation

## Evaluating Recommender Systems: Conversion
- Predicting conversation technique
- 
| items | recommendation score | converted |
|-------|----------------------|-----------| 
| Item1 | 0.1                  | No        | 
| Item2 | 0.88                 | Yes       |
| Item3 | 0.24                 | Yes       |

          Cut-off = 0.5

| items | Recommended | converted |
|-------|-------------|-----------|
| Item1 | No          | No        |
| Item2 | Yes         | Yes       |
| Item3 | No          | Yes       |

|               | Recommended         | Not Recommended     |
|---------------|---------------------|---------------------|
| Converted     | True Positive (TP)  | False Negative (FN) |
| Not Converted | False Positive (FP) | True Negative (TN)  |

- **Precision** = TP / (TP + FP) = percentage of recommended items that are relevant.
- **Recall** = TP / (TP + FN) = percentage of relevant items that are recommended.
- **F-measure** = 2 * (Precision * Recall) / (Precision + Recall) = harmonic mean of precision and recall.
- Trade-off between precision and recall.
- Precision/recall on top N
- Precision-recall curve.
- Sensitivity = TP/(TP+FN) = recall
- Specificity = TN/(TN+FP) = percentage of non-relevant items that are not recommended.
- Make table with sensitivity, specificity and 1-specificity for each possible cut-off
- ROC curve plots sensitivity vs 1-specificity for all possible cut-offs.
- Perfect model has sensitivity of 1 and specificity of 1.
- What about intersecting ROC curves?
- ROC curve can be summarized by area underneath (AUC) - th higher the better.
- AUC provides estimate of probability that randomly chosen converter gets higher recommendation 
  score that randomly chosen non-converter.
- Diagonal represents benchmark.
- AUC > 0.5
- Precision-recall curve emphasizes percentage of successful recommendations.
- ROC curve emphasizes percentage of non-converters that were recommended.
- Online video rental recommender system: false positive rate very relevant.
- Precision-recall curves better to highlight differences for highly imbalanced data sets.

## Evaluating Recommender Systems: Ranking
- Ranking: 
- - useful for top k recommendations.
- - predicted and actual rankings should correspond

| Items             | Actual Ranking | Predicted Ranking |
|-------------------|----------------|-------------------|
| Gladiator         | 10             | 8                 |
| Rocky 6           | 6              | 7                 |
| Rambo 2           | 8              | 6                 | 
| The Godfather     | 4              | 5                 |
| Lord of the Rings | 6              | 1                 |

- Precision cut-off k: Precision(k) = TP/(TP+FP)

| Rank | Recommendation    | Result | 
|------|-------------------|--------|
| 1    | Lord of the Rings | TP     |
| 2    | Gladiator         | FP     |
| 3    | Rambo 2           | TP     |
| 4    | The Godfather     | FP     |
| 5    | Rocky 6           | FP     |

Precision for top 3 = 2/3
Precision for top 5 = 2/5

| Rank | Perfect   | Scenario 1   | Precision (i) | Scenario 2   | Precision (i) | Scenario 3 | Precision (i) |
|------|-----------|--------------|---------------|--------------|---------------|------------|---------------|
| 1    | Rocky 6   | Silent Hill  | 0             | Harry Potter | 0             | Rocky 6    | 1/1           |
| 2    | Rambo 2   | Harry Potter | 0             | Rambo 2      | 1/2           | Rambo 2    | 2/2           |
| 3    | Gladiator | Rocky 6      | 1/3           | Gladiator    | 2/3           | Gladiator  | 3/3           |


- **Spearman** rank order correlation:
- - measures degree to which monolithic relationship exists between predicted and actual ratings.
- - Compute numeric ranks by assigning 1 to the lowest rating, 2 to second-lowest rating, etc.
- - Average in case of tied ratings.
- - Spearman then becomes Pearson correlation between predicted and actual ranks.
- - Ranges between -1(perfect disagreement) and 1(perfect agreement).

| Items | Actual Rating | Predicted Rating | Score Actual 

## Evaluating Recommender Systems: Other Criteria
