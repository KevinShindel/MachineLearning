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
- 

## User-User Collaborative Filtering: Prediction
- 

## User-User Collaborative Filtering in Python
- 

## Item-Item Collaborative Filtering
- 

## Item-Item Collaborative Filtering in Python
- 

## K-nearest neighbor based Filtering: Extensions
- 

## Slope One predictors in Python
- 

## K-nearest neighbor based filtering: Advantages
- 

## K-nearest neighbor based filtering: Disadvantages
- 

## K-nearest neighbor based Filtering : Scientific Perspective
- 
