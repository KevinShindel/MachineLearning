# OverSampling Guide

- [Official documentation](https://imbalanced-learn.org/stable/over_sampling.html)

## Legend

#### Dataset Size

- ⭐ = Small (<50k rows)
- ⭐⭐ = Small–Medium
- ⭐⭐⭐ = Medium (~50k–500k)
- ⭐⭐⭐⭐ = Large (~500k–5M)
- ⭐⭐⭐⭐⭐ = Very Large (millions+)

#### Speed

- ⭐ = Very Slow
- ⭐⭐ = Slow
- ⭐⭐⭐ = Moderate
- ⭐⭐⭐⭐ = Fast
- ⭐⭐⭐⭐⭐ = Very Fast

## Under-Sampling

| Class Name                               | Type  | Dataset Size | Speed | Purpose                                   | Other                                |
|------------------------------------------|-------|--------------|-------|-------------------------------------------|--------------------------------------|
| `RandomUnderSampler`                     | Under | ⭐⭐⭐⭐⭐        | ⭐⭐⭐⭐⭐ | Fast reduction of majority class          | Industry standard for large datasets |
| `NearMiss`                               | Under | ⭐⭐           | ⭐⭐    | Keep majority samples closest to minority | Distance-based, expensive            |
| `TomekLinks`                             | Under | ⭐⭐           | ⭐⭐    | Remove overlapping majority samples       | Good cleanup before modeling         |
| `EditedNearestNeighbours (ENN)`          | Under | ⭐⭐           | ⭐     | Remove noisy samples                      | Very expensive on large data         |
| `RepeatedEditedNearestNeighbours (RENN)` | Under | ⭐            | ⭐     | Aggressive noise removal                  | Multiple ENN passes                  |
| `AllKNN`                                 | Under | ⭐            | ⭐     | Progressive ENN cleaning                  | Computationally heavy                |
| `CondensedNearestNeighbour (CNN)`        | Under | ⭐⭐           | ⭐⭐    | Keep representative majority samples      | Produces compact dataset             |
| `OneSidedSelection (OSS)`                | Under | ⭐⭐           | ⭐⭐    | Remove redundant majority examples        | CNN + Tomek Links                    |
| `NeighbourhoodCleaningRule (NCL)`        | Under | ⭐⭐           | ⭐     | Clean class boundaries                    | Strong noise reduction               |
| `InstanceHardnessThreshold (IHT)`        | Under | ⭐⭐⭐          | ⭐⭐    | Remove hard-to-classify majority samples  | Requires model training              |
| `ClusterCentroids`                       | Under | ⭐⭐⭐          | ⭐⭐⭐   | Replace majority with centroids           | Information compression              |

## OverSampling Guide

| Class Name          | Type    | Dataset Size | Speed | Purpose                                  | Other                            |
|---------------------|---------|--------------|-------|------------------------------------------|----------------------------------|
| `RandomOverSampler` | Over    | ⭐⭐⭐⭐⭐        | ⭐⭐⭐⭐⭐ | Duplicate minority samples               | Fastest oversampler              |
| `SMOTE`             | Over    | ⭐⭐⭐          | ⭐⭐⭐   | Generate synthetic minority samples      | Most widely used                 |
| `BorderlineSMOTE`   | Over    | ⭐⭐           | ⭐⭐    | Focus on decision boundary               | Better for overlap regions       |
| `SVMSMOTE`          | Over    | ⭐            | ⭐     | SMOTE guided by SVM                      | Very slow                        |
| `KMeansSMOTE`       | Over    | ⭐⭐           | ⭐⭐    | Cluster-aware synthetic generation       | Better for complex distributions |
| `ADASYN`            | Over    | ⭐⭐           | ⭐⭐    | Generate more samples in difficult areas | Can amplify noise                |
| `SMOTENC`           | Over    | ⭐⭐⭐          | ⭐⭐⭐   | Mixed numerical/categorical features     | Common in business data          |
| `SMOTEN`            | Over    | ⭐⭐           | ⭐⭐⭐   | Fully categorical datasets               | Rare but useful                  |
| `SMOTEENN`*         | Combine | —            | —     | Listed below                             | Combination method               |
| `SMOTETomek`*       | Combine | —            | —     | Listed below                             | Combination method               |

## Combine Methods 

| Class Name   | Type    | Dataset Size | Speed | Purpose               | Other                            |
|--------------|---------|--------------|-------|-----------------------|----------------------------------|
| `SMOTEENN`   | Combine | ⭐⭐           | ⭐⭐    | SMOTE + ENN cleaning  | Often strong accuracy, expensive |
| `SMOTETomek` | Combine | ⭐⭐⭐          | ⭐⭐⭐   | SMOTE + Tomek cleanup | Popular balanced approach        |

## Ensemble Methods

| Class Name                       | Type     | Dataset Size | Speed | Purpose                                 | Other                              |
|----------------------------------|----------|--------------|-------|-----------------------------------------|------------------------------------|
| `BalancedBaggingClassifier`      | Ensemble | ⭐⭐⭐⭐         | ⭐⭐⭐   | Bagging with balanced bootstrap samples | Good baseline                      |
| `BalancedRandomForestClassifier` | Ensemble | ⭐⭐⭐⭐         | ⭐⭐⭐⭐  | Random forest with class balancing      | Production-friendly                |
| `EasyEnsembleClassifier`         | Ensemble | ⭐⭐⭐⭐         | ⭐⭐⭐⭐  | Multiple undersampled AdaBoost models   | Very effective on severe imbalance |
| `RUSBoostClassifier`             | Ensemble | ⭐⭐⭐          | ⭐⭐⭐   | Random undersampling + Boosting         | Good for difficult imbalance       |


## Practical Recommendations

| Situation                          | Recommended Method                                                    |
|------------------------------------|-----------------------------------------------------------------------|
| Dataset < 50k rows                 | `SMOTE`, `BorderlineSMOTE`, `SMOTEENN`                                |
| Dataset 50k–500k rows              | `RandomUnderSampler`, `SMOTENC`, `BalancedRandomForestClassifier`     |
| Dataset > 500k rows                | `RandomUnderSampler`, `BalancedRandomForestClassifier`, class weights |
| Millions of rows                   | Class weights, `EasyEnsembleClassifier`, gradient boosting            |
| High-dimensional One-Hot data      | Avoid ENN, NearMiss, AllKNN                                           |
| Noisy data                         | `ENN`, `NCL`, `SMOTEENN`                                              |
| Strong class overlap               | `BorderlineSMOTE`, `SMOTETomek`                                       |
| Extreme imbalance (1:100+)         | `EasyEnsembleClassifier`, `BalancedRandomForestClassifier`            |
| Mixed categorical/numeric features | `SMOTENC`                                                             |
| All categorical features           | `SMOTEN`                                                              |



### Categorical Features OverSampling 

```python

import numpy as np
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTENC
from collections import Counter

X, y = make_classification(
    n_classes=2,
    class_sep=2,
    weights=[0.1, 0.9],
    n_informative=3,
    n_redundant=1,
    n_features=5,
    n_clusters_per_class=1,
    n_samples=100,
    random_state=42
)

categorical_features = [0, 3]

print("Before SMOTE-NC:", Counter(y))

smote_nc = SMOTENC(categorical_features=categorical_features, random_state=42)
X_res, y_res = smote_nc.fit_resample(X, y)

print("After SMOTE-NC:", Counter(y_res))
```