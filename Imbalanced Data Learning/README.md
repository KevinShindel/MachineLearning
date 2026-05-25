# OverSampling Guide

- [Official documentation](https://imbalanced-learn.org/stable/over_sampling.html)


| SMOTE Variant                 | 	Best Use Case                                                | 	Main Strength                                                                     | 	When to Use                                                                         |
|-------------------------------|---------------------------------------------------------------|------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------|
| SMOTE (Standard)              | 	Moderately imbalanced continuous datasets                    | 	Balances classes using synthetic interpolated samples.                            | 	Use for numeric, low-noise datasets works as a general solution                     |
| ADASYN (Adaptive SMOTE)       | 	Datasets with region-wise imbalance	                         | Generates adaptive synthetic data for hard-to-learn samples	                       | Use for hard to classify minority regions to improve boundary learning               |
| Borderline SMOTE              | 	Minority samples close to class boundaries                   | 	Generates samples near decision boundaries to reduce misclassification	           | Use when classes overlap or boundaries are frequently confused                       |
| SMOTE-ENN (Hybrid)            | 	Noisy datasets with misclassified or ambiguous samples       | 	Combines SMOTE and ENN to oversample and clean noisy instances                    | 	Use when the dataset has noise or outliers and we want a cleaner, balanced dataset. |
| SMOTE-TOMEK (Hybrid)          | 	Datasets with overlapping classes needing clearer separation | 	Removes Tomek links post-SMOTE to reduce overlap and improve separation	          | Use when we want to improve boundary clarity after oversampling.                     |
| SMOTE-NC (Nominal Continuous) | 	Datasets with both categorical and continuous features       | 	Handles mixed features via numeric interpolation and categorical mode assignment	 | Use for datasets with categorical columns not suited for purely numeric data         |

### Advantages
- Balances dataset by generating synthetic minority samples, improving model performance.
- Reduces overfitting compared to simple duplication.
- Works with various classifiers and high-dimensional data.
- Variants (Borderline-SMOTE, ADASYN) adapt to different imbalance scenarios.

### Limitations
- Can amplify noise if minority class has noisy samples.
- May blur class boundaries when classes overlap.
- Higher computational cost for large datasets or complex variants.
- Less effective for extreme class imbalance; relies on nearest neighbors.



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