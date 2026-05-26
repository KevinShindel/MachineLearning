# Outlier Detection

## Approaches
1. Outlier Removing
2. Outlier Replacing
3. Outlier Transformation ( aka log transform )
4. Winsorization ( aka data clippin`)
5. Algorithmic Imputation
6. Isolation Forest

## Techniques Description table

| Technique              | Advantages                                                                     | Disadvantages                                                                     | Time Complexity | Best Cases                                             | Not Recommended                                                 | Metrics to Evaluate Effectiveness                                                                |
|------------------------|--------------------------------------------------------------------------------|-----------------------------------------------------------------------------------|-----------------|--------------------------------------------------------|-----------------------------------------------------------------|--------------------------------------------------------------------------------------------------|
| Removing               | Simple, fast, improves model robustness, removes noisy observations completely | Information loss, can reduce dataset size, dangerous for small datasets           | Low             | Large datasets with obvious anomalies/errors           | Small datasets, rare-event prediction, fraud/anomaly detection  | Model score improvement (F1, RMSE, ROC-AUC), percentage of removed rows, distribution comparison |
| Replacing              | Preserves row count, easy to implement, stabilizes variance                    | Can distort distribution, introduces artificial values, weak statistical validity | Low             | Moderate outliers in structured/tabular data           | Highly skewed data, multimodal distributions                    | Mean/median shift, variance reduction, model metric improvement                                  |
| Transformation         | Reduces skewness, stabilizes variance, improves linear model assumptions       | Harder interpretation, not suitable for negative/zero values (for log transform)  | Low             | Right-skewed financial/sales/scientific data           | Data with many zeros/negatives unless using Box-Cox/Yeo-Johnson | Skewness reduction, kurtosis reduction, normality tests, RMSE/R² improvements                    |
| Winsorization          | Keeps all rows, limits influence of extreme values, robust for statistics      | Extreme values still exist indirectly, threshold selection sensitive              | Low             | Financial data, robust regression/statistical modeling | When true anomalies are important signals                       | Quantile stability, coefficient stability, model variance reduction                              |
| Algorithmic Imputation | More statistically accurate, preserves structure, multivariate-aware           | Computationally expensive, risk of data leakage, harder to tune/debug             | Medium–High     | Complex datasets with feature correlations             | Very large datasets, real-time systems                          | Cross-validation score, imputation error, reconstruction error, downstream model metrics         |

---

## Choosing the Best Technique

| Scenario                                  | Recommended Technique                          |
|-------------------------------------------|------------------------------------------------|
| Few obvious data-entry errors             | Removing                                       |
| Small dataset where row loss is expensive | Replacing / Winsorization                      |
| Highly skewed distributions               | Transformation                                 |
| Financial/statistical analysis            | Winsorization                                  |
| Correlated multivariate features          | Algorithmic Imputation                         |
| Real-time ML pipeline                     | Winsorization / Replacing                      |
| Deep learning preprocessing               | Transformation + Winsorization                 |
| Tree-based models (RF/XGBoost)            | Often no treatment needed unless extreme noise |

---

## Practical Rule of Thumb

| Dataset Type               | Recommended Strategy          |
|----------------------------|-------------------------------|
| Small tabular dataset      | Winsorization                 |
| Large noisy dataset        | Removing                      |
| Skewed numerical features  | Log/Power transformation      |
| Correlated enterprise data | KNN/Iterative Imputation      |
| Financial data             | Winsorization + RobustScaler  |
| Sensor/IoT streams         | Clipping + Rolling statistics |

---
## Common Metrics for Outlier Treatment Evaluation

| Metric             | Purpose                                      |
|--------------------|----------------------------------------------|
| Skewness           | Measure asymmetry reduction                  |
| Kurtosis           | Measure heavy-tail reduction                 |
| IQR                | Check spread stabilization                   |
| Variance / Std Dev | Detect volatility reduction                  |
| KS Test            | Compare original vs transformed distribution |
| Shapiro-Wilk Test  | Evaluate normality                           |

---

### 1. Data Removing approach
- Description: If total outliers quantity is not too large we can just remove them 

#### Concept

```python
import pandas as pd

df = pd.DataFrame([])

Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)

IQR = Q3 - Q1

lower_bound = IQR - 1.5 * Q1
upper_bound = IQR + 1.5 * Q3

df = df[(df >= lower_bound) | (df <= upper_bound )]
```

---

### 2. Outlier Replacing
- Description: replace outliers with median / mean / mode etc. 

#### Concept

```python
import pandas as pd

df = pd.DataFrame([])

outlier_column = 'col_1'

Q1 = df.quantile(0.25)
Q2 = df.quantile(0.25)
Q3 = df.quantile(0.75)

IQR = Q3 - Q1

lower_bound = IQR - 1.5 * Q1
upper_bound = IQR + 1.5 * Q3


df.loc[df[outlier_column] < lower_bound, outlier_column] = df[outlier_column].mean()
df.loc[df[outlier_column] < lower_bound, outlier_column] = df[outlier_column].median()
df.loc[df[outlier_column] < lower_bound, outlier_column] = df[outlier_column].mode()
```

---

### 3. Transformation
- Description: Math transformations can mitigate the impact of outliers on stat. analyses

#### Concept

```python
import pandas as pd
import numpy as np

df = pd.DataFrame([])

outlier_column = 'col_1'

df[outlier_column] = np.log(df[outlier_column]) 
```

---

### 4. Winsorization
- Description: Replace extreme values with the less closes permissible values i.e. lower and upper bounds

#### Concept

```python
import pandas as pd

df = pd.DataFrame([])

outlier_column = 'col_1'

Q1 = df.quantile(0.25)
Q2 = df.quantile(0.25)
Q3 = df.quantile(0.75)

IQR = Q3 - Q1

lower_bound = IQR - 1.5 * Q1
upper_bound = IQR + 1.5 * Q3

df[outlier_column] = df[outlier_column].clip(lower=lower_bound, upper=upper_bound)
```

---

### 5. Algorithmic Imputation
- Description: Using RobustScaler and KNNImputer we can impute outlier with predicted one

#### Concept

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.impute import KNNImputer

df = pd.DataFrame([])

outlier_column = 'col_1'

Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)

IQR = Q3 - Q1

lower_bound = IQR - 1.5 * Q1
upper_bound = IQR + 1.5 * Q3

# Replace outliers with Null
df[(df >= lower_bound) | (df <= upper_bound )] = np.nan

columns = df.columns

# Apply scaling
scaler = RobustScaler()
scaled_data = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_data, columns=columns)

# Impute missing values
knn_imputer = KNNImputer(n_neighbors=5) # adjust count of neighbors
imputed_data = knn_imputer.fit_transform(scaled_data)

# Inverse the transformation 
original_scale_data = scaler.inverse_transform(imputed_data)
original_scale_df = pd.DataFrame(original_scale_data, columns=scaled_df.columns)
```

### 6. Isolation Forest
- Description: Isolation Forest based on DecisionTree model, and can predict possible anomalies by easy split some examples

#### Concept

```python
from sklearn.ensemble import IsolationForest
import pandas as pd

df = pd.DataFrame([])
anomaly_inputs = ['NPHI', 'RHOB'] # features to detect
model = IsolationForest(contamination=0.01, # threshold which used for decision 
                        random_state=32)
model.fit(df[anomaly_inputs])

df['anomaly_scores'] = model.decision_function(df[anomaly_inputs]) # probability of anomaly
df['anomaly'] = model.predict(df[anomaly_inputs]) # flag field ( 1 for ok, -1 for anomaly )
```