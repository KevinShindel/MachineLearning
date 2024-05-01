## Type of Variables

- Continuous:
- - Defined on continuous interval [0, 1], [0, 100], [0, 100000]
- - Income, savings balance, credit score [0, 1000], [0, 100000], [0, 100]
- Categorical:
- - Binary:
- - - 0 or 1, male | female, true | false 
- - Nominal:
- - - No order, color, country, occupation [red, green, blue], [USA, UK, India]
- - Ordinal:
- - - Order, low | medium | high, 1st class | 2nd class | 3rd class [low, medium, high], [1st class, 2nd class, 3rd class]
- - - Purpose of loan, industry sector, martial status [single, married, divorced] [student, self-employed, employed, retired]

## De-normalizing Data
- Analytics requires all data in single table
- - Rows = instances, observations, lines
- - Columns = variables, features, attributes
- De-normalizing data
- - merge normalized data into single table
- - - Join tables on common key

## Sampling

- Take sample of past data to build analytics model
- Timing of sample:
- - How far back to go?
- - Trade-off many data versus recent data
- Sample taken must be from normal business period to get  as accurate a picture as possible of target population

Example of original data table:

| ID | Age | Income | Savings | CreditScore | Default |
|----|-----|--------|---------|-------------|---------|
| 1  | 25  | 50000  | 10000   | 600         | 0       |
| 2  | 35  | 75000  | 20000   | 700         | 0       |
| 3  | 45  | 100000 | 30000   | 800         | 1       |
| 4  | 55  | 150000 | 40000   | 900         | 1       |
| 5  | 65  | 200000 | 50000   | 1000        | 1       |


## Under-sampling:
- Good customers in trading set are removed

Example of Under-sampled data table (data loss):

| ID | Age | Income | Savings | CreditScore | Default |
|----|-----|--------|---------|-------------|---------|
| 1  | 25  | 50000  | 10000   | 600         | 0       |
| 3  | 45  | 100000 | 30000   | 800         | 1       |
| 5  | 65  | 200000 | 50000   | 1000        | 1       |
| 9  | 25  | 50000  | 10000   | 600         | 0       |

## Oversampling:
- Bad customers in trading set are duplicated

Example of Oversampled data table (data duplication):

| ID | Age | Income | Savings | CreditScore | Default |
|----|-----|--------|---------|-------------|---------|
| 1  | 25  | 50000  | 10000   | 600         | 0       |
| 1  | 25  | 50000  | 10000   | 600         | 0       |
| 3  | 45  | 100000 | 30000   | 800         | 1       |
| 3  | 45  | 100000 | 30000   | 800         | 1       |
| 3  | 45  | 100000 | 30000   | 800         | 1       |
| 5  | 65  | 200000 | 50000   | 1000        | 1       |
| 5  | 65  | 200000 | 50000   | 1000        | 1       |

## Trial and error to determine optimal odds for under and over-sampling

### Determine optimal odds for under and over-sampling example:
```python
import pandas as pd

# Read data from CSV file
hmeq = pd.read_csv("hmeq.csv")

# Count number of good and bad customers
good = hmeq[hmeq["BAD"] == 0].shape[0]
bad = hmeq[hmeq["BAD"] == 1].shape[0]

# Determine optimal odds for under and over-sampling
odds = good / bad

# Under-sampling
under_sample = hmeq[hmeq["BAD"] == 0].sample(n=bad, replace=False)

# Over-sampling
over_sample = hmeq[hmeq["BAD"] == 1].sample(n=good, replace=True)

# Combine under and over-sampled data
sample = pd.concat([under_sample, over_sample], axis=0)
```

## Depend on analytical technique used (logistic regression, decision tree, neural network, etc.)

### SMOTE technique: Synthetic Minority Over-sampling Technique

- Step 1: For each minority class observation, find k nearest neighbors
- Step 2: Choose one of the k nearest neighbors and create a synthetic observation

### Example of SMOTE technique:

**Tim data**

| Amount | Ratio | 
|--------|-------|
| 2800   | 0.79  |

**Bard Data**

| Amount | Ratio |
|--------|-------|
| 1000   | 0.21  |

**SMOTE**
0.6 - is random number between 0 and 1

Amount = 2800 + 0.6 * (1000-2800) = 1720
Ratio = 0.79 + 0.6 * (0.21-0.79) = 0.55

| Amount | Ratio |
|--------|-------|
| 1720   | 0.55  |


### Sampling in Python 

```python
import pandas as pd
# Read data from CSV file
hmeq = pd.read_csv("hmeq.csv")

# Seed is specified using random_state argument
my_sample = hmeq.sample(n=1000, replace=False, random_state=12345)
```

## Visual Data Exploration

- Histograms
- Bar charts
- Scatter plots

### Visual Data Exploration in Python

```python
import matplotlib.pyplot as plt
import pandas as pd

# Read data from CSV file
mortgage = pd.read_csv("mortgage.csv")

mortgage.hist(column='FICO_orig_time', bins=100, grid=False, normed=True)
plt.title("Histogram of FICO_orig_time")
plt.xlabel("FICO_orig_time")
plt.ylabel("Density")
plt.show()
```

## Descriptive Statistics

- Mean
- Median
- Mode
- Standard deviation
- Variance
- Percentile values

## Missing Values

### Example of Dataset with missing values

| ID | Age| Income | Marital Status | Credit Score | Class |
|----|----|--------|----------------|--------------|-------|
| 1  | 25 | 50000  | Single         | 600          | Bad   |
| 2  | 35 | 75000  | Married        | 700          | Good  |
| 3  | 45 | 100000 | ?              | 800          | Good  |
| 4  | 55 | ?      | Divorced       | 900          | Bad   |
| 5  | 65 | 200000 | Single         | ?            | Good  |
| 6  | 75 | 250000 | ?              | ?            | Good  |
| 7  | 85 | 300000 | Divorced       | ?            | Good  |
| 8  | 95 | ?      | Single         | 1300         | Good  |
| 9  | 105| 400000 | Married        | 1400         | Bad   |
| 10 | 115| 450000 | Divorced       | 1500         | Good  |

### Reasons for missing values
- Non-applicable (default date not known fpr non defaulters)
- Non disclosed (income not disclosed)
- Error when merging data (data not available for some records)

### Deal with missing values
- Keep
- - Fact that data is missing can indicate pattern
- - Add additional category for missing value
- - Add additional missing value indicator variable (1 = missing, 0 = not missing, one per variable or one per record)
- Delete
- - Remove records with missing values
- - When too missing values, removing variable or record might be only option
- Replace (Impute)
- - Estimating missing values using imputation techniques
- - Be consistent when treating missing values during the model development and during the model usage

### Imputation Techniques

#### For Continuous Variables
- Replace with mean/median/mode (median more robust to outliers)
- If missing values only occur during the model development, can also replace with mean/median/mode of all records of the same class

#### For Categorical Variables
- Replace with mode (most frequent value)
- If missing values only occur during the model development, can also replace with mode of all records of the same class

#### Regression Imputation
- predict missing values using other variables
- If missing values only occur during the model using, you cannot use the target class as a predictor

```python
import pandas as pd

# Read data from CSV file
my_sample = pd.read_csv("my_sample.csv")

my_sample_na_mean = my_sample.fillna(value=my_sample.mean())
```

## Outliers

- Extreme or unusual observations
- - Due to recording, data entry errors or noise
- Types of outliers
- - Invalid observation: age = 200
- - Valid Observation: CEO salary = $ 1,000,000
- Detection vs treatment

### To detect outliers use box plots

```python
import matplotlib.pyplot as plt

# Read data from CSV file
my_sample = pd.read_csv("my_sample.csv")

my_sample.boxplot(column='FICO_orig_time')
plt.title("Boxplot of FICO_orig_time")
plt.show()
```
### Implementation of IQR method in Python

- Median M: P(x <= M) = 0.5
- Lower quartile Q1: P(x <= Q1) = 0.25
- Upper quartile Q3: P(x <= Q3) = 0.75
- Interquartile range IQR = Q3 - Q1
- Minimum = Q1 - 1.5 * IQR
- Maximum = Q3 + 1.5 * IQR
- Outliers: x < Minimum or x > Maximum
- Z-score: (x - mean) / standard deviation

```python
import pandas as pd
import numpy as np

df = pd.read_csv('c:/temp/hmeq.csv')
q1 = df.quantile(0.25)
q3 = df.quantile(0.75)
iqr = q3 - q1
# calculate maximum and minimum
maximum = q3 + 1.5 * iqr
minimum = q1 - 1.5 * iqr
# find outliers
df = df[(df < minimum) & (df > maximum)]

outlier_exist = np.all(df.isnull())
print(f'Outliers exist: {not outlier_exist}')
```


## Treatment of Outliers
- For invalid outliners:
- - E.g age = 300 years
- - Treat as missing value (delete, keep, replace)
- For valid outliers:
- - Based on z-score:
- - - Replace all variable values with z-score > 3 with mean +3 times standard deviation
- - - Replace all variable values with z-score < -3 with mean -3 times standard deviation
- Expert based limits

### Outliers in Python

```python
import pandas as pd
import numpy as np
from scipy import stats


my_sample = pd.read_csv("my_sample.csv")
my_sample_na_mean = my_sample.fillna(value=my_sample.mean())

# Select all numeric variables
num_variables = my_sample_na_mean \
    .select_dtypes(include=[np.number]).columns.tolist()

# Remove the target variable if selected
num_variables.remove("BAD")

# Compute z-scores for numeric variables
my_sample_cnt = my_sample_na_mean[num_variables]
z_scores = my_sample_cnt.apply(stats.zscore)
max_abs = z_scores.apply(lambda x: max(abs(x)) < 3, axis="columns")
filtered_sample = z_scores.loc[max_abs, :]
```

## Categorization

- Also called coarse classification, classing, binning, grouping
- Categorical variables:
- - Purpose of loan=second hand car, first-hand car, travel cash: high risk
- - Purpose of loan=study, new car, furniture: medium risk
- - Purpose of loan=house, wedding: low risk
- Continuous variables:
- - Age < 30 years: young
- - Age >= 30 and < 60 years: middle-aged
- - Age >= 60 years: old

```python
import pandas as pd
hmeq = pd.read_csv('c:/temp/hmeq.csv')
# Do equal frequency binning of the CLAGE variable based on the quartiles
hmeq["CLAGE_cat"] = pd.qcut(hmeq.CLAGE, q=4, labels=False)
```

### Transform string to categorical variable in Python

| CustomerID | Age | Purpose | G/B  |
|------------|-----|---------|------|
| C1         | 44  | car     | Good |
| C2         | 33  | house   | Bad  |
| C3         | 55  | car     | Good |
| C4         | 22  | study   | Good |
| C5         | 66  | travel  | Bad  |

1. Code numerically (car=1, house=2, study=3, travel=4)
2. Create dummy variables (car=1, house=0, study=0, travel=0)

```python
import pandas as pd

# Read data from CSV file
my_sample = pd.read_csv("my_sample.csv")

# Create dummy variables
my_sample_dummies = pd.get_dummies(my_sample, columns=["Purpose"])

# transform string to categorical variable
my_sample["G/B"] = my_sample["G/B"].astype('category')
```

Sample of table after transformation with dummy variables:

| CustomerID | Age | G/B  | Purpose_car | Purpose_house | Purpose_study | Purpose_travel |
|------------|-----|------|-------------|---------------|---------------|----------------|
| C1         | 44  | Good | 1           | 0             | 0             | 0              |
| C2         | 33  | Bad  | 0           | 1             | 0             | 0              |
| C3         | 55  | Good | 1           | 0             | 0             | 0              |
| C4         | 22  | Good | 0           | 0             | 1             | 0              |
| C5         | 66  | Bad  | 0           | 0             | 0             | 1              |

Sample of table after transformation with numerical coding:

| CustomerID | Age | G/B  | Purpose |
|------------|-----|------|---------|
| C1         | 44  | Good | 1       |
| C2         | 33  | Bad  | 2       |
| C3         | 55  | Good | 1       |
| C4         | 22  | Good | 3       |
| C5         | 66  | Bad  | 4       |


## Five popular methods of categorical variable encoding
1. Binning
2. Pivot tables
3. Chi-squared analysis
4. Business logic
5. Decision trees

## Binning introduction

> Income variable: 1000, 1200, 1300, 2000, 1800, 1400
> Equal interval binning
> - Bin width = (max - min) / number of bins = (2000 - 1000) / 3 = 333.33 = 334
> - Bin 1: 1000 - 1333 = 1000, 1200, 1300
> - Bin 2: 1334 - 1667 = 1400, 1500
> - Bin 3: 1668 - 2000 = 1800
> - Bin 4: 2001 - 2334 = 2000
> Equal frequency binning (histogram equalization)
> - 2 bins: 1000, 1200, 1300, 1400, 1500
> - Bin 1: 1000, 1200, 1300
> - Bin 2: 1400, 1500, 1800, 2000
> However, binning can lead to loss of information


### Pivot tables introduction

> Pivot table: Purpose of loan vs. default
> - Purpose of loan: car, house, study, travel
> - Default: good, bad
> - Pivot table: count of good and bad defaults for each purpose of loan
> - Purpose of loan: car, house, study, travel
> - Good: 100, 200, 300, 400
> - Bad: 50, 100, 150, 200

#### Pivot table example in Python

| CustomerID | Age | Purpose | G/B  |
|------------|-----|---------|------|
| C1         | 44  | car     | Good |
| C2         | 33  | house   | Bad  |
| C3         | 55  | car     | Good |
| C4         | 22  | study   | Good |
| C5         | 66  | travel  | Bad  |

| Purpose | Good | Bad |
|---------|------|-----|
| car     | 2    | 1   |
| house   | 0    | 1   |
| study   | 1    | 0   |
| travel  | 0    | 1   |


```python
import pandas as pd

# Read data from CSV file
my_sample = pd.read_csv("my_sample.csv")

pivot_table = pd.pivot_table(my_sample, values='Age', index='Purpose', columns='G/B', aggfunc='count')
```

## Chi-squared analysis introduction

> Chi-squared analysis: Purpose of loan vs. default
> - Purpose of loan: car, house, study, travel
> - Default: good, bad
> - Chi-squared analysis: test if purpose of loan is independent of default

### Chi-squared analysis table (contingency table)

| Purpose | Good | Bad |
|---------|------|-----|
| car     | 2    | 1   |
| house   | 0    | 1   |
| study   | 1    | 0   |
| travel  | 0    | 1   |


### Chi-squared analysis example in Python

```python
import pandas as pd
from scipy.stats import chi2_contingency

# Read data from CSV file
my_sample = pd.read_csv("my_sample.csv")

# Create contingency table
contingency_table = pd.crosstab(my_sample['Purpose'], my_sample['G/B'])

# Perform chi-squared test
chi2, p, dof, expected = chi2_contingency(contingency_table)
```

## TODO: Business logic introduction

## Weight of Evidence (WOE) and Information Value (IV)

- Categorization reduces number of dummy variables needed for categorical variables
- Categorization introduces new dummy variables for continuous variables
- Steel need a lot of dummy variables for continuous variables
- WOE codding

> WOE = ln(p_good / p_bad)
> p_good = num_good / cnt_total
> p_bad = num_bad / cnt_total
> if p_good > p_bad, WOE > 0
> if p_good < p_bad, WOE < 0


### Example of WOE table (from negative WOE to positive)

| Age     | Count | Distr.Count | Goods | Distr.Goods | Bads | Distr.Bads | WOE   |
|---------|-------|-------------|-------|-------------|------|------------|-------|
| 20-30   | 100   | 0.1         | 90    | 0.3         | 10   | 0.05       | 1.39  |
| 30-40   | 200   | 0.2         | 150   | 0.5         | 50   | 0.25       | 0.58  |
| 40-50   | 300   | 0.3         | 200   | 0.6         | 100  | 0.5        | 0.18  |
| 50-60   | 200   | 0.2         | 100   | 0.3         | 100  | 0.5        | -0.41 |
| 60-70   | 100   | 0.1         | 50    | 0.1         | 50   | 0.25       | -0.92 |
| 70-80   | 50    | 0.05        | 20    | 0.05        | 30   | 0.15       | -1.39 |

### Information Value (IV) table (from negative IV to positive)

| Age     | Goods | Bads | Distr.Goods | Distr.Bads | WOE   | IV   |
|---------|-------|------|-------------|------------|-------|------|
| Missing | 42    | 58   | 0.14        | 0.29       | -0.71 | 0.11 |
| 20-30   | 90    | 10   | 0.3         | 0.05       | 1.39  | 0.34 |
| 30-40   | 150   | 50   | 0.5         | 0.25       | 0.58  | 0.15 |
| 40-50   | 200   | 100  | 0.6         | 0.5        | 0.18  | 0.02 |
| 50-60   | 100   | 100  | 0.3         | 0.5        | -0.41 | 0.08 |
| 60-70   | 50    | 50   | 0.1         | 0.25       | -0.92 | 0.14 |
| 70-80   | 20    | 30   | 0.05        | 0.15       | -1.39 | 0.21 |

IV = sum((p_good - p_bad) * WOE) = 0.11 + 0.34 + 0.15 + 0.02 + 0.08 + 0.14 + 0.21 = 1.05

Rule of thumb:
- IV < 0.02: not useful for prediction
- IV 0.02 - 0.1: weak predictor
- IV 0.1 - 0.3: medium predictor
- IV 0.3 - 0.5: strong predictor
- IV > 0.5: suspicious or too good to be true

IV: measure of predictive power used to 
- Access appropriateness of categorization
- Do variable selection

Category boundaries can be adjusted to maximize IV

## Numbers of categories ?

- Trade-off: 
- - Fewer categories: less overfitting, less predictive power, because of simplicity, 
    interpretability and stability
- - More categories to keep predictive power, but risk of overfitting
- Practical: perform sensitivity analysis:
- - IV versus number of categories
- - Decide on cut-off: IV > 0.1, IV > 0.2, IV > 0.3, elbow point?
- Note: fever values in categories, less reliable/robust/stable WOE values

### Laplace smoothing
n = 0.5 # smoothing factor
WOE = ln((num_good + n) / (num_bad + n))
- Larger n?
- - less reliance on data
- - WOE closer to 0, no increased or decreased risk

### WOE and IV in Python

```python
import pandas as pd
import numpy as np

# Calculate information value
def calc_iv(df, feature, target, pr=False):
    """
    Set pr=True to enable printing of output.
    
    Output: 
      * iv: float,
      * data: pandas.DataFrame
    """

    lst = []

    df[feature] = df[feature].fillna("NULL")

    for i in range(df[feature].nunique()):
        val = list(df[feature].unique())[i]
        lst.append([feature,                                                        # Variable
                    val,                                                            # Value
                    df[df[feature] == val].count()[feature],                        # All
                    df[(df[feature] == val) & (df[target] == 0)].count()[feature],  # Good (think: Fraud == 0)
                    df[(df[feature] == val) & (df[target] == 1)].count()[feature]]) # Bad (think: Fraud == 1)

    data = pd.DataFrame(lst, columns=['Variable', 'Value', 'All', 'Good', 'Bad'])

    data['Share'] = data['All'] / data['All'].sum()
    data['Bad Rate'] = data['Bad'] / data['All']
    data['Distribution Good'] = (data['All'] - data['Bad']) / (data['All'].sum() - data['Bad'].sum())
    data['Distribution Bad'] = data['Bad'] / data['Bad'].sum()
    data['WoE'] = np.log(data['Distribution Good'] / data['Distribution Bad'])

    data = data.replace({'WoE': {np.inf: 0, -np.inf: 0}})

    data['IV'] = data['WoE'] * (data['Distribution Good'] - data['Distribution Bad'])

    data = data.sort_values(by=['Variable', 'Value'], ascending=[True, True])
    data.index = range(len(data.index))

    if pr:
        print(data)
        print('IV = ', data['IV'].sum())


    iv = data['IV'].sum()
    # print(iv)

    return iv, data
```

```python
import pandas as pd

hmeq = pd.read_csv('c:/temp/hmeq.csv')

# Categorize variable in 10 bins based on distribution
hmeq["CLAGE_cat"] = pd.qcut(hmeq.CLAGE, q=10, labels=False)

# Run calc_IV function from 
# https://www.kaggle.com/puremath86/iv-woe-starter-for-python
calc_iv(hmeq, "CLAGE_cat", "BAD")
```