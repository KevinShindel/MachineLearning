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

## Undersampling:
- Good customers in trading set are removed

## Oversampling:
- Bad customers in trading set are duplicated

## Trial and error to determine optimal odds for under and over-sampling
## Depend on analytical technique used (logistic regression, decision tree, neural network, etc.)


Example of original data table:

| ID | Age | Income | Savings | CreditScore | Default |
|----|-----|--------|---------|-------------|---------|
| 1  | 25  | 50000  | 10000   | 600         | 0       |
| 2  | 35  | 75000  | 20000   | 700         | 0       |
| 3  | 45  | 100000 | 30000   | 800         | 1       |
| 4  | 55  | 150000 | 40000   | 900         | 1       |
| 5  | 65  | 200000 | 50000   | 1000        | 1       |

Example of Under-sampled data table (data loss):

| ID | Age | Income | Savings | CreditScore | Default |
|----|-----|--------|---------|-------------|---------|
| 1  | 25  | 50000  | 10000   | 600         | 0       |
| 3  | 45  | 100000 | 30000   | 800         | 1       |
| 5  | 65  | 200000 | 50000   | 1000        | 1       |
| 9  | 25  | 50000  | 10000   | 600         | 0       |

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

### Dataset with missing values

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

# Reasons for missing values
- Non-applicable (default date not known fpr non defaulters)
- Non disclosed (income not disclosed)
- Error when merging data (data not available for some records)

# Deal with missing values
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