# Association Rules
- Purpose: detect frequent patterns in data
- Descriptive analysis: not target variable
- Example applications: Market basket analysis, web click stream analysis, intrusion detection, bioinformatics, etc.

# Support and Confidence
- Support: fraction of transactions that contain all items in the rule
- Confidence: fraction of transactions that contain all items in the antecedent also contain the consequent
- Lift: ratio of observed support to that expected if X and Y were independent
- Conviction: ratio of expected support to observed support if X and Y were independent
- Examples: 
- - If a customer has a car loan and car insurance, then customer has checked an account in 80% of cases
- - If a customer buys spaghetti, then customer buys red wine in 70% of cases

| Transaction | Items                                 |
|-------------|---------------------------------------|
| 1           | {bread, milk}                         |
| 2           | {bread, butter, milk}                 |
| 3           | {water, beer, baby food}              |
| 4           | {coke, beer, diapers}                 |
| 5           | {spaghetti, diapers, baby food, beer} |
| 6           | {apples, wine, baby food}             |

- Support (X -> Y) = number of transactions containing X and Y / total number of transactions
- Confidence (X -> Y) = number of transactions containing X and Y / number of transactions containing X
- Lift (X -> Y) = Support (X -> Y) / (Support (X) * Support (Y))

- Item set {bread, milk} has support 2/6 = 0.33
- Association rule {bread} -> {milk} has confidence 2/3 = 0.67

# Association Rule Mining
- Step 1: Find all frequent item sets
- Step 2: Discovery of all derived association rules having confidence above a given threshold
- Apriori property: every subset of a frequent item set must be frequent
- Once frequent item has been found, association rules can be generated in a straightforward manner:
- - For each frequent item set, generate all non-empty subsets
- - For every non-empty subset, generate the rule that includes the subset in the antecedent and the rest in the consequent

# Association rule mining in Python
```python
# Importing the libraries
import pandas as pd

# We're using the mlxtend package here
# Install with pip install mlxtend
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

dataset = [
    ['Spaghetti', 'Tomato Sauce', 'Beer', 'Diapers', 'Wine'],
    ['Spaghetti', 'Grated Cheese', 'Wine'],
    ['Beer', 'Diapers'],
    ['Spaghetti', 'Grated Cheese', 'Bread'],
    ['Spaghetti', 'Tomato Sauce', 'Bread', 'Wine'],
    ['Spaghetti', 'Beer', 'Wine'],
    ['Spaghetti', 'Bread', 'Grated Cheese', 'Wine'],
    ['Grated Cheese', 'Wine', 'Beer'],
    ['Beer', 'Wine'],
]

trans_enc = TransactionEncoder()
te_ary = trans_enc.fit(dataset).transform(dataset)

df = pd.DataFrame(te_ary, columns=trans_enc.columns_)
frequent_item_sets = apriori(df, min_support=0.3, use_colnames=True)
print(frequent_item_sets)

rules = association_rules(frequent_item_sets, metric="confidence", min_threshold=0.6)
print(rules)
```


# Lift
- Lift is a measure of how much more likely item Y is to be bought when item X is bought, compared to when item Y is bought independently of item X
- Lift = 1: X and Y are independent (no association)
- Lift > 1: X and Y are positively correlated (complementary effect)
- Lift < 1: X and Y are negatively correlated (substitution effect)
- Lift = Support (X -> Y) / (Support (X) * Support (Y))

|            | Tea | Not Tea | Total | 
|------------|-----|---------|-------|
| Coffee     | 150 | 750     | 900   |
| Not Coffee | 50  | 50      | 100   |
| Total      | 200 | 800     | 1000  |

Lift = (150/1000) / (900/1000 * 200/1000) = 1.67

# Association Rule Extension
- Include item quantities and/or prices
- Absence of items
- Multilevel association rules

# Post-Processing Association Rules
- Filter out trivial rules
- Sensitivity analysis
- Visualization
- Measure economic impact

# Association Rule Applications
- Detect which products are frequently bought together
- Implications for: targeted marketing, store layout, inventory management, etc.
- Fraud detection

# Sequence Rules
- Find maximal sequences among all sequences in the data that have minimum support and confidence
- Order of items is important
- Example: Home -> Electronics -> Kitchen -> Garden
- Transaction time or sequence field
- Inter-transaction patterns 