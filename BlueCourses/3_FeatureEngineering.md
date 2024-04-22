
# Feature Engineering Definition

- Transforming dataset variables to help analytical models achieve better performance it terms 
  of: predictive power and interpretability.
- Simple example: 
  - Date of birth -> Age
- Manual feature engineering versus automated feature engineering.
- The best way to improve the performance of an analytical model is by designing smart features.


## RFM features
- Recency, Frequency, Monetary Value
- Recency: How recently a customer has made a purchase
- Frequency: How often a customer makes a purchase
- Monetary Value: How much money a customer spends on purchases
- Can be operationalized in many ways, e.g.:
  - Recency: Number of days since last purchase
  - Frequency: Number of purchases in the last year
  - Monetary Value: Total spend in the last year
- Useful for behavioral credit scoring
- Very popular in marketing and fraud detection

## Trend Features
- Trends summarize historical evolution
- Examples:
  - Average purchase value in the last 3 months
  - Average number of purchases in the last 6 months
  - Average time between purchases in the last year
- Can be useful for size variables (e.g. number of employees) or time series data
- Beware of denominators equal to zero
- Can put higher weight on recent values
- Extension: time series analysis
- 