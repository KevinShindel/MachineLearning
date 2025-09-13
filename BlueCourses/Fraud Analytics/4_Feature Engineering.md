## Feature Engineering
- Transform data set variables into features.
- Engineer additional smart features that help detect fraud.
- Time feature: analyze timestamps and create features that indicate suspicious timing patterns.
- RFM features: Recency, Frequency, Monetary value of transactions.

## Time feature
- Certain events are expected to occur at specific times.
- Aim: capture info about time aspect by meaningful features.
- Dealing with time can be tricky: time zones, daylight saving time, different formats.
- Do not use arithmetic mean to compute average time.
- To compute avg. time use **Mises-Fisher** distribution.
- Periodic normal distribution.

## Recency feature
- recency = current_date - last_transaction_date

## Frequency feature
- how often a customer makes a purchase per time period

## Monetary feature
- Monetary: intensity of transactions  ( e.g. amount spent)
- Operationalized as :  Average, Sum, Max, Min, Std. Dev.

## Featurization
- Frequancy_auth = count of auth transactions per user
- Rec_auth = days since last auth transaction per user
- Mon_auth = avg amount of auth transactions per user
