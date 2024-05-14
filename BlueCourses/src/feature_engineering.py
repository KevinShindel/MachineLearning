"""
Description: This script contains examples of feature engineering techniques.
Author: Kevin Shindel
Date: 2024-11-05
"""

# import libraries
import matplotlib.pyplot as plt
import pandas as pd
import math
import numpy as np


def boxcox(x, lmbda):
    if lmbda == 0:
        return np.log(x)
    else:
        return (x ** lmbda - 1) / lmbda


def main():
    # RFM feature engineering

    # example of RFM feature engineering
    data = pd.read_csv('data.csv')
    data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
    data['InvoiceDate'] = data['InvoiceDate'].dt.date
    data['TotalSum'] = data['Quantity'] * data['UnitPrice']

    # Calculate Recency, Frequency and Monetary value for each customer
    rfm = data.groupby('CustomerID').agg(
        {'InvoiceDate': lambda x: (data['InvoiceDate'].max() - x.max()).days,
         'InvoiceNo': 'count',
         'TotalSum': 'sum'})

    rfm.rename(columns={'InvoiceDate': 'Recency',
                        'InvoiceNo': 'Frequency',
                        'TotalSum': 'MonetaryValue'}, inplace=True)

    # Create RFM score
    rfm['R'] = pd.qcut(rfm['Recency'], 5, labels=[5, 4, 3, 2, 1])
    rfm['F'] = pd.qcut(rfm['Frequency'], 5, labels=[1, 2, 3, 4, 5])
    rfm['M'] = pd.qcut(rfm['MonetaryValue'], 5, labels=[1, 2, 3, 4, 5])

    rfm['RFM_Score'] = rfm['R'].astype(str) + rfm['F'].astype(str) + rfm['M'].astype(str)

    # Trend feature engineering

    # example of Trend feature engineering
    data = pd.read_csv('data.csv')
    data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
    max_year = data['InvoiceDate'].dt.year.max()
    avg_purchase_3m = data
    avg_purchase_6m = data
    avg_time_bw_purchases = data

    # Calculate average purchase in the last 3 months
    avg_purchase_3m = data[data['InvoiceDate'] > pd.to_datetime(str(max_year) + '-03-01')]
    avg_purchase_3m = avg_purchase_3m.groupby('CustomerID')['TotalSum'].mean()

    # Calculate average purchase in the last 6 months
    avg_purchase_6m = data[data['InvoiceDate'] > pd.to_datetime(str(max_year) + '-06-01')]
    avg_purchase_6m = avg_purchase_6m.groupby('CustomerID')['TotalSum'].mean()

    # Calculate average time between purchases
    data = data.sort_values(by=['CustomerID', 'InvoiceDate'])
    data['TimeDiff'] = data.groupby('CustomerID')['InvoiceDate'].diff().dt.days
    avg_time_bw_purchases = data.groupby('CustomerID')['TimeDiff'].mean()
    print(avg_time_bw_purchases)

    # TODO: Create example of Logarithm feature engineering

    # TODO: Create example of Power feature engineering

    # example of Box-Cox Transformation
    hmeq = pd.read_csv('hmeq.csv')

    hmeq['BOXCOX_LOAN'] = boxcox(hmeq['LOAN'], 0.5)

    hmeq.hist(column='LOAN', bins=50)
    hmeq.hist(column='BOXCOX_LOAN', bins=50)
    plt.show()

    # TODO: Create example of Yeo Johnson Transformation

    # TODO: Create example of Performance Optimization

    # TODO: Principal Component Analysis (PCA)

    # TODO: Create example t-SNE feature engineering


if __name__ == '__main__':
    main()
