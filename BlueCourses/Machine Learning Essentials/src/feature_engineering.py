"""
Description: This script contains examples of feature engineering techniques.
Author: Kevin Shindel
Date: 2024-11-05
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def boxcox(x, lmbda):
    if lmbda == 0:
        return np.log(x)
    else:
        return (x**lmbda - 1) / lmbda


def main():
    # RFM feature engineering

    # example of RFM feature engineering
    data = pd.read_csv("data.csv")
    data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"])
    data["InvoiceDate"] = data["InvoiceDate"].dt.date
    data["TotalSum"] = data["Quantity"] * data["UnitPrice"]

    # Calculate Recency, Frequency and Monetary value for each customer
    rfm = data.groupby("CustomerID").agg(
        {
            "InvoiceDate": lambda x: (data["InvoiceDate"].max() - x.max()).days,
            "InvoiceNo": "count",
            "TotalSum": "sum",
        }
    )

    rfm.rename(
        columns={
            "InvoiceDate": "Recency",
            "InvoiceNo": "Frequency",
            "TotalSum": "MonetaryValue",
        },
        inplace=True,
    )

    # Create RFM score
    rfm["R"] = pd.qcut(rfm["Recency"], 5, labels=[5, 4, 3, 2, 1])
    rfm["F"] = pd.qcut(rfm["Frequency"], 5, labels=[1, 2, 3, 4, 5])
    rfm["M"] = pd.qcut(rfm["MonetaryValue"], 5, labels=[1, 2, 3, 4, 5])

    rfm["RFM_Score"] = (
        rfm["R"].astype(str) + rfm["F"].astype(str) + rfm["M"].astype(str)
    )

    # Trend feature engineering
    # example of Trend feature engineering
    data = pd.read_csv("data.csv")
    data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"])

    # Calculate average time between purchases
    data = data.sort_values(by=["CustomerID", "InvoiceDate"])
    data["TimeDiff"] = data.groupby("CustomerID")["InvoiceDate"].diff().dt.days
    avg_time_bw_purchases = data.groupby("CustomerID")["TimeDiff"].mean()
    print(avg_time_bw_purchases)

    hmeq = pd.read_csv("hmeq.csv")

    # example of Box-Cox Transformation
    hmeq["BOXCOX_LOAN"] = boxcox(hmeq["LOAN"], 0.5)
    # example of Power feature engineering
    hmeq["PRW_LOAN"] = np.power(hmeq["LOAN"], 2)
    # example of Log feature engineering
    hmeq["LOG_LOAN"] = np.log1p(hmeq["LOAN"])

    hmeq.hist(column="LOAN", bins=50)
    hmeq.hist(column="BOXCOX_LOAN", bins=50)
    plt.show()


if __name__ == "__main__":
    main()
