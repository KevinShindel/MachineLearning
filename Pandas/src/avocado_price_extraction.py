"""
Description: Extracting avocado price data from the dataset
Author: Kevin Shindel
Date: 2024-08-05
"""
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np


def main():
    # avocado price prediction
    avocado_df = pd.read_csv('../../dataset/avocado.zip',
                             compression='zip',
                             parse_dates=True,
                             index_col='Date',
                             usecols=['Date', 'AveragePrice', 'region', 'Total Volume', 'year'])
    avocado_df.columns = avocado_df.columns.str.replace(' ', '_').str.lower()

    grouped_avocado_df = avocado_df.groupby(['region']).agg({'averageprice': 'mean', 'total_volume': 'sum'})

    top_10_by_volume_df = (grouped_avocado_df.
                     sort_values(by='total_volume', ascending=False).
                     head(10).
                     reset_index())

    # Pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(
        x=top_10_by_volume_df['total_volume'],
        labels=top_10_by_volume_df['region'],
        autopct='%1.1f%%',
        startangle=140,
        colors=plt.cm.Set3.colors
    )
    plt.title('Top 10 Avocado Regions by Total Volume (%)')
    plt.tight_layout()
    plt.show()


    albany_avocado_df = avocado_df[avocado_df['region'] == 'Albany']
    albany_avocado_df = albany_avocado_df.drop('region', axis=1)

    albany_avocado_df: pd.DataFrame = (albany_avocado_df.
                                       sort_index().
                                       reset_index().
                                       drop_duplicates(subset='Date', keep='last').
                                       set_index('Date').
                                       resample('D').
                                       ffill())

    print(albany_avocado_df.describe())
    print(albany_avocado_df.info())

    albany_avocado_df['AveragePrice'].plot()
    plt.show()


if __name__ == '__main__':
    main()
