"""
Author: Kevin Shindel
Date: 2024-18-04

Description:
"""

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing, HoltWintersResults

from SciKit.italy.utils import adjust_model

MAX_PREDICTION = 4


def create_model(df: pd.DataFrame) -> HoltWintersResults:
    # this function will create a model for the data

    # split historical data into training and testing sets
    max_year = df.index.max().year
    test_max_year = max_year - MAX_PREDICTION

    # split the data into training and testing sets
    test = df.loc[str(test_max_year):]
    train = df.loc[:str(test_max_year - 1)]

    # adjust the model
    params = adjust_model(df)

    # create a model with best params
    model = ExponentialSmoothing(train, trend=params[0],
                                 seasonal=params[1],
                                 seasonal_periods=params[2])

    # fit the model
    model_fit = model.fit()

    return model_fit


def load_data() -> pd.DataFrame:
    # this function will load the data
    df = pd.read_csv('../../dataset/ItalyDisastersDB_1900_2024.csv',
                     usecols=['Start Year', "Total Damage, Adjusted ('000 US$)"],
                     index_col='Start Year')

    # convert string year to date
    df.index = pd.to_datetime(df.index, format='%Y')
    # df.index = df.index.year

    # rename column 'Total Damage, Adjusted ('000 US$)' to 'total_damage' and Start Year to 'year'
    df = df.rename(columns={"Total Damage, Adjusted ('000 US$)": "total_damage", "Start Year": "year"})

    # drop rows with missing values in column 'Total Damage, Adjusted ('000 US$)'
    df = df.dropna(subset=["total_damage"])

    # convert column 'total_damage' to int
    df['total_damage'] = df['total_damage'].astype(int)

    # convert column 'total_damage' to numeric and multiply by 1000
    # df['total_damage'] = df['total_damage'] * 1000

    # group by 'Start Year' and sum 'total_damage'
    df = df.groupby(df.index).sum()

    return df


def forecasting_and_plotting(model: HoltWintersResults, df: pd.DataFrame) -> None:
    # this function will forecast and plot the data

    # forecast the whole historical data
    forecast = model.predict(start=2020, end=2024)

    # convert index into datetime
    forecast.index = pd.to_datetime(forecast.index, format='%Y')

    # plot the historical data and forecast
    plt.figure(figsize=(12, 6))
    plt.plot(df, label='Historical Data')
    plt.plot(forecast, label='Forecast')
    plt.legend()
    plt.title('Historical Data and Forecast')
    plt.xlabel('Year')
    plt.ylabel('Total Damage')
    plt.show()

    # forecast the whole historical data
    forecast = model.predict(start=2020, end=2034)

    # convert index into datetime
    forecast.index = pd.to_datetime(forecast.index, format='%Y')

    # plot the historical data and forecast
    plt.figure(figsize=(12, 6))
    plt.plot(df, label='Historical Data')
    plt.plot(forecast, label='Forecast')
    plt.legend()
    plt.title('Historical Data and Forecast')
    plt.xlabel('Year')
    plt.ylabel('Total Damage')
    plt.show()

    # TODO: This prediction does not make sense, because ML cannot predict catastrophes. It is not possible to predict
    # TODO: Further development does not make sense, because the prediction is not possible. The model is not useful.


if __name__ == '__main__':
    # load the data
    df = load_data()

    # create a model
    model = create_model(df)

    # forecast and plot the data
    forecasting_and_plotting(model, df)
