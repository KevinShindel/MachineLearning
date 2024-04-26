"""
Author: Kevin Shindel
Date: 2024-18-04

Description: This script is used to forecast the temperature anomaly and CO2 emissions for the years 2024-2028.
The script uses the Holt-Winters Exponential Smoothing model to forecast the temperature anomaly and CO2 emissions.
The script also calculates the insurance rate for each year based on the temperature anomaly.
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')


def forecast_warming():
    df = pd.read_csv('../../dataset/temperature_anomaly.csv',
                     index_col='year',
                     parse_dates=True,
                     usecols=['year', 'avg_anomaly_temp'])

    # Add a constant to make all values positive
    df['avg_anomaly_temp'] = df['avg_anomaly_temp'] + abs(df['avg_anomaly_temp'].min()) + 1

    # Train the model on data up to 2024
    train = df.loc[:'2020', 'avg_anomaly_temp']
    model = ExponentialSmoothing(train, seasonal_periods=3, trend='mul', seasonal='add').fit()

    # evaluate on test data
    test = df.loc['2020':'2024', 'avg_anomaly_temp']
    predictions = model.predict(start='2020', end='2024')
    mse = mean_squared_error(test, predictions)
    print(f'Mean Squared Error: {mse}')

    # 3. Plot test data vs forecasted data
    test_forecast = model.predict(start='1860', end='2022')
    df.plot(label='Real Warming')
    test_forecast.plot(label='Predicted Warming')
    plt.ylabel('Temperature Anomaly in Celsius')
    plt.xlabel('Year')
    plt.title('Global warming anomaly Predicted vs Real from 1860 to 2022')
    plt.legend()
    plt.show()

    # 4. Plot the forecasted data
    forecast = model.predict(start='2024', end='2028')
    df['2020':'2024'].plot(label='Real Warming')
    forecast.plot(label='predicted Warming')
    plt.xlabel('Year')
    plt.ylabel('Temperature Anomaly in Celsius')
    plt.title('Global warming anomaly Real and Predicted for 2023-2028')
    plt.legend()
    plt.show()


    forecasted_df = pd.DataFrame(forecast,
                                 columns=['avg_anomaly_temp'],
                                 )
    # union forecasted and real data
    full_df = pd.concat([df, forecasted_df['2025':'2028']])
    return full_df


def forecast_emissions():
    # Read the data
    df = pd.read_csv('../../dataset/co2_emission.csv',
                     index_col='year',
                     parse_dates=True,
                     usecols=['year', 'emissions'])

    # Train the model on data up to 2022
    train = df.loc[:'2022', 'emissions']
    model = ExponentialSmoothing(train, seasonal_periods=3, trend='mul', seasonal='add').fit()

    # Forecast the emissions for 2021-2026
    forecast = model.predict(start='2023', end='2028')
    test_forecast = model.predict(start='1860', end='2022')

    # Calculate the mean squared error
    mse = mean_squared_error(df, test_forecast)
    print(f'Mean Squared Error: {mse}')

    # 1. Plot test data vs forecasted data
    df.plot(label='Real Emissions')
    test_forecast.plot(label='Predicted Emissions')
    plt.ylabel('Emission')
    plt.xlabel('Year')
    plt.title('CO2 emission Predicted vs Real from 1860 to 2022')
    plt.legend()
    plt.show()

    # 2. Create plot with real data and forecasted data
    df['2020':'2022'].plot(label='Real Emissions')
    forecast.plot(label='Predicted Emissions')
    plt.xlabel('Year')
    plt.ylabel('Emission')
    plt.title('CO2 emission Real and Forecasted from 2020 to 2028')
    plt.legend()
    plt.show()

    # union forecasted and real data
    forecasted_df = pd.DataFrame(forecast,
                                 columns=['emissions'],
                                 index=pd.date_range('2023', periods=6, freq='AS')
                                 )
    full_df = pd.concat([df, forecasted_df])
    return full_df


def forecast_claims(e_df=None, w_df=None):
    # Calculate the claims based on the temperature anomaly

    # read gross and clean insurance index from historical data
    claims_df = pd.read_csv('../../dataset/IT_claims_2004_2020.csv',
                            index_col='year', parse_dates=True, thousands=',')

    # check correlations between gross and clean index
    corr = claims_df['gross'].corr(claims_df['clean'])  # correlation is weak: 0.074
    print(f'Correlation between gross and clean index: {corr}')

    # we have an n/a value in dataframe lest fill with mean
    claims_df['clean'] = claims_df['clean'].fillna(claims_df['clean'].mean())

    # join warming, emissions and claims in one frame
    all_in_df = claims_df.join([w_df, e_df], how='left')

    # lest watch the correlation between warming/emissions and gross
    corr1 = all_in_df['emissions'].corr(all_in_df['gross']) # negative correlation: -0.477 - means that as emissions grow, gross index decreases
    corr2 = all_in_df['avg_anomaly_temp'].corr(all_in_df['gross']) # positive correlation 0.26 - means that as temperature grows, gross index grows

    print('Correlation between CO2 emissions and gross: ', corr1)
    print('Correlation between Warming and gross: ', corr2)


    # 6. Correlation matrix heatmap between warming, emissions and gross
    sns.heatmap(all_in_df[['emissions', 'gross']].corr(), annot=True)
    plt.title('Correlation between Gross index and Emissions for 2004 – 2020',
              x=0.6)
    plt.tight_layout()
    plt.show()

    # we found that correlation between emissions and gross index is negative, so we need cant predict gross index based on emissions and warming

    # let`s predict gross index only by historical data
    gross_df = all_in_df['gross']
    x_train, x_test = gross_df.loc[:'2015'], gross_df.loc['2016':]

    model = ExponentialSmoothing(x_train, seasonal_periods=3, trend='mul', seasonal='add').fit()

    # lest predict gross from 2004 to 2020
    forecast = model.predict(start='2004', end='2020')

    forecast = pd.DataFrame(forecast, columns=['gross'])
    forecast.index.name = 'year'

    gross_df = gross_df.to_frame()

    # 7. show plot forecasted vs real data
    gross_df['gross'].plot(label='Real')
    forecast['gross'].plot(label='Predicted')
    plt.xlabel('Year')
    plt.ylabel('Gross Index')
    plt.legend()
    plt.title('Gross claims expenditure in Italy Predicted vs Real for 2004-2020')
    plt.tight_layout()
    plt.show()

    # Forecast the gross for 2016-2020
    forecast = model.predict(start='2016', end='2020')

    # calc mse
    mse = mean_squared_error(x_test, forecast)
    print(f'Mean Squared Error: {mse}')  #

    # Forecast the gross for 2021-2028
    forecast = model.predict(start='2021', end='2028')

    # 8. Plot the forecasted data
    forecast.plot(label='Forecasted Gross')
    plt.xlabel('Year')
    plt.ylabel('Gross Index')
    plt.title('Gross Claims Expenditures Predicted 2020 – 2028 based on the historical data of Gross Claims Expenditures 2004 – 2020',
              fontsize=7, x=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    emission_df = forecast_emissions()
    warming_df = forecast_warming()

    # find correlation between co2 emissions and global warming
    corr = emission_df['emissions'].corr(warming_df['avg_anomaly_temp'])
    print('Correlation between CO2 emissions and global warming: ', corr)  # considered high correlation : 0.705

    # Lets watch to heatmap emissions vs warming
    df = pd.concat([emission_df, warming_df], axis=1).dropna(how='all')

    # Calculate the correlation matrix for 1860 and 2020 years
    df_corr = df['1860':'2020']
    corr_matrix = df_corr.corr()

    # 5. Correlation matrix heatmap between warming and emissions
    sns.heatmap(corr_matrix, annot=True)
    plt.title('Correlation of Global Warming to Emissions in Italy for the 1860 – 2020 period',
              fontsize=11, x=0.6)
    plt.tight_layout()
    plt.show()

    forecast_claims(emission_df, warming_df)

