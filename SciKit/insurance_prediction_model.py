import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing

CURRENT_RATE = 0.15  # Insurance rate for 1 class air transportation


def adjust_model(df):
    # Split the data into training and testing sets
    test = df.loc['2020':'2024', 'avg_anomaly_temp']
    train = df.loc[:'2024', 'avg_anomaly_temp']

    # Define the parameter grid
    trend_params = ['add', 'mul', None]
    seasonal_params = ['add', 'mul', None]
    seasonal_periods_params = [1, 3, 6, 12, 24]

    # Initialize the best parameters and the best score
    best_params = None
    best_score = float('inf')

    # Grid search
    for trend in trend_params:
        for seasonal in seasonal_params:
            for seasonal_periods in seasonal_periods_params:
                try:
                    # Fit the model
                    model = ExponentialSmoothing(train, trend=trend, seasonal=seasonal,
                                                 seasonal_periods=seasonal_periods).fit()

                    # Make predictions
                    predictions = model.predict(start='2020', end='2024')

                    # Calculate the score
                    score = mean_squared_error(test, predictions)

                    # Update the best score and the best parameters
                    if score < best_score:
                        best_score = score
                        best_params = (trend, seasonal, seasonal_periods, score)
                except:
                    continue

    # Print the best parameters
    print(f'Best parameters: trend={best_params[0]}, seasonal={best_params[1]}, seasonal_periods={best_params[2]}')


def forecast_temperature(df):
    # Train the model on data up to 2024
    train = df.loc[:'2024', 'avg_anomaly_temp']
    model = ExponentialSmoothing(train, seasonal_periods=3, trend='mul', seasonal='add').fit()

    # evaluate on test data
    test = df.loc['2020':'2024', 'avg_anomaly_temp']
    predictions = model.predict(start='2020', end='2024')
    mse = mean_squared_error(test, predictions)
    print(f'Mean Squared Error: {mse}')

    # Plot the real data and the predicted data for 2020-2024
    test = df.loc['2015':'2024', 'avg_anomaly_temp']
    test.plot(label='Real')

    forecast = model.predict(start='2020', end='2024')

    forecast.plot(label='Forecasted')
    plt.legend()
    plt.show()

    # Forecast the temperature for 2024-2028
    test = df.loc['2023':, 'avg_anomaly_temp']
    test.plot(label='Real')
    forecast = model.predict(start='2023', end='2028')

    # Plot the forecasted data
    forecast.plot(label='Forecasted')
    plt.legend()
    plt.show()


def forecast_emissions():
    # Read the data
    df = pd.read_csv('../dataset/co2_emission.csv', index_col='year', parse_dates=True, usecols=['year', 'emissions'])

    # Train the model on data up to 2022
    train = df.loc[:'2022', 'emissions']
    model = ExponentialSmoothing(train, seasonal_periods=3, trend='mul', seasonal='add').fit()

    # Forecast the emissions for 2021-2026
    forecast = model.predict(start='2021', end='2026')

    test_forecast = model.predict(start='1860', end='2022')

    # Calculate the mean squared error
    mse = mean_squared_error(df, test_forecast)
    print(f'Mean Squared Error: {mse}')

    # Plot test data vs forecasted data
    df.plot(label='Real')
    test_forecast.plot(label='Forecasted')
    plt.legend()
    plt.show()

    # Create plot with real data and forecasted data
    df['2020':'2022'].plot(label='Real', ylabel='CO2 Emissions', xlabel='Year')
    forecast.plot(label='Forecasted', ylabel='CO2 Emissions', xlabel='Year')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    forecast_emissions()



    # #  Read the data
    # df = pd.read_csv('../dataset/temperature_anomaly.csv',
    #                  index_col='year',
    #                  parse_dates=True,
    #                  usecols=['year', 'avg_anomaly_temp'])
    # MAX_YEAR = df.index.max().year
    #
    # # set insurance rate for current year
    # df['insurance_rate'] = np.nan
    # df.loc[df.index.year == MAX_YEAR, 'insurance_rate'] = CURRENT_RATE
    #
    # # normalize temperature data from min year to max year
    # min_val = df['avg_anomaly_temp'].min()
    # max_val = df['avg_anomaly_temp'].max()
    #
    # df['avg_anomaly_temp'] = (df['avg_anomaly_temp'] - min_val) / (max_val - min_val)
    #
    # # calculate linear relation for insurance rate
    # df['anomaly_coeff'] = np.nan
    # df.loc[df.index.year == MAX_YEAR, 'anomaly_coeff'] = df.loc[
    #                                                          df.index.year == MAX_YEAR, 'avg_anomaly_temp'] / CURRENT_RATE
    #
    # # get anomaly coefficient
    # ANOMALLY_COEFF = df.loc[df.index.year == MAX_YEAR, 'anomaly_coeff'].values[0]
    #
    # # calculate insurance rate for other years
    # df['insurance_rate'] = df['avg_anomaly_temp'] / ANOMALLY_COEFF
    #
    # # prepare dataset for forecasting training
    # df.index = pd.to_datetime(df.index, format='%Y')
    #
    # # Add a constant to make all values positive
    # df['avg_anomaly_temp'] = df['avg_anomaly_temp'] + abs(df['avg_anomaly_temp'].min()) + 1
    #
    #
    # # Train the model on data up to 2024
    # train = df.loc[:'2024', 'avg_anomaly_temp']
    # model = ExponentialSmoothing(train, seasonal_periods=3, trend='mul', seasonal='add').fit()
    #
    # # evaluate on test data
    # test = df.loc['2020':'2024', 'avg_anomaly_temp']
    # predictions = model.predict(start='2020', end='2024')
    # mse = mean_squared_error(test, predictions)
    # print(f'Mean Squared Error: {mse}')
    #
    # # Plot the real data and the predicted data for 2020-2024
    # test = df.loc['2015':'2024', 'avg_anomaly_temp']
    # test.plot(label='Real')
    #
    # forecast = model.predict(start='2020', end='2024')
    #
    # forecast.plot(label='Forecasted')
    # plt.legend()
    # plt.show()
    #
    # # Forecast the temperature for 2024-2028
    # test = df.loc['2023':, 'avg_anomaly_temp']
    # test.plot(label='Real')
    # forecast = model.predict(start='2023', end='2028')
    #
    # # Plot the forecasted data
    # forecast.plot(label='Forecasted')
    # plt.legend()
    # plt.show()
