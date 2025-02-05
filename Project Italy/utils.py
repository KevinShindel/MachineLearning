import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing

MAX_PREDICTION = 4


def found_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function will detect outliers in the data
    :param df: DataFrame
    :return: DataFrame
    """

    # lets detect Outliers, calculate IQR first
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    iqr = q3 - q1

    # calculate maximum and minimum
    maximum = q3 + 1.5 * iqr
    minimum = q1 - 1.5 * iqr

    # find outliers
    df = df[(df < minimum) & (df > maximum)]

    outlier_exist = np.all(df.isnull())
    print(f'Outliers exist: {not outlier_exist}')

    # use Z-score method to detect outliers
    z_scores = (df - df.mean()) / df.std()  # or z_scores = df.apply(stats.zscore)

    # filter out the outliers
    max_abs = z_scores.apply(lambda x: np.abs(x) < 3, axis='columns')
    filtered_entries = z_scores[max_abs]

    return filtered_entries


def adjust_model(df):
    # this function will adjust the model for the data
    # lets calculate the max year for test data and train data
    max_year = df.index.max().year
    test_max_year = max_year - MAX_PREDICTION

    # split the data into training and testing sets
    test = df.loc[str(test_max_year):]
    train = df.loc[:str(test_max_year-1)]

    # Define the parameter grid
    trend_params = ['add', 'mul', None]
    seasonal_params = ['add', 'mul', None]
    seasonal_periods_params = [3, 6, 12, 24]

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
                    predictions = model.predict(start=max_year-MAX_PREDICTION, end=max_year)

                    # seems test data is not equaled to predictions, need to equal to len and fill with 0
                    if len(test) < len(predictions):
                        test = test.resample('AS').sum().fillna(0)

                    # Calculate the score
                    score = mean_squared_error(test, predictions)

                    # Update the best score and the best parameters
                    if score < best_score:
                        best_score = score
                        best_params = (trend, seasonal, seasonal_periods, score)
                except Exception as Err:
                    print(Err)
                    continue

    # Print the best parameters
    print(f'Best parameters: trend={best_params[0]}, seasonal={best_params[1]}, seasonal_periods={best_params[2]}')
    return best_params
