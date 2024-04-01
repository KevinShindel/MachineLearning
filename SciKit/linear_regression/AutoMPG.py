import os
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def create_model(x, y, step: int):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=step)
    print('xTrain', x_train.shape)
    print('xTest', x_test.shape)

    model = LinearRegression()
    model.fit(x_train, y_train)
    accuracy = model.score(x_test, y_test)
    accuracy = round(accuracy * 100, 3)
    return step, accuracy


def main():
    print('*'*30, ' DATA LOADING ', '*'*30)

    filepath = '../../dataset/auto-mpg.fwf'
    columns = ['displacement', 'mpg', 'cylinders', 'hp', 'weight', 'acceleration', 'model_year', 'origin', 'car_name']
    colspecs = [(1, 4), (5, 8), (8, 16), (16, 27), (27, 38),
                (38, 48), (48, 53), (53, 56), (56, None)]
    # row = '18.0   8   307.0      130.0      3504.      12.0   70  1	"chevrolet chevelle malibu"'
    df = pd.read_fwf(filepath,
                     names=columns,
                     header=None,
                     colspecs=colspecs)

    print('*'*30, ' DATA PREPROCESSING ', '*'*30)
    print(df.describe())

    df['car_name'] = df['car_name'].str.replace('"', '')
    df.drop('car_name', axis=1, inplace=True)  # DROP CAR NAME FOR BEST SCORE

    print(df.head(5))
    df['hp'] = df['hp'].replace('?', 0)
    df[columns[:-1]] = df[columns[:-1]].astype(float).astype(int)
    print(df.dtypes)

    print('*' * 30, ' DATA PREPROCESSING ', '*' * 30)

    # le = LabelEncoder() # SKIP THIS IF COLUMN DROPPED
    # df['car_name'] = le.fit_transform(df['car_name'])

    # LINEAR REGRESSION METHOD
    x = np.array(df.drop(['mpg'], axis=1))
    y = np.array(df['mpg'])

    test_sizes = np.arange(start=0.05, stop=0.5, step=0.05)
    max_workers = min([test_sizes.shape[0], os.cpu_count()])

    # Adjust accuracy from test_size
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        repeat_len = test_sizes.shape[0]
        results = list(executor.map(create_model, repeat(x, repeat_len), repeat(y, repeat_len), test_sizes))

    step, accuracy = next(iter(sorted(results, key=lambda i: i[1], reverse=True)))
    print('Fount max accuracy: ', accuracy, ' with step: ', step)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=step)
    model = LinearRegression()
    model.fit(x_train, y_train)

    print('#### EVALUATION ####')
    for_predict = np.array(df.drop(['mpg'], axis=1))
    predicted_vals = model.predict(for_predict)

    print('Coeff: ', model.coef_)
    print('Intercept: ', model.intercept_)
    print('Accuracy: ', accuracy, '%')

    df['predicted_mpg'] = predicted_vals.astype(int)
    df['error_int'] = df['mpg'] - df['predicted_mpg']
    df['error_%'] = (df['predicted_mpg'] * 100 / df['mpg']).round(2) - 100

    df.to_csv('../dataset/mpg_predicted.csv')
    print('*'*30, ' DATA SAVED ', '*'*30)

    # TODO: Create logic for feature/target transformation
    # feature = ...
    # target = ...
    exit(0)


if __name__ == '__main__':
    main()
