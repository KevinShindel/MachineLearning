import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def main():
    print('#### LOAD DATA ####')
    filepath = '../../dataset/houses_to_rent.csv'
    used_columns = ['city',
                    'rooms',
                    'bathroom',
                    'parking spaces',
                    'fire insurance',
                    'furniture',
                    'rent amount',
                    'animal']

    df = pd.read_csv(filepath, usecols=used_columns)
    print(df.head())

    print('#### CLEAR DATA ####')
    df['rent amount'] = df['rent amount'].map(lambda x: int(x[2:].replace(',', '')))
    df['fire insurance'] = df['fire insurance'].map(lambda x: int(x[2:].replace(',', '')))
    print(df.head())

    le = LabelEncoder()
    df['furniture'] = le.fit_transform(df['furniture'])
    df['animal'] = le.fit_transform(df['animal'])
    print(df.head())

    print('#### SPLIT DATA ####')
    x = np.array(df.drop(['rent amount'], axis=1))
    y = np.array(df['rent amount'])

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    model = LinearRegression()
    model.fit(x_train, y_train)
    accuracy = model.score(x_test, y_test)
    print('Coeff: ', model.coef_)
    print('Intercept: ', model.intercept_)
    print('Accuracy: ', round(accuracy*100, 3), '%')

    print('#### EVALUATION ####')
    test_vals = model.predict(x_test)

    error = []
    for i, testVal in enumerate(test_vals):
        error.append(y_test[i] - testVal)
        print(f'Actual value: {y_test[i]}, predicted: {int(test_vals[i])}, error: {int(error[i])}')

    for_predict = np.array(df.drop(['rent amount'], axis=1))

    predicted_vals = model.predict(for_predict)

    df['predicted rent amount'] = predicted_vals.astype(int)
    df['error_int'] = df['rent amount'] - df['predicted rent amount']
    df['error_%'] = (df['predicted rent amount'] * 100 / df['rent amount']).round(2) - 100

    df.to_csv('dataset/predicted.csv')


if __name__ == '__main__':
    main()
