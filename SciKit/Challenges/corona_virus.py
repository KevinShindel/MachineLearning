"""
Description: This script is used to predict the number cases of coronavirus in the world.
Author: Kevin Shindel
Date: 2024-08-05
"""

# import libraries
from pandas import read_csv
import numpy as np
import matplotlib.pylab as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


def main():
    print('*' * 30, ' LOAD DATA ', '*' * 30)
    filepath = '../../dataset/total_cases.zip'
    df = read_csv(filepath, usecols=['World'])
    df = df.rename(columns={'World': 'cases'}).fillna(0).astype(int)
    total_days = df.shape[0] + 1
    df['days'] = np.arange(1, total_days)

    print(df.describe())
    print('*' * 30, ' PREPARE DATA ', '*' * 30)
    x = np.array(df['days']).reshape(-1, 1)  # create array from one value each row
    y = np.array(df['cases']).reshape(-1, 1)  # create array from one value each row

    poly_feat = PolynomialFeatures(degree=3)
    x = poly_feat.fit_transform(x)

    print('*' * 30, ' TRAIN DATA ', '*' * 30)
    model = LinearRegression()
    model.fit(x, y)
    accuracy = model.score(x, y)
    accuracy = round((accuracy * 100) / 3, 2)  # check accuracy!
    print('Accuracy: ', accuracy, ' %')

    y0 = model.predict(x)

    print('*' * 30, ' PREDICTION ', '*' * 30)
    days = 50
    x1 = np.arange(1, total_days+days).reshape(-1, 1)
    y1 = model.predict(poly_feat.fit_transform(x1))
    plt.plot(y1, '--r')
    plt.plot(y, '--b')
    plt.show()


if __name__ == '__main__':
    main()
