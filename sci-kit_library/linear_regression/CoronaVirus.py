from pandas import read_csv
import numpy as np
import matplotlib.pylab as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


def main():
    print('*' * 30, ' LOAD DATA ', '*' * 30)
    filepath = '../../dataset/total_cases.csv'
    df = read_csv(filepath, usecols=['World'])
    df = df.rename(columns={'World': 'cases'}).fillna(0).astype(int)
    TOTAL_DAYS = df.shape[0] + 1
    df['days'] = np.arange(1, TOTAL_DAYS)

    print(df.describe())
    print('*' * 30, ' PREPARE DATA ', '*' * 30)
    x = np.array(df['days']).reshape(-1, 1)  # create array from one value each row
    y = np.array(df['cases']).reshape(-1, 1)  # create array from one value each row

    polyFeat = PolynomialFeatures(degree=3)
    x = polyFeat.fit_transform(x)

    print('*' * 30, ' TRAIN DATA ', '*' * 30)
    model = LinearRegression()
    model.fit(x, y)
    accuracy = model.score(x, y)
    accuracy = round((accuracy * 100) / 3, 2)  # check accuracy!
    print('Accuracy: ', accuracy, ' %')

    y0 = model.predict(x)

    print('*' * 30, ' PREDICTION ', '*' * 30)
    DAYS = 50
    x1 = np.arange(1, TOTAL_DAYS+DAYS).reshape(-1, 1)
    y1 = model.predict(polyFeat.fit_transform(x1))
    plt.plot(y1, '--r')
    plt.plot(y, '--b')
    plt.show()


if __name__ == '__main__':
    main()
