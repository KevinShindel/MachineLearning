# q: please write a program to predict the price using LogisticRegression
#    and the following data.
# a: 1. read data from csv file
#    2. split data into train and test
#    3. train model
#    4. predict
#    5. evaluate
#    6. plot
#    7. predict price for 2019-01-01
#    8. predict price for 2019-01-02
#    9. predict price for 2019-01-03

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np

def load_data():
    avocado_df = pd.read_csv('../../data/avocado.csv')
    return avocado_df

def preprocess_data(avocado_df: pd.DataFrame):
    avocado_df['Date'] = pd.to_datetime(avocado_df['Date'])
    avocado_df = avocado_df[['Date', 'AveragePrice']]
    # avocado_df.drop(['Unnamed: 0', ''], axis=1, inplace=True)
    # avocado_df['year'] = avocado_df['Date'].dt.year
    # avocado_df['month'] = avocado_df['Date'].dt.month
    # avocado_df['day'] = avocado_df['Date'].dt.day
    # avocado_df = avocado_df.drop(['Date'], axis=1)
    # avocado_df.drop(['Unnamed: 0'], axis=1, inplace=True)
    # le = LabelEncoder()
    # avocado_df['type'] = le.fit_transform(avocado_df['type'])
    # avocado_df['region'] = le.fit_transform(avocado_df['region'])

    return avocado_df


def split_data(avocado_df: pd.DataFrame):
    X = avocado_df['Date']
    y = avocado_df['AveragePrice'].astype('int')
    return X, y


def train_model(X_train: pd.DataFrame, y_train: pd.DataFrame):
    model = LogisticRegression(max_iter=1000)
    X_train = np.array(X_train).reshape(-1, 1)
    y_train = np.array(y_train).reshape(-1, 1)
    model.fit(X_train, y_train)
    return model


def predict(model: LogisticRegression, X_test: pd.DataFrame):
    y_pred = model.predict(X_test)
    return y_pred


def evaluate(y_test: pd.DataFrame, y_pred: pd.DataFrame):
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2


def plot(y_test: pd.DataFrame, y_pred: pd.DataFrame):
    plt.scatter(y_test, y_pred)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted')
    plt.show()


def predict_price(model: LogisticRegression, X: pd.DataFrame, date: str):
    price = model.predict(X.loc[date])
    return price


def main():
    avocado_df = load_data()
    avocado_df = preprocess_data(avocado_df)
    X, y = split_data(avocado_df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = train_model(X_train, y_train)
    X_test = np.array(X_test).reshape(-1, 1)
    y_test = np.array(y_test).reshape(-1, 1)
    y_pred = predict(model, X_test)
    mse, r2 = evaluate(y_test, y_pred)
    print('Mean squared error: %.2f' % mse)
    print('Coefficient of determination: %.2f' % r2)
    plot(y_test, y_pred)
    price = predict_price(model, X, '2019-01-01')
    print('Predicted price for 2019-01-01: %.2f' % price)
    price = predict_price(model, X, '2019-01-02')
    print('Predicted price for 2019-01-02: %.2f' % price)
    price = predict_price(model, X, '2019-01-03')
    print('Predicted price for 2019-01-03: %.2f' % price)

if __name__ == '__main__':
    main()
