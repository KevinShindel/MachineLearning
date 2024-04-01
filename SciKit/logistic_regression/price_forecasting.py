
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def load_data():
    avocado_df = pd.read_csv('../../data/avocado.csv')
    return avocado_df


def preprocess_data(avocado_df: pd.DataFrame):
    avocado_df['Date'] = pd.to_datetime(avocado_df['Date'])
    avocado_df = avocado_df[['Date', 'AveragePrice']]
    return avocado_df


def split_data(avocado_df: pd.DataFrame):
    x = avocado_df['Date']
    y = avocado_df['AveragePrice'].astype('int')
    return x, y


def train_model(x_train: pd.DataFrame, y_train: pd.DataFrame):
    model = LogisticRegression(max_iter=1000)
    x_train = np.array(x_train).reshape(-1, 1)
    y_train = np.array(y_train).reshape(-1, 1)
    model.fit(x_train, y_train)
    return model


def predict(model: LogisticRegression, x_test: pd.DataFrame):
    y_pred = model.predict(x_test)
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


def predict_price(model: LogisticRegression, x: pd.DataFrame, date: str):
    price = model.predict(x.loc[date])
    return price


def main():
    avocado_df = load_data()
    avocado_df = preprocess_data(avocado_df)
    x, y = split_data(avocado_df)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    model = train_model(x_train, y_train)
    x_test = np.array(x_test).reshape(-1, 1)
    y_test = np.array(y_test).reshape(-1, 1)
    y_pred = predict(model, pd.DataFrame(x_test))
    mse, r2 = evaluate(pd.DataFrame(y_test), y_pred)
    print('Mean squared error: %.2f' % mse)
    print('Coefficient of determination: %.2f' % r2)
    plot(pd.DataFrame(y_test), y_pred)
    price = predict_price(model, x, '2019-01-01')
    print('Predicted price for 2019-01-01: %.2f' % price)
    price = predict_price(model, x, '2019-01-02')
    print('Predicted price for 2019-01-02: %.2f' % price)
    price = predict_price(model, x, '2019-01-03')
    print('Predicted price for 2019-01-03: %.2f' % price)


if __name__ == '__main__':
    main()
