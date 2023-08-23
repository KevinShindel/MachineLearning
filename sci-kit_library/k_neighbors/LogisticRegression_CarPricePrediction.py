from random import randint

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder

TRAIN_SIZE = 0.8
TEST_SIZE = 0.2
MAX_ITER = 1000


# 2. Read the data as a data frame
def read_data():
    path = '../../../MachineLearning/dataset/CarPrice_Assignment.csv'
    df = pd.read_csv(path)
    return df


# 3. prepare the data for training
def prepare_data(df):
    df.drop(['car_ID', 'CarName', 'price'], axis=1, inplace=True)
    label_encoder = LabelEncoder()
    columns_to_encode = df.select_dtypes(include=['object']).columns

    df[columns_to_encode] = df[columns_to_encode].apply(lambda col: label_encoder.fit_transform(col))

    return df


def create_model():
    model = LogisticRegression(max_iter=MAX_ITER)
    return model


def train_model(model, train_df):
    model.fit(train_df.drop('horsepower', axis=1), train_df['horsepower'])
    return model


def split_data(df):
    test_df, train_df = np.split(df, [int(TEST_SIZE * len(df))])
    return test_df, train_df


def test_model(model, test_df):
    predictions = model.predict(test_df.drop('horsepower', axis=1))
    return predictions


def evaluate_model(predictions, test_df):
    accuracy = metrics.accuracy_score(test_df['horsepower'], predictions)
    return accuracy


def create_knearest_model():
    model = KNeighborsRegressor(n_neighbors=5)
    return model



def main():
    car_df = read_data()
    car_df = prepare_data(car_df)
    # test_df, train_df = split_data(car_df)
    # model = create_model()
    # trained_model = train_model(model, train_df)
    # predictions = test_model(trained_model, test_df)
    # accuracy = evaluate_model(predictions, test_df)
    #
    # print(accuracy) # 0.25 % accuracy
    x_train, x_test, y_train, y_test = train_test_split(car_df.drop('horsepower', axis=1),
                                                        car_df['horsepower'],
                                                        test_size=TEST_SIZE,
                                                        random_state=randint(0, 100)
                                                        )
    knn_model = create_knearest_model()
    # Train the regressor
    knn_model.fit(x_train, y_train)

    # Make predictions on the test set
    y_pred = knn_model.predict(x_test)

    accuracy = knn_model.score(x_test, y_test)
    print(f"Accuracy: {accuracy}")

    # Calculate the mean squared error of the regressor
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")  # 0.75 % accuracy

    # improve model
    improved_knn = KNeighborsRegressor(n_neighbors=4, weights='distance')
    improved_knn.fit(x_train, y_train)
    improved_accuracy = improved_knn.score(x_test, y_test)
    print(f"Improved Accuracy: {improved_accuracy}")  # 0.89 % accuracy


if __name__ == '__main__':
    main()
