"""
Decision Tree Classifier for Iris dataset
Author: Kevin Shindel
Date: 2024-08-05
"""

# import libraries
import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier


def load_data():
    iris_df = load_iris()
    return iris_df


def prepare_data(iris_df):
    return iris_df.data, iris_df.target


def create_model():
    return DecisionTreeClassifier()


def train_model(model, features, labels):
    model.fit(features, labels)


def predict(model, features):
    return model.predict(features)


def main():
    iris_df = load_data()
    features, labels = prepare_data(iris_df)  # features = iris_df.data, labels = iris_df.target
    model = create_model()  # model = DecisionTreeClassifier()
    train_model(model, features, labels)  # Fitting the model
    pred = predict(model, features)  # Predicting the model

    accuracy = np.mean(
        pred == labels
    )
    print(accuracy)


if __name__ == '__main__':
    main()
