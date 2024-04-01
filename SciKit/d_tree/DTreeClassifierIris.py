import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree


def load_data():
    iris_df = load_iris()
    return iris_df


def prepare_data(iris_df):
    return iris_df.data, iris_df.target


def create_model():
    return tree.DecisionTreeClassifier()


def train_model(model, features, labels):
    model.fit(features, labels)


def predict(model, features):
    return model.predict(features)


def main():
    iris_df = load_data()
    features, labels = prepare_data(iris_df)
    model = create_model()
    train_model(model, features, labels)
    pred = predict(model, features)

    accuracy = np.mean(
        pred == labels
    )
    print(accuracy)


if __name__ == '__main__':
    main()
