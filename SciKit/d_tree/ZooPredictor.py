from random import randint

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


class ZooPredictor:
    def __init__(self, model, features, target, data_path='dataset/zoo.csv'):
        self.__model = model
        self.__features = features
        self.__target = target
        self.__data_path = data_path
        self.__score = 0

    def predict(self, data: pd.DataFrame):
        row_for_prediction = data.loc[:, self.__features]
        predicted = self.__model.predict(row_for_prediction.values)
        return predicted

    def train(self):
        raw_data = pd.read_csv(self.__data_path)
        x = raw_data.loc[:, self.__features]
        y = raw_data.loc[:, [self.__target]]
        x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                            random_state=randint(0, 1000),
                                                            test_size=0.3)
        self.__model.fit(x_train, y_train)
        self.__score = round(self.__model.score(x_test.values, y_test.values), 2)
        return self

    @property
    def score(self):
        return self.__score


def main():
    model = DecisionTreeClassifier()
    features = ['hair', 'feathers', 'eggs', 'milk', 'airborne', 'aquatic', 'predator', 'toothed', 'backbone',
                'breathes', 'venomous', 'fins', 'legs', 'tail', 'domestic', 'catsize']
    target = 'eatable'
    predictor = ZooPredictor(model, features, target).train()

    data = pd.read_csv('../../dataset/zoo.csv')
    data_for_predict = data.copy()
    for row in data.iterrows():
        row_df = row[1].to_frame().T
        predicted = predictor.predict(row_df)
        data_for_predict.loc[row[0], 'predicted_eatable'] = predicted[0]

    data_for_predict.to_csv('zoo_predicted.csv', index=False)
    print('Score: ', predictor.score)


if __name__ == '__main__':
    main()
