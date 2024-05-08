"""
This script is used to predict if an animal is eatable or not based on the dataset provided.
Author: Kevin Shindel
Date: 2024-08-05
"""

# import libraries
from random import randint

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


class ZooPredictor:
    """ This class is used to predict if an animal is eatable or not based on the dataset provided. """

    def __init__(self, model, features, target, data_path='dataset/zoo.csv'):
        """ This method initializes the ZooPredictor class. """
        self.__model = model
        self.__features = features
        self.__target = target
        self.__data_path = data_path
        self.__score = 0

    def predict(self, data: pd.DataFrame):
        """ This method is used to predict if an animal is eatable or not based on the dataset provided. """
        row_for_prediction = data.loc[:, self.__features]
        predicted = self.__model.predict(row_for_prediction.values)
        return predicted

    def train(self):
        """ This method is used to train the model. """
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
        """ This method returns the score of the model. """
        return self.__score


def main():
    model = DecisionTreeClassifier()  # create a model of DecisionTreeClassifier
    features = ['hair', 'feathers', 'eggs', 'milk', 'airborne', 'aquatic', 'predator', 'toothed', 'backbone',
                'breathes', 'venomous', 'fins', 'legs', 'tail', 'domestic', 'catsize']  # features
    target = 'eatable'  # target variable
    predictor = ZooPredictor(model, features, target).train()  # create an instance of ZooPredictor and train the model

    data = pd.read_csv('../../dataset/zoo.csv')  # read the dataset
    data_for_predict = data.copy()  # create a copy of the dataset
    for row in data.iterrows():  # iterate over the dataset
        row_df = row[1].to_frame().T  # create a DataFrame from the row
        predicted = predictor.predict(row_df)  # predict if the animal is eatable or not
        data_for_predict.loc[row[0], 'predicted_eatable'] = predicted[0]  # add the prediction to the dataset

    data_for_predict.to_csv('zoo_predicted.csv', index=False)  # save the dataset to a file
    print('Score: ', predictor.score)  # print the score of the model


if __name__ == '__main__':
    main()
