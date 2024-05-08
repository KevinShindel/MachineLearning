"""
Description: This script trains a logistic regression model to predict whether a person survived the Titanic disaster
                using the Titanic dataset. The script imports the dataset, cleans the data, splits the data into training
                and test sets, creates a logistic regression model, trains the model, tests the model, evaluates the model,
                improves the model using feature selection and pipeline, and presents the results using cross validation.
Author: Kevin Shindel
Date: 2024-08-05
"""

# import libraries
import numpy as np
import pandas as pd
from sklearn import feature_selection, pipeline
from sklearn import metrics
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

TEST_SIZE = 0.2
TRAIN_SIZE = 0.8
MAX_ITER = 1000


def import_data():
    # import train and test datasets
    online_titanic_file = 'https://raw.githubusercontent.com/Geoyi/Cleaning-Titanic-Data/master/titanic_original.csv'
    titanic_df = pd.read_csv(online_titanic_file)
    return titanic_df


def clean_data(df):
    # drop columns that are not useful
    df = df.drop(['name', 'ticket', 'cabin'], axis=1)
    return df


def split_data(df):
    # split dataset into training and test sets
    train_df = df.sample(frac=TRAIN_SIZE, random_state=0)
    test_df = df.drop(train_df.index)
    return train_df, test_df


def create_model():
    return LogisticRegression(max_iter=MAX_ITER)


def train_model(model, train_df):
    # train model
    model.fit(train_df.drop('survived', axis=1), train_df['survived'])
    return model


def test_model(model, test_df):
    # test model
    predictions = model.predict(test_df.drop('survived', axis=1))
    return predictions


def evaluate_model(predictions, test_df):
    # evaluate model
    accuracy = metrics.accuracy_score(test_df['survived'], predictions)
    return accuracy


def improve_model(model):
    # improve model by using feature selection and pipeline to chain the feature selection and model
    improved_model = pipeline.Pipeline([
        ('feature_selection', feature_selection.SelectKBest(k='all')),
        ('classification', model)
    ])
    return improved_model


def present_results(model):
    # present results using cross validation to get a more accurate estimate of the model's performance
    results = model_selection.cross_val_score(model,
                                              train_df.drop('survived', axis=1),
                                              train_df['survived'], cv=10,
                                              scoring='accuracy')
    mean = results.mean()
    std = results.std()
    print('Improved Accuracy: %.3f (%.3f)' % (mean, std))


def prepare_data(cleaned_titanic_df):
    # prepare data by converting categorical data to numerical data and filling in missing values
    # with the mean value of the column
    # and dropping rows with missing values in the embarked column
    encoder = LabelEncoder()

    cleaned_titanic_df['sex'] = encoder.fit_transform(cleaned_titanic_df['sex'])
    cleaned_titanic_df['embarked'] = encoder.fit_transform(cleaned_titanic_df['embarked'])

    cleaned_titanic_df['age'] = cleaned_titanic_df['age'].fillna(cleaned_titanic_df['age'].mean())
    cleaned_titanic_df['fare'] = cleaned_titanic_df['fare'].fillna(cleaned_titanic_df['fare'].mean())
    cleaned_titanic_df['body'] = cleaned_titanic_df['body'].fillna(cleaned_titanic_df['body'].mean())

    cleaned_titanic_df['home.dest'] = cleaned_titanic_df['home.dest'].fillna('Unknown').astype('category').cat.codes
    cleaned_titanic_df['boat'] = cleaned_titanic_df['boat'].fillna('Unknown').astype('category').cat.codes

    cleaned_titanic_df.dropna(inplace=True, axis=0)
    return cleaned_titanic_df


def predict_one(model, cleaned_titanic_df, train_df):
    # predict whether a random person survived or not using the improved model
    random_loc = np.random.randint(len(cleaned_titanic_df))
    one_person_df = cleaned_titanic_df.take([random_loc], axis=0)

    model.fit(train_df.drop('survived', axis=1), train_df['survived'])

    prediction = model.predict(one_person_df.drop('survived', axis=1))
    accuracy = metrics.accuracy_score(one_person_df['survived'], prediction)
    return accuracy


if __name__ == '__main__':
    titanic_df = import_data()
    cleaned_titanic_df = clean_data(titanic_df)
    prepared_titanic_df = prepare_data(cleaned_titanic_df)
    train_df, test_df = split_data(cleaned_titanic_df)
    model = create_model()
    trained_model = train_model(model, train_df)
    predictions = test_model(trained_model, test_df)
    accuracy = evaluate_model(predictions, test_df)
    print('Non improved model accuracy: ', accuracy)

    improved_model = improve_model(model)
    present_results(improved_model)

    # accuracy = predict_one(improved_model, prepared_titanic_df, train_df)
    # print(accuracy)


