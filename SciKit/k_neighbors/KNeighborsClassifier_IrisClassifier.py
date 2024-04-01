import pandas as pd
from sklearn import feature_selection, pipeline
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

TEST_SIZE = 0.2
TRAIN_SIZE = 0.8
MAX_ITER = 1000


# 2. import dataset
def import_data():
    # import train and test datasets
    iris_df = pd.read_csv('https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv')
    return iris_df


# 3. clean dataset
def clean_data(iris_df):
    # encode species
    le = LabelEncoder()
    iris_df['species'] = le.fit_transform(iris_df['species'])

    return iris_df


# 4. split dataset into training and test sets
def split_data(iris_df):
    # split dataset into training and test sets
    train_df = iris_df.sample(frac=TRAIN_SIZE, random_state=0)
    test_df = iris_df.drop(train_df.index)
    return train_df, test_df


# 5. create model
def create_model():
    # create model
    model = KNeighborsClassifier(n_neighbors=3)
    return model


# 6. train model
def train_model(model, train_df):
    # train model
    model.fit(train_df.drop('species', axis=1), train_df['species'])
    return model


# 7. test model
def test_model(model, test_df):
    # test model
    predictions = model.predict(test_df.drop('species', axis=1))
    return predictions


# 8. evaluate model
def evaluate_model(predictions, test_df):
    # evaluate model
    accuracy = metrics.accuracy_score(predictions, test_df['species'])
    return accuracy


# 9. improve model
def improve_model(model):
    # improve model
    improved_model = pipeline.Pipeline([
        ('feature_selection', feature_selection.SelectKBest(k='all')),
        ('classification', model)
    ])

    return improved_model


# 10. present results
def present_results(accuracy):
    # present results
    print('Accuracy: {0:.2f}'.format(accuracy))


def main():
    iris_df = import_data()
    iris_df = clean_data(iris_df)
    train_df, test_df = split_data(iris_df)
    model = create_model()
    model = train_model(model, train_df)
    predictions = test_model(model, test_df)
    accuracy = evaluate_model(predictions, test_df)
    present_results(accuracy)
    improved_model = improve_model(model)
    improved_model = train_model(improved_model, train_df)
    predictions = test_model(improved_model, test_df)
    accuracy = evaluate_model(predictions, test_df)
    present_results(accuracy)


if __name__ == '__main__':
    main()
