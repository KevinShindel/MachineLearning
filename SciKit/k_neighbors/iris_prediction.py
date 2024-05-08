import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


def data_preparation():
    # load dataset
    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['species'] = data.target

    # Arrange Data into Features Matrix and Target Vector

    feature_names = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

    feature_df = df.loc[:, feature_names]
    target_df = df.loc[:, 'species']

    x = df.loc[:, feature_names].values
    y = df.loc[:, 'species'].values


def linear_regression_intro():
    # set the style of the axes and the text color
    sns.set(style="darkgrid")
    # load the dataset
    df = pd.read_csv("../../dataset/linear.csv")
    df.head()

    # check for missing values
    df.isnull().sum()
    # drop the missing values
    df.dropna(inplace=True, how='any')

    # check for missing values
    assert bool(df.isnull().sum().any()) is False

    x = df.loc[:, ['x']].values
    y = df.loc[:, ['y']].values

    # make model
    reg = LinearRegression(fit_intercept=True)
    reg.fit(x, y)

    # make predictions
    once_prediction = reg.predict(x[0].reshape(1, -1))
    multiple_predictions = reg.predict(x[:5].reshape(-1, 1))

    # measure the performance
    score = round(reg.score(x, y), 2)
    print('Current score: ', score, ' %')

    # Plot the data
    plt.scatter(df['x'], df['y'])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    # Plotting the Best Fit Linear Regression Line in Red
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 7))

    ax.scatter(x, y, color='black')
    ax.plot(x, reg.predict(x), color='red', linewidth=3)
    ax.grid(True,
            axis='both',
            zorder=0,
            linestyle=':',
            color='k')
    ax.tick_params(labelsize=18)
    ax.set_xlabel('x', fontsize=24)
    ax.set_ylabel('y', fontsize=24)
    ax.set_title("Linear Regression Line with Intercept", fontsize=16)
    fig.tight_layout()
    plt.show()

    # Plotting Models With or Without Intercept
    # Model with Intercept (like earlier in notebook)
    reg_inter = LinearRegression(fit_intercept=True)
    reg_inter.fit(x, y)
    predictions_inter = reg_inter.predict(x)
    score_inter = reg_inter.score(x, y)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 7))

    for index, model in enumerate([LinearRegression(fit_intercept=True), LinearRegression(fit_intercept=False)]):
        model.fit(x, y)
        predictions = model.predict(x)
        score = model.score(x, y)

        ax[index].scatter(x, y, color='black')
        ax[index].plot(x, model.predict(x), color='red', linewidth=3)

        ax[index].tick_params(labelsize=18)
        ax[index].set_xlabel('x', fontsize=18)
        ax[index].set_ylabel('y', fontsize=18)
        ax[index].set_xlim(left=0, right=150)
        ax[index].set_ylim(bottom=0)

    ax[0].set_title('fit_intercept = True', fontsize=20)
    ax[1].set_title('fit_intercept = False', fontsize=20)
    fig.tight_layout()
    plt.show()


def train_split_tutorial():
    zoo_data_df = pd.read_csv("../../dataset/zoo.csv")
    zoo_data_df.head()

    # data preparation
    features = list(zoo_data_df.columns[:-1])
    target = zoo_data_df.columns[-1]
    x = pd.get_dummies(zoo_data_df.loc[:, features])

    # X = zoo_data_df.loc[:, features]
    y = zoo_data_df.loc[:, [target]]

    # split the data
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=324)

    # create the model regression classifier
    lg = KNeighborsClassifier()
    lg.fit(x_train, y_train)

    # measure the performance
    score = round(lg.score(x_test.values, y_test.values), 2)
    print('Current score: ', score, ' %')

    # make predictions
    predicted_row = x_test.iloc[0, :]
    once_prediction = lg.predict(predicted_row.values.reshape(1, -1))
    print('Predicted eatable: ', once_prediction)
    predicted_row['eatable'] = once_prediction[0]
    print(predicted_row)
    predicted_df = pd.DataFrame(predicted_row).T
    reversed_predicted = pd.from_dummies(predicted_df, sep='animal_name').rename(columns={'': 'animal_name'})
    print(reversed_predicted)


def regression_w_scaler():
    iris_data = load_iris()
    iris_df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
    iris_df['species'] = iris_data.target

    x_train, x_test, y_train, y_test = train_test_split(iris_df[['petal length (cm)']], iris_df['species'],
                                                        random_state=0)
    scaler = StandardScaler()

    # fit and transform the data
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # create the model regression classifier
    lg = LinearRegression()
    lg.fit(x_train_scaled, y_train)

    # observable pental length is the most important feature
    observable = x_test_scaled[0].reshape(1, -1)
    actual = y_test.iloc[0]

    print('prediction', lg.predict(observable)[0])
    print('actual', actual)

    example_df = pd.DataFrame()
    example_df.loc[:, 'petal length (cm)'] = x_test_scaled.reshape(-1)
    example_df.loc[:, 'species'] = y_test.values
    example_df['logistic_predicted'] = pd.DataFrame(lg.predict(x_test_scaled))

    print(example_df.head())

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 7))
    virginica_filter = example_df['species'] == 1
    versicolor_filter = example_df['species'] == 0

    ax.scatter(example_df.loc[virginica_filter, 'petal length (cm)'].values,
               example_df.loc[virginica_filter, 'logistic_predicted'].values,
               color='green',
               s=60,
               label='virginica')

    ax.scatter(example_df.loc[versicolor_filter, 'petal length (cm)'].values,
               example_df.loc[versicolor_filter, 'logistic_predicted'].values,
               color='blue',
               s=60,
               label='versicolor')

    ax.axhline(y=0.5, color='yellow')

    ax.axhspan(0.5, 1, alpha=0.05, color='green')
    ax.axhspan(0, 0.4999, alpha=0.05, color='blue')

    ax.text(0.5, .6, 'Classified as viginica', fontsize=16)
    ax.text(0.5, .4, 'Classified as versicolor', fontsize=16)

    ax.set_ylim(0, 1)
    ax.legend(loc='lower right', markerscale=1.0, fontsize=12)
    ax.tick_params(labelsize=18)
    ax.set_xlabel('petal length (cm)', fontsize=24)
    ax.set_ylabel('probability of virginica', fontsize=24)
    ax.set_title('Logistic Regression Predictions', fontsize=24)
    fig.tight_layout()
    plt.show()


def get_split_data():
    iris_data = load_iris()
    iris_df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
    iris_df['species'] = iris_data.target
    x_train, x_test, y_train, y_test = train_test_split(iris_df[['petal length (cm)']], iris_df['species'],
                                                        random_state=0)
    scaler = StandardScaler()

    # fit and transform the data
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    return x_train_scaled, x_test_scaled, y_train, y_test


def measuring_w_performance():
    # Measuring Model Performance
    x_train_scaled, x_test_scaled, y_train, y_test = get_split_data()

    lg = LinearRegression()
    lg.fit(x_train_scaled, y_train)
    score = lg.score(x_test_scaled, y_test)

    y_test = y_test.values.reshape(-1, 1)
    cm = metrics.confusion_matrix(y_test, lg.predict(x_test_scaled))

    plt.figure(figsize=(9, 9))

    sns.heatmap(cm, annot=True,
                fmt=".0f",
                linewidths=.5,
                square=True,
                cmap='Blues_r')

    plt.ylabel('Actual label', fontsize=17)
    plt.xlabel('Predicted label', fontsize=17)
    plt.title(f'Accuracy Score: {score}', size=17)
    plt.tick_params(labelsize=12)
    plt.show()


if __name__ == '__main__':
    # data_preparation()
    # linear_regression_intro()
    # train_split_tutorial()
    # regression_w_scaler()
    measuring_w_performance()
