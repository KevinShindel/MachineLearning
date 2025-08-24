from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from pandas import Series
import numpy as np
import seaborn as sns

def show_report(y_pred, y_test):
    report = classification_report(y_test, y_pred)
    print('Classification Report: \n', report)
    print('Accuracy: \n', accuracy_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    cmd = ConfusionMatrixDisplay(cm)
    cmd.plot(cmap='viridis', text_kw={'color': 'black'})

    plt.show()
    print('Confusion Matrix: \n', cm)


# MI Scores functions
def make_mi_scores(_X, _y, _discrete_features):
    _mi_scores = mutual_info_regression(_X, _y, discrete_features=_discrete_features)
    _mi_scores = Series(_mi_scores, name="MI Scores", index=_X.columns)
    _mi_scores = _mi_scores.sort_values(ascending=False)
    return _mi_scores


def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")


def detect_outliers(original_df):
    df = original_df.copy()
    # select numerical columns

    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    iqr = q3 - q1

    # calculate maximum and minimum
    maximum = q3 + 1.5 * iqr
    minimum = q1 - 1.5 * iqr

    # find outliers
    filtered_df = df[(df < minimum) | (df > maximum)]

    outlier_exist = np.all(filtered_df.isnull())
    print('Outliers exists: ', not outlier_exist)
    return filtered_df


def normalize_num_data(data, columns):
    for col in columns:
        data[col] = data[col].fillna(data[col].mean())
        q1 = data[col].quantile(0.25)
        q3 = data[col].quantile(0.75)
        iqr = q3 - q1

        # calculate maximum and minimum
        multiplier = 3
        maximum = q3 + multiplier * iqr
        minimum = max(0, q1 - multiplier * iqr)

        data[col] = np.where(
            data[col] > maximum,
            maximum,
            data[col]
        )

        data[col] = np.where(
            data[col] < minimum,
            minimum,
            data[col]
        )

        skew = data[col].skew()

        sns.histplot(data[col], kde=True)
        plt.title(f'{col} Distribution (Skew: {skew:.2f})')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.show()

    return data
