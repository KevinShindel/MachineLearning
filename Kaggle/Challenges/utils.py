from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from pandas import Series
import numpy as np


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