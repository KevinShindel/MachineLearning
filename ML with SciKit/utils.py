from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def show_report(y_pred, y_test):
    print(classification_report(y_test, y_pred))
    print('Accuracy: ', accuracy_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    cmd = ConfusionMatrixDisplay(cm)
    cmd.plot(cmap='viridis', text_kw={'color': 'black'})

    plt.show()
    print('Confusion Matrix: \n', cm)


def split_500_hits() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """ Split 500 hits dataset into train and test set """
    col_to_drop = ['PLAYER', 'CS']
    data = pd.read_csv('../dataset/500hits.csv', encoding='latin-1')
    data.drop(col_to_drop, axis=1, inplace=True)

    X = data.drop('HOF', axis=1)  # features
    y = data['HOF']  # target

    # split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
