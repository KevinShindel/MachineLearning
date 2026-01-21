from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np


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


def find_optimal_score(self, inertia_values: list, silhouette_scores: list) -> tuple:
    """
    This function finds the optimal number of clusters (k) using both the Elbow Method and Silhouette Method.
    """
    wcss = inertia_values  # inertia values
    s_score = silhouette_scores  # silhouette
    k_values = range(self.min_k, len(wcss) + (self.min_k - 1))

    # Define points for the line connecting the first and last WCSS values
    x1, y1 = k_values[0], wcss[0]
    x2, y2 = k_values[-1], wcss[-1]

    # Calculate the distances from each point to the line (Euclidean Distance)
    distances = []
    for i, (x, y) in enumerate(zip(k_values, wcss), start=0):
        numerator = abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1)
        denominator = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
        distances.append(numerator / denominator)

    # Find the index of the maximum distance (elbow point)
    optimal_k_elbow = distances.index(max(distances)) + (self.min_k - 1)

    # Silhouette method
    optimal_k_silhouette = k_values[s_score.index(max(s_score))]

    return optimal_k_silhouette, optimal_k_elbow