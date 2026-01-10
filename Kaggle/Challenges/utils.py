from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import (classification_report,
                             accuracy_score,
                             confusion_matrix,
                             ConfusionMatrixDisplay,
                             roc_curve,
                             roc_auc_score)
import matplotlib.pyplot as plt
from pandas import Series
import numpy as np
import seaborn as sns
from sklearn.model_selection import cross_val_predict, StratifiedKFold

from optuna.study import StudyDirection
from optuna import Study
from optuna.trial import FrozenTrial


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


def get_stats(model, X_val, y_val, X_train, y_train):
    """
    Description: Get model statistics including ROC curve, confusion matrix,
     and classification report for both validation and training datasets.
    Args:
        model: Trained machine learning model.
        X_val: Validation feature set.
        y_val: Validation target set.
        X_train: Training feature set.
        y_train: Training target set.
    """

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42, )  # Stratified K-Fold Cross-Validation
    try:  # Try to get decision function scores (for models that support it)
        y_cross_val_predicted = cross_val_predict(model, X_val, y_val, cv=cv, method='decision_function')
        y_cross_train_predicted = cross_val_predict(model, X_train, y_train, cv=cv, method='decision_function')
    except Exception as err:  # Fallback to probability scores (for models that do not support decision function)
        print(f'Error using decision_function: {err}. Falling back to predict_proba.')
        y_cross_val_predicted = cross_val_predict(model, X_val, y_val, cv=cv, method='predict_proba')[:, 1]
        y_cross_train_predicted = cross_val_predict(model, X_train, y_train, cv=cv, method='predict_proba')[:, 1]

    model_name = type(model).__name__  # Get the model name

    fpr, tpr, thresholds = roc_curve(y_val, y_cross_val_predicted)  # FP Rate, TP Rate, Thresholds
    roc_score = roc_auc_score(y_val, y_cross_val_predicted)  # ROC AUC Score

    def get_conf_matrix(model_name, _y_val, y_pred):
        """ Plot confusion matrix for the model."""
        conf_matrix = confusion_matrix(_y_val, (y_pred > 0.5).astype(int))

        plt.figure(figsize=(6, 4))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Cancelled', 'Cancelled'],
                    yticklabels=['Not Cancelled', 'Cancelled'])
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title(f'{model_name} Confusion Matrix')
        plt.show()

    def plot_roc_curve(model_name, fpr, tpr, score):
        """ Plot ROC curve for the model."""
        plt.figure(figsize=(10, 5))
        plt.plot(fpr, tpr, linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.axis([0, 1, 0, 1])
        plt.title(f'{model_name} ROC Curve (AUC = {score:.4f})')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate (Recall)')
        plt.grid(visible=True)
        plt.show()

    plot_roc_curve(model_name, fpr, tpr, roc_score)  # Plot ROC curve
    get_conf_matrix(model_name, y_val, y_cross_val_predicted)  # Plot confusion matrix

    # Print Validation classification report
    report = classification_report(y_val, (y_cross_val_predicted > 0.5).astype(int))
    print(f'{model_name} Validation Dataset Classification Report:')
    print(report)

    # Print Training classification report
    report = classification_report(y_train, (y_cross_train_predicted > 0.5).astype(int))
    print(f'{model_name} Training Dataset Classification Report:')
    print(report)


class EarlyStoppingCallback:
    """
    Early stopping callback for Optuna studies.
    Stops the study if there is no improvement in the best value for a specified number of trials (patience).
    """
    def __init__(self, patience: int, min_delta: float = 0.0):
        """
        Args:
            patience (int): Number of trials to wait for improvement before stopping.
            min_delta (float): Minimum change in the monitored value to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_value = None

    def __call__(self, study: Study, trial: FrozenTrial):
        if self.best_value is None:
            self.best_value = study.best_value
            return

        if study.direction == StudyDirection.MINIMIZE:
            if study.best_value < self.best_value - self.min_delta:
                self.best_value = study.best_value
                self.counter = 0
            else:
                self.counter += 1
        else:
            if study.best_value > self.best_value + self.min_delta:
                self.best_value = study.best_value
                self.counter = 0
            else:
                self.counter += 1

        if self.counter >= self.patience:
            study.stop()
            print(f'Early stopping triggered after {self.counter} trials with no improvement.')