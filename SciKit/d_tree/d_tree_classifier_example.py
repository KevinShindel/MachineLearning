"""
This is an example of a decision tree classifier using the SciKit library.
Author: Kevin Shindel
Date: 2024-08-05
"""

# import libraries
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier


def main():
    """
    This function demonstrates how to use the DecisionTreeClassifier from the SciKit library.
    """
    clf = DecisionTreeClassifier()  # Create a DecisionTreeClassifier object
    features = [[140, 1], [130, 1], [150, 0], [170, 0]]  # Training data
    labels = [0, 0, 1, 1]  # Labels for the training data
    features_test = [[150, 0], [130, 1], [140, 1], [170, 0]]  # Test data
    labels_test = [1, 0, 0, 1]  # Labels for the test data

    clf = clf.fit(features, labels)  # Fit the model
    pred = clf.predict(features_test)  # Predict the labels
    acc = accuracy_score(pred, labels_test)  # Calculate the accuracy
    print("Accuracy: ", acc)  # Print accuracy of the model


if __name__ == '__main__':
    main()
