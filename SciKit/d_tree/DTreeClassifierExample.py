from sklearn.metrics import accuracy_score
from sklearn import tree


def main():
    clf = tree.DecisionTreeClassifier()
    features = [[140, 1], [130, 1], [150, 0], [170, 0]]
    labels = [0, 0, 1, 1]
    features_test = [[150, 0], [130, 1], [140, 1], [170, 0]]
    labels_test = [1, 0, 0, 1]

    clf = clf.fit(features, labels)
    pred = clf.predict(features_test)
    acc = accuracy_score(pred, labels_test)
    print(acc)
    print(len(features[0]))
    print(len(features_test[0]))
    print(len(labels))
    print(len(labels_test))
    print(len(pred))
    print(pred)
    print(labels_test)
    print(features_test)
    print(features)
    print(labels)


if __name__ == '__main__':
    main()
