import random

import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor


def main():
    # Load the Boston Housing dataset
    boston_dataset_online = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
    boston_df = pd.read_csv(boston_dataset_online)

    # Split the dataset into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(boston_df.drop('medv', axis=1),
                                                        boston_df['medv'],
                                                        test_size=0.2,
                                                        random_state=random.randint(0, 100)
                                                        )

    # Create a KNN regressor with k=5
    knn = KNeighborsRegressor(n_neighbors=5)

    # Train the regressor
    knn.fit(x_train, y_train)

    # Make predictions on the test set
    y_pred = knn.predict(x_test)

    accuracy = knn.score(x_test, y_test)
    print(f"Accuracy: {accuracy}")

    # Calculate the mean squared error of the regressor
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    improved_knn = KNeighborsRegressor(n_neighbors=4, weights='distance')
    improved_knn.fit(x_train, y_train)
    improved_y_pred = improved_knn.predict(x_test)
    improved_accuracy = improved_knn.score(x_test, y_test)
    print(f"Improved Accuracy: {improved_accuracy}")
    improved_mse = mean_squared_error(y_test, improved_y_pred)
    print(f"Improved Mean Squared Error: {improved_mse}")


if __name__ == "__main__":
    main()

