import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt


def predict_iris_via_random_forest():
    # load data
    iris_df = pd.read_csv('../files/iris.csv')

    # split data to features/target
    X = iris_df.drop('Species', axis=1)
    y = iris_df['Species']

    # split data to train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=42)

    # train model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Random Forest Accuracy: {accuracy}')

    # save model
    # joblib.dump(model, 'models/iris_model.pkl')
    # print('Model saved')

    # Predict one iris
    iris = [[5.1, 3.5, 1.4, 0.2]]
    model = joblib.load('models/iris_model.pkl')
    prediction = model.predict(iris)
    print(f'Prediction: {prediction}')

    # Predict multiple iris
    iris = [[5.1, 3.5, 1.4, 0.2], [6.2, 3.4, 5.4, 2.3]]
    prediction = model.predict(iris)
    print(f'Prediction: {prediction}')


def predict_iris_via_ann():
    # load data
    iris_df = pd.read_csv('../files/iris.csv')

    # create label encoder
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    iris_df['Species'] = le.fit_transform(iris_df['Species'])

    print(iris_df.head())

    # convert data to numpy array
    np_iris = iris_df.values

    # split data to features/target
    X_data = np_iris[:, :4]
    Y_data = np_iris[:, 4]

    # create scaler model and fit it
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler()

    X_data = sc.fit_transform(X_data)
    Y_data = tf.keras.utils.to_categorical(Y_data, 3)

    # Split training and test data
    X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.10)

    # Number of classes in the target variable
    NB_CLASSES = 3

    # Create a sequencial model in Keras
    model = tf.keras.models.Sequential()

    # Add the first hidden layer
    model.add(Dense(128,  # Number of nodes
                    # input_shape=(4,),  # Number of input variables
                    name='Hidden-Layer-1',  # Logical name
                    activation='relu'))  # activation function

    # Add a second hidden layer
    model.add(Dense(128,
                    name='Hidden-Layer-2',
                    activation='relu'))

    # Add an output layer with softmax activation
    model.add(Dense(NB_CLASSES,
                    name='Output-Layer',
                    activation='softmax'))

    # Compile the model with loss & metrics
    model.compile(loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Print the model meta-data
    model.summary()

    # Make it verbose so we can see the progress
    VERBOSE = 1

    # Setup Hyper Parameters for training

    # Set Batch size
    BATCH_SIZE = 16

    # Set number of epochs
    EPOCHS = 10

    # Set validation split. 20% of the training data will be used for validation
    # after each epoch
    VALIDATION_SPLIT = 0.2

    # Fit the model. This will perform the entire training cycle, including
    # forward propagation, loss computation, backward propagation and gradient descent.
    # Execute for the specified batch sizes and epoch
    # Perform validation after each epoch
    history = model.fit(X_train,
                        Y_train,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        verbose=VERBOSE,
                        validation_split=VALIDATION_SPLIT)

    # Plot accuracy of the model after each epoch.
    pd.DataFrame(history.history)["accuracy"].plot(figsize=(8, 5))
    plt.title("Accuracy improvements with Epoch")
    plt.show()

    # Evaluate the model against the test dataset and print results
    print("\nEvaluation against Test Dataset :\n------------------------------------")
    model.evaluate(X_test, Y_test)

    # Raw prediction data
    prediction_input = [[6.6, 3., 4.4, 1.4]]

    # Scale prediction data with the same scaling model
    scaled_input = sc.transform(prediction_input)

    # Get raw prediction probabilities
    raw_prediction = model.predict(scaled_input)
    print("Raw Prediction Output (Probabilities) :", raw_prediction)

    # Find prediction
    prediction = np.argmax(raw_prediction)
    print("Prediction is ", le.inverse_transform([prediction]))

    # calculate accuracy for Sequential model
    loss, accuracy = model.evaluate(X_test, Y_test)
    print(f"ANN Accuracy: {accuracy}")

    # save model
    model.save('models/iris_ann_model.ker+'
               'as')


if __name__ == '__main__':
    predict_iris_via_ann()
    predict_iris_via_random_forest()
