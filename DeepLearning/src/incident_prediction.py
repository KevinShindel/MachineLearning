import pandas as pd
import tensorflow as tf
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np


def main():
    # Load the data
    incident_data = pd.read_csv('../files/root_cause_analysis.csv')

    # create label encoder
    label_encoder = preprocessing.LabelEncoder()

    # transform the ROOT_CLAUSE column
    incident_data['ROOT_CAUSE'] = label_encoder.fit_transform(incident_data['ROOT_CAUSE'])

    # print dtypes
    print(incident_data.dtypes)

    # show head
    print(incident_data.head())

    # split data into features and labels
    x_data = incident_data.drop('Incident', axis=1)
    y_data = incident_data['Incident']

    # convert strings to categorical values
    y_data = tf.keras.utils.to_categorical(y_data)

    # split data into training and testing
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2,
                                                        random_state=42)

    # show cnt of features and labels
    print('x_train shape:', x_train.shape)
    print('y_train shape:', y_train.shape)

    # Build and evaluate the model

    # constants
    EPOCHS = 20
    BATCH_SIZE = 64
    VERBOSE = 1
    N_HIDDEN = 128
    VALIDATION_SPLIT = 0.2
    OUTPUT_CLASSES = len(label_encoder.classes_.size)

    # create Keras model
    model = tf.keras.models.Sequential()

    # add hidden layer
    model.add(tf.keras.layers.Dense(N_HIDDEN, input_shape=(x_train.shape[1],),
                                    name='dense_layer_1',
                                    activation='relu'))

    # add hidden layer 2
    model.add(tf.keras.layers.Dense(N_HIDDEN, name='dense_layer_2', activation='relu'))

    # add output layer
    model.add(tf.keras.layers.Dense(OUTPUT_CLASSES, name='dense_layer_3', activation='softmax'))

    # compile the model
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    # train the model
    model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=VERBOSE,
              validation_split=VALIDATION_SPLIT)

    # evaluate the model
    print('Model evaluation')

    test_loss, test_acc = model.evaluate(x_test, y_test)

    print('Test accuracy:', test_acc)
    print('Test loss:', test_loss)

    # predict root clause, by pass individual features

    # constants
    CPU_LOAD = 1
    MEMORY_LOAD = 0
    DELAY = 0
    ERROR_1000 = 0
    ERROR_1001 = 1
    ERROR_1002 = 1
    ERROR_1003 = 0

    prediction = np.argmax(model.predict(np.array([[CPU_LOAD, MEMORY_LOAD, DELAY, ERROR_1000,
                                                    ERROR_1001, ERROR_1002, ERROR_1003]])), axis=-1)

    encoded = label_encoder.inverse_transform(prediction)

    print('Predicted root cause:', encoded)

    # predict a batch of data
    data = np.array([[1, 0, 0, 0, 1, 1, 0],
                     [0, 1, 1, 1, 0, 0, 0],
                     [1, 1, 0, 1, 1, 0, 1],
                     [0, 0, 0, 0, 0, 1, 0],
                     [1, 0, 1, 0, 1, 1, 1]])

    predictions = np.argmax(model.predict(data), axis=1)

    encoded = label_encoder.inverse_transform(predictions)

    print('Predicted root causes:', encoded)


if __name__ == '__main__':
    main()
