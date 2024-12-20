import keras
import pandas as pd
from keras.src.layers import Dense, BatchNormalization, Input
from keras.src.optimizers import SGD, Adagrad, RMSprop, Adam
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from sklearn.datasets import load_iris
from tensorflow.keras import models as k_models


def get_rca_data() -> tuple:
    """
    This function loads the root cause analysis data and returns the feature and target variables.
    """
    symptom_data = pd.read_csv("../dataset/root_cause_analysis.csv", index_col='ID')

    label_encoder = preprocessing.LabelEncoder()
    symptom_data['ROOT_CAUSE'] = label_encoder.fit_transform(
        symptom_data['ROOT_CAUSE'])

    X = symptom_data.drop('ROOT_CAUSE', axis=1)
    y = symptom_data['ROOT_CAUSE']

    return X, y


def base_model_config():
    return {
        "HIDDEN_NODES": [32, 64],
        "HIDDEN_ACTIVATION": "relu",
        "OUTPUT_NODES": 3,
        "OUTPUT_ACTIVATION": "softmax",
        "WEIGHTS_INITIALIZER": "random_normal",
        "BIAS_INITIALIZER": "zeros",
        "NORMALIZATION": "none",
        "OPTIMIZER": "rmsprop",
        "LEARNING_RATE": 0.001,
        "REGULARIZER": None,
        "DROPOUT_RATE": 0.0,
        "EPOCHS": 10,
        "BATCH_SIZE": 16,
        "VALIDATION_SPLIT": 0.2,
        "VERBOSE": 0,
        "LOSS_FUNCTION": "categorical_crossentropy",
        "METRICS": ["accuracy"]
    }


def get_data():
    iris_data = load_iris(as_frame=True)

    # Convert the data to a Pandas DataFrame
    iris_df = pd.DataFrame(data=iris_data.data, columns=iris_data.feature_names)


    # Use a Label encoder to convert String to numeric values for the target variable
    label_encoder = preprocessing.LabelEncoder()
    iris_df['Species'] = label_encoder.fit_transform(
        iris_data['target'])

    # Convert input to numpy array
    np_iris = iris_df.to_numpy()

    # Separate feature and target variables
    X_data = np_iris[:, 0:4]
    Y_data = np_iris[:, 4]

    # Create a scaler model that is fit on the input data.
    X_data = StandardScaler().fit_transform(X_data)

    # Convert target variable as a one-hot-encoding array
    Y_data = tf.keras.utils.to_categorical(Y_data, 3)

    # Return Feature and Target variables
    return X_data, Y_data


def create_and_run_model(model_config, X, Y, model_name):
    model = k_models.Sequential(name=model_name)

    for layer in range(len(model_config["HIDDEN_NODES"])):

        if layer == 0:
            model.add(keras.layers.Input(shape=(X.shape[1],)))
            model.add(
                Dense(model_config["HIDDEN_NODES"][layer],
                      name="Dense-Layer-" + str(layer),
                      kernel_initializer=model_config["WEIGHTS_INITIALIZER"],
                      bias_initializer=model_config["BIAS_INITIALIZER"],
                      kernel_regularizer=model_config["REGULARIZER"],
                      activation=model_config["HIDDEN_ACTIVATION"]))
        else:

            if model_config["NORMALIZATION"] == "batch":
                model.add(BatchNormalization())

            if model_config["DROPOUT_RATE"] > 0.0:
                model.add(keras.layers.Dropout(model_config["DROPOUT_RATE"]))

            model.add(
                keras.layers.Dense(model_config["HIDDEN_NODES"][layer],
                                   name="Dense-Layer-" + str(layer),
                                   kernel_initializer=model_config["WEIGHTS_INITIALIZER"],
                                   bias_initializer=model_config["BIAS_INITIALIZER"],
                                   kernel_regularizer=model_config["REGULARIZER"],
                                   activation=model_config["HIDDEN_ACTIVATION"]))

    model.add(keras.layers.Dense(model_config["OUTPUT_NODES"],
                                 name="Output-Layer",
                                 activation=model_config["OUTPUT_ACTIVATION"]))

    optimizer = get_optimizer(model_config["OPTIMIZER"],
                              model_config["LEARNING_RATE"])

    model.compile(loss=model_config["LOSS_FUNCTION"],
                  optimizer=optimizer,
                  metrics=model_config["METRICS"])

    if model_config['VERBOSE'] > 0:
        print("\n******************************************************")
        model.summary()

    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y,
        stratify=Y,
        test_size=model_config["VALIDATION_SPLIT"])

    history = model.fit(X_train,
                        Y_train,
                        batch_size=model_config["BATCH_SIZE"],
                        epochs=model_config["EPOCHS"],
                        verbose=model_config["VERBOSE"],
                        validation_data=(X_val, Y_val))

    return history


def plot_graph(accuracy_measures, title):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(20, 12))
    for experiment in accuracy_measures.keys():
        plt.plot(accuracy_measures[experiment],
                 label=experiment,
                 linewidth=3)

    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.legend()
    plt.show()


def get_optimizer(optimizer_name, learning_rate):
    # 'sgd','rmsprop','adam','adagrad'

    match optimizer_name:
        case 'sgd':
            optimizer = SGD(learning_rate=learning_rate)
        case 'adagrad':
            optimizer = Adagrad(learning_rate=learning_rate)
        case 'rmsprop':
            optimizer = RMSprop(learning_rate=learning_rate)
        case 'adam':
            optimizer = Adam(learning_rate=learning_rate)
        case _:
            raise ValueError("Optimizer not supported")

    return optimizer


def build_model(input_shape=(0, ), optimizer='adam', learning_rate=0.001):
    _optimizer = get_optimizer(optimizer, learning_rate)

    # Initialising the ANN
    classifier = k_models.Sequential([
        Input(shape=input_shape),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='softmax')
    ])

    # Compiling the ANN
    classifier.compile(loss='mean_absolute_error',
                       # optimizer=_optimizer,
                       optimizer=Adam(0.001),
                       metrics=['mean_absolute_error'])

    return classifier
