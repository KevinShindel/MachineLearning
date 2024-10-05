"""
Autor: Kevin Shindel
Date: 2024-03-01

IT Operations: Root Cause Analysis
    - A data center teams wants to build a model to predict causes of issues reported by customers.
    - They use a system monitoring tool to track CPU, memory and application latency for all their servers.
    - In addition, they also track specific errors reported by applications.
"""

# import libraries
from utils import get_rca_data, base_model_config, create_and_run_model, plot_graph


def predict_causes():
    # load data and process data
    symptom_data = get_rca_data()
    accuracy_measures = {}
    layer_count = 2

    # Base Minimal Model
    model_config = base_model_config()
    model_config["HIDDEN_NODES"] = [16]
    model_config["NORMALIZATION"] = None
    model_config["OPTIMIZER"] = "rmsprop"
    model_config["LEARNING_RATE"] = 0.001
    model_config["REGULARIZER"] = None
    model_config["DROPOUT_RATE"] = 0.0

    X, Y = get_rca_data()

    model_name = "Base-Model-" + str(layer_count)

    history = create_and_run_model(model_config, X, Y, model_name)

    accuracy_measures[model_name] = history.history["accuracy"]

    # Adding all optimizations
    model_config = base_model_config()
    model_config["HIDDEN_NODES"] = [32, 32]
    model_config["NORMALIZATION"] = "batch"
    model_config["OPTIMIZER"] = "rmsprop"
    model_config["LEARNING_RATE"] = 0.001
    model_config["REGULARIZER"] = "l2"
    model_config["DROPOUT_RATE"] = 0.2

    X, Y = get_rca_data()

    model_name = "Optimized-Model-" + str(layer_count)

    history = create_and_run_model(model_config, X, Y, model_name)

    accuracy_measures[model_name] = history.history["accuracy"]

    plot_graph(accuracy_measures, "Compare Cumulative Improvements")


def experiment_w_hyperparams():
    # TODO: Implement hyperparameter tuning
    pass


if __name__ == '__main__':
    predict_causes()
