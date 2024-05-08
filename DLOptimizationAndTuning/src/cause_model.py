
"""
Autor: Kevin Shindel
Date: 2024-03-01

IT Operations: Root Cause Analysis
    - A data center teams wants to build a model to predict causes of issues reported by customers.
    - They use a system monitoring tool to track CPU, memory and application latency for all their servers.
    - In addition, they also track specific errors reported by applications.
"""

# import libraries
from DLOptimizationAndTuning.src.utils import get_rca_data, base_model_config, create_and_run_model, plot_graph

if __name__ == '__main__':

    # load data and process data
    symptom_data = get_rca_data()
    # MAX_NODES = 32
    #
    # # Tuning the neural network model (hidden layers)
    # accuracy_measures = {}
    # layer_list = []
    # for layer_count in range(1, 6):
    #     # 32 nodes in each layer
    #     layer_list.append(MAX_NODES)
    #
    #     model_config = base_model_config()
    #     X, Y = get_rca_data()
    #
    #     model_config["HIDDEN_NODES"] = layer_list
    #     model_name = "Layers-" + str(layer_count)
    #     history = create_and_run_model(model_config, X, Y, model_name)
    #
    #     accuracy_measures[model_name] = history.history["accuracy"]
    #
    # plot_graph(accuracy_measures, "Compare Hidden Layers")
    #
    # # Best result is with 3 layers
    #
    # # Tuning the neural network model (nodes in each layer)
    # accuracy_measures = {}
    # node_increment = 8
    #
    # for node_count in range(1, 5):
    #
    #     # have 2 hidden layers in the networks as selected above
    #     layer_list = []
    #     for layer_count in range(2):
    #         layer_list.append(node_count * node_increment)
    #
    #     model_config = base_model_config()
    #     X, Y = get_rca_data()
    #
    #     model_config["HIDDEN_NODES"] = layer_list
    #     model_name = "Nodes-" + str(node_count * node_increment)
    #     history = create_and_run_model(model_config, X, Y, model_name)
    #
    #     accuracy_measures[model_name] = history.history["accuracy"]
    #
    # # plot the graph
    # plot_graph(accuracy_measures, "Compare Nodes per Layers")
    #
    # # Best result is with 32 nodes in each layer
    #
    # # Tuning back propagation algorithm
    #
    # # Tuning the neural network model (use optimizer)
    # accuracy_measures = {}
    #
    # optimizer_list = ['sgd', 'rmsprop', 'adam', 'adagrad']
    # for optimizer in optimizer_list:
    #     model_config = base_model_config()
    #     X, Y = get_rca_data()
    #
    #     model_config["OPTIMIZER"] = optimizer
    #     model_name = "Optimizer-" + optimizer
    #     history = create_and_run_model(model_config, X, Y, model_name)
    #
    #     accuracy_measures[model_name] = history.history["accuracy"]
    #
    # plot_graph(accuracy_measures, "Compare optimizers")

    # Best result is with adam optimizer

    # Tuning the neural network model (use learning rate)

    # accuracy_measures = {}
    #
    # learning_rate_list = [0.001, 0.005, 0.01, 0.1, 0.5]
    # for learning_rate in learning_rate_list:
    #     model_config = base_model_config()
    #     X, Y = get_rca_data()
    #
    #     # Fix Optimizer to the one chosen above
    #     model_config["OPTIMIZER"] = "rmsprop"
    #     model_config["LEARNING_RATE"] = learning_rate
    #     model_name = "Learning-Rate-" + str(learning_rate)
    #     history = create_and_run_model(model_config, X, Y, model_name)
    #
    #     # Using validation accuracy
    #     accuracy_measures[model_name] = history.history["accuracy"]
    #
    # plot_graph(accuracy_measures, "Compare Learning Rates")

    # Best result is with learning rate 0.001

    # Tuning the neural network model (Avoid overfitting)

    # accuracy_measures = {}
    #
    # regularizer_list = [None, 'l1', 'l2']
    # for regularizer in regularizer_list:
    #     model_config = base_model_config()
    #     X, Y = get_rca_data()
    #
    #     model_config["REGULARIZER"] = regularizer
    #     model_name = "Regularizer-" + str(regularizer)
    #     history = create_and_run_model(model_config, X, Y, model_name)
    #
    #     # Switch to validation accuracy
    #     accuracy_measures[model_name] = history.history["val_accuracy"]
    #
    # plot_graph(accuracy_measures, "Compare Regularizers")

    # Best result is with no regularizer

    # Tuning the neural network model (Dropout)

    # accuracy_measures = {}
    #
    # dropout_list = [0.0, 0.1, 0.2, 0.5]
    # for dropout in dropout_list:
    #     model_config = base_model_config()
    #     X, Y = get_rca_data()
    #
    #     # Use the regularizer chosen above
    #     model_config["REGULARIZER"] = "l2"
    #     model_config["DROPOUT_RATE"] = dropout
    #     model_name = "Dropout-" + str(dropout)
    #     history = create_and_run_model(model_config, X, Y, model_name)
    #
    #     # Using validation accuracy
    #     accuracy_measures[model_name] = history.history["val_accuracy"]
    #
    # plot_graph(accuracy_measures, "Compare Dropout")

    # Best result is with dropout rate 0.5

    # Final model
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