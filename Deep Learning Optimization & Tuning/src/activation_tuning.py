from utils import base_model_config, get_data, create_and_run_model, plot_graph


def iterate_activation_functions():
    accuracy_measures = {}

    activation_list = ['relu', 'sigmoid', 'tanh']

    for activation in activation_list:
        model_config = base_model_config()
        X, Y = get_data()

        model_config["HIDDEN_ACTIVATION"] = activation
        model_name = "Model-" + activation
        history = create_and_run_model(model_config, X, Y, model_name)

        accuracy_measures["Model-" + activation] = history.history["accuracy"]

    plot_graph(accuracy_measures, "Compare Batch Size and Epoch")


def experiment_w_best_hyperparams():
    accuracy_measures = {}
    activation = 'tanh' # ReLU - 0.95 / tanh - 0.98
    node_count = 32
    layer_count = 4

    model_config = base_model_config()

    model_config["HIDDEN_ACTIVATION"] = activation
    model_config["EPOCHS"] = 25
    model_config["BATCH_SIZE"] = 16
    model_config["HIDDEN_NODES"] = [node_count] * layer_count

    # have 2 hidden layers in the networks
    X, Y = get_data()

    model_name = "Model-" + activation
    history = create_and_run_model(model_config, X, Y, model_name)

    accuracy_measures[model_name] = history.history["accuracy"]

    plot_graph(accuracy_measures, "Compare Batch Size and Epoch")

    print(history.history["accuracy"])


if __name__ == '__main__':
    # iterate_activation_functions()
    experiment_w_best_hyperparams()
