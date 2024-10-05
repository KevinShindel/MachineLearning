from utils import base_model_config, get_data, create_and_run_model, plot_graph


def iteration_nodes_experiment():
    accuracy_measures = {}

    for node_count in range(8, 40, 8):

        # have 2 hidden layers in the networks
        layer_list = []
        for layer_count in range(2):
            layer_list.append(node_count)

        model_config = base_model_config()
        X, Y = get_data()

        model_config["HIDDEN_NODES"] = layer_list
        model_name = "Nodes-" + str(node_count)
        history = create_and_run_model(model_config, X, Y, model_name)

        accuracy_measures[model_name] = history.history["accuracy"]

    plot_graph(accuracy_measures, "Compare Batch Size and Epoch")


def experiment_w_best_hyperparams():
    accuracy_measures = {}
    node_count = 32
    layer_count = 4

    model_config = base_model_config()

    # have 2 hidden layers in the networks
    model_config["HIDDEN_NODES"] = [node_count] * layer_count
    model_config["EPOCHS"] = 25
    model_config["BATCH_SIZE"] = 16

    X, Y = get_data()

    model_name = "Nodes-" + str(node_count)
    history = create_and_run_model(model_config, X, Y, model_name)

    accuracy_measures[model_name] = history.history["accuracy"]

    plot_graph(accuracy_measures, "Compare Batch Size and Epoch")

    print(history.history["accuracy"])


if __name__ == '__main__':
    # iteration_nodes_experiment()
    experiment_w_best_hyperparams()
