from utils import base_model_config, get_data, create_and_run_model, plot_graph


def iteration_layer_experiment():
    accuracy_measures = {}
    layer_list = []
    for layer_count in range(1, 6):
        # 32 nodes in each layer
        layer_list.append(32)

        model_config = base_model_config()
        X, Y = get_data()

        model_config["HIDDEN_NODES"] = layer_list
        model_name = "Layers-" + str(layer_count)
        history = create_and_run_model(model_config, X, Y, model_name)

        accuracy_measures[model_name] = history.history["accuracy"]

    plot_graph(accuracy_measures, "Compare Batch Size and Epoch")


def experiment_w_best_hyperparams():
    accuracy_measures = {}
    layer_count = 4
    model_config = base_model_config()

    X, Y = get_data()

    model_config["HIDDEN_NODES"] = [32] * layer_count
    model_config["EPOCHS"] = 25
    model_config["BATCH_SIZE"] = 16

    model_name = "Layers-" + str(layer_count)
    history = create_and_run_model(model_config, X, Y, model_name)

    accuracy_measures[model_name] = history.history["accuracy"]

    plot_graph(accuracy_measures, "Compare Batch Size and Epoch")

    print(history.history["accuracy"])


if __name__ == '__main__':
    iteration_layer_experiment()
    experiment_w_best_hyperparams()
