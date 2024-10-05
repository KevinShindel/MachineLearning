from DLOptimizationAndTuning.src.utils import base_model_config, get_data, create_and_run_model, plot_graph


def iteration_weights_experiment():
    accuracy_measures = {}

    initializer_list = ["random_normal",
                        'zeros',
                        'ones',
                        "random_uniform",
                        "he_normal",
                        "he_uniform",
                        "lecun_normal",
                        "lecun_uniform",
                        "glorot_normal",
                        "glorot_uniform"]

    for initializer in initializer_list:
        model_config = base_model_config()
        X, Y = get_data()

        model_config["WEIGHTS_INITIALIZER"] = initializer
        model_name = "Model-" + initializer
        history = create_and_run_model(model_config, X, Y, model_name)

        accuracy_measures[model_name] = history.history["accuracy"]

    plot_graph(accuracy_measures, "Compare Batch Size and Epoch")


def experiment_w_best_hyperparams():
    accuracy_measures = {}
    initializer = "he_normal"
    activation = 'tanh'
    node_count = 32
    layer_count = 4

    model_config = base_model_config()

    model_config["WEIGHTS_INITIALIZER"] = initializer
    model_config["EPOCHS"] = 25
    model_config["BATCH_SIZE"] = 16
    model_config["HIDDEN_ACTIVATION"] = activation
    model_config["HIDDEN_NODES"] = [node_count] * layer_count

    X, Y = get_data()

    model_name = "Model-" + initializer
    history = create_and_run_model(model_config, X, Y, model_name)

    accuracy_measures[model_name] = history.history["accuracy"]

    plot_graph(accuracy_measures, "Compare Batch Size and Epoch")

    print(history.history["accuracy"])

if __name__ == '__main__':
    # iteration_weights_experiment()
    experiment_w_best_hyperparams()
