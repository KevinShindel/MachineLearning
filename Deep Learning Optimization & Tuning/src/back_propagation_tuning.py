from utils import create_and_run_model, get_data, plot_graph, base_model_config


def iteration_experiment():
    accuracy_measures = {}

    normalization_list = ['none', 'batch']
    for normalization in normalization_list:
        model_config = base_model_config()
        X, Y = get_data()

        model_config["NORMALIZATION"] = normalization
        model_name = "Normalization-" + normalization
        history = create_and_run_model(model_config, X, Y, model_name)

        accuracy_measures[model_name] = history.history["accuracy"]

    plot_graph(accuracy_measures, "Compare Batch Size and Epoch")


def experiment_w_best_hyperparams():
    model_config = base_model_config()
    X, Y = get_data()

    model_config["NORMALIZATION"] = 'batch'
    model_config["HIDDEN_NODES"] = [32] * 2
    model_config["EPOCHS"] = 25
    model_config["BATCH_SIZE"] = 16

    model_name = "Normalization-batch"
    history = create_and_run_model(model_config, X, Y, model_name)
    plot_graph({model_name: history.history["accuracy"]},
               "Compare Batch Size and Epoch")

    accuracy_and_epoch = zip(history.history["accuracy"], history.epoch)
    # best epoch by accuracy
    best_epoch = max(accuracy_and_epoch, key=lambda x: x[0])[1]
    print(f'Best epoch by accuracy: {best_epoch}')


if __name__ == '__main__':
    # iteration_experiment()
    experiment_w_best_hyperparams()
