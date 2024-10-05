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
    # TODO: Implement this function
    pass


if __name__ == '__main__':
    iteration_experiment()
