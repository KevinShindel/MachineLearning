from DLOptimizationAndTuning.src.utils import base_model_config, get_data, create_and_run_model, plot_graph


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
    pass


if __name__ == '__main__':
    iteration_experiment()
