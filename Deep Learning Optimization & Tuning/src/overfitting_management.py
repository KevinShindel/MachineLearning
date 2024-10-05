from utils import base_model_config, get_data, create_and_run_model, plot_graph


def iteration_overfitting_experiment():
    accuracy_measures = {}

    regularizer_list = ['l1', 'l2']
    for regularizer in regularizer_list:
        model_config = base_model_config()
        X, Y = get_data()

        model_config["REGULARIZER"] = regularizer
        model_config["EPOCHS"] = 25
        model_name = "Regularizer-" + regularizer
        history = create_and_run_model(model_config, X, Y, model_name)

        # Switch to validation accuracy
        accuracy_measures[model_name] = history.history["val_accuracy"]

    plot_graph(accuracy_measures, "Compare Regularizers")


def experiment_w_best_hyperparams():
    # TODO: Create method with tunned hyperparameters
    pass


if __name__ == '__main__':
    iteration_overfitting_experiment()
    # experiment_w_best_hyperparams()
