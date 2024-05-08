from DLOptimizationAndTuning.src.utils import base_model_config, get_data, create_and_run_model, plot_graph

if __name__ == '__main__':
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


# TODO: Create method with tunned hyperparameters