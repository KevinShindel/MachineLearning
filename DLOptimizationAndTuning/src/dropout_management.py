from DLOptimizationAndTuning.src.utils import base_model_config, get_data, create_and_run_model, plot_graph

if __name__ == '__main__':

    accuracy_measures = {}

    dropout_list = [0.0, 0.1, 0.2, 0.5]
    for dropout in dropout_list:
        model_config = base_model_config()
        X, Y = get_data()

        model_config["DROPOUT_RATE"] = dropout
        model_config["EPOCHS"] = 25
        model_name = "Dropout-" + str(dropout)
        history = create_and_run_model(model_config, X, Y, model_name)

        # Using validation accuracy
        accuracy_measures[model_name] = history.history["val_accuracy"]

    plot_graph(accuracy_measures, "Compare Dropout")


# TODO: Create method with tunned hyperparameters
