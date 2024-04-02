from DLOptimizationAndTuning.utils import base_model_config, get_data, create_and_run_model, plot_graph

if __name__ == '__main__':

    # Initialize the measures
    accuracy_measures = {}

    for batch_size in range(16, 128, 16):
        # Load default configuration
        model_config = base_model_config()
        # Acquire and process input data
        X, Y = get_data()

        # set epoch to 20
        model_config["EPOCHS"] = 30
        # Set batch size to experiment value
        model_config["BATCH_SIZE"] = batch_size
        model_name = "Batch-Size-" + str(batch_size)
        history = create_and_run_model(model_config, X, Y, model_name)

        accuracy_measures[model_name] = history.history["accuracy"]

    plot_graph(accuracy_measures, "Compare Batch Size and Epoch")

    # Optimal Epoch and Batch Size
    # Load default configuration
    model_config = base_model_config()
    # Acquire and process input data
    X, Y = get_data()

    # set epoch to 20
    model_config["EPOCHS"] = 25
    # Set batch size to experiment value
    model_config["BATCH_SIZE"] = 16
    model_name = "Batch-Size-" + str(16)
    history = create_and_run_model(model_config, X, Y, model_name)

    accuracy_measures[model_name] = history.history["accuracy"]

    plot_graph(accuracy_measures, "Compare Batch Size and Epoch")