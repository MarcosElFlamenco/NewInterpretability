import matplotlib.pyplot as plt
import re
import json


def plot_probe_loss(
    results_datapath, 
    model_names=None, 
    training_configs=None, 
    probe_datasets=None, 
    max_train_games=None, 
    min_epochs=None, 
    max_epochs=None
):
    """
    Plots the accuracy list for experiment results based on given filtering criteria.

    Parameters:
        experiment_results (list): List of experiment dictionaries.
        model_names (list): List of model names to include.
        training_configs (list): List of training configurations to include.
        probe_datasets (list): List of probe datasets to include.
        max_train_games (list): List of max_train_games to include.
        min_epochs (int): Minimum number of epochs to consider for plotting.
        max_epochs (int): Maximum number of epochs to consider for plotting.

    Returns:
        None: Plots the results.
    """
    with open(results_datapath, 'r', encoding='utf-8') as f:
        experiment_results = json.load(f)
    filtered_results = []

    # Filter the experiment results based on provided criteria
    for result in experiment_results:
        if model_names and result["model_name"] not in model_names:
            continue
        if training_configs and result["training_config"] not in training_configs:
            continue
        if probe_datasets and result["probe_dataset"] not in probe_datasets:
            continue
        if max_train_games and result["max_train_games"] not in max_train_games:
            continue

        filtered_results.append(result)

    # Plot the accuracy lists for the filtered results
    plt.figure(figsize=(10, 6))

    for result in filtered_results:
        accuracy_list = result["accuracy_list"]
        num_epochs = result["num_epochs"]

        # Apply epoch filtering if specified
        if min_epochs:
            accuracy_list = accuracy_list[min_epochs:]
        if max_epochs:
            accuracy_list = accuracy_list[:max_epochs]

        label = (
            f"Model: {result['model_name']}, Config: {result['training_config']}, "
            f"Dataset: {result['probe_dataset']}, Max Games: {result['max_train_games']}"
        )

        plt.plot(accuracy_list, label=label)

    plt.xlabel("Iters")
    plt.ylabel("Accuracy")
    plt.title("Probe Accuracy Over Iters")
    plt.legend()
    plt.grid(True)
    plt.show()
