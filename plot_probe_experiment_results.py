import matplotlib.pyplot as plt
import re
import json

def plot_accuracy_by_iterations(
    results_datapath,
    model_prefixes,
    max_train_games_list=None,
    num_epochs_list=None,
    probe_datasets=None,
    training_configs=None,
    selection_arg="probe_dataset",
    debugging=False
):
    """
    Plots accuracy vs. iterations, filtered by the given criteria.
    
    Args:
        data (list of dict): The JSON-like dataset.
        model_prefixes (list of str): Which model prefixes to plot. Each prefix gets its own color.
        max_train_games_list (list of str or int, optional): Only include rows if max_train_games is in this list.
        num_epochs_list (list of str or int, optional): Only include rows if num_epochs is in this list.
        probe_datasets (list of str, optional): Only include rows if probe_dataset is in this list.
        training_configs (list of str, optional): Only include rows if training_config is in this list.
        selection_arg (str): Must be one of {"num_epochs", "max_train_games", "probe_dataset", "training_config"}.
                             Controls which column is used to vary the marker shape.
    """

    with open(results_datapath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # --- 0) Normalize optional lists to sets for quick membership checks
    if max_train_games_list is not None:
        max_train_games_list = set(str(x) for x in max_train_games_list)
    if probe_datasets is not None:
        probe_datasets = set(probe_datasets)
    if training_configs is not None:
        training_configs = set(training_configs)
    
    # --- 1) A small helper to parse out the numeric iteration from model_name suffix
    #     e.g. "big_random16M_vocab32_50K" -> 50000, "lichess_karvhyp_500K" -> 500000
    def extract_iterations(model_name):
        # Look for a pattern like "_(\d+)K" at the end
        match = re.search(r"_(\d+)K$", model_name)
        if match:
            return int(match.group(1)) * 1000
        return None

    # --- 2) Filter the data
    filtered_data = []
    for row in data:
        if debugging:
            print(f"model {row}")
        # skip if doesn't start with any of the chosen model_prefixes
        if not any(row["model_name"].startswith(pref) for pref in model_prefixes):
            continue
        if debugging:
            print(1)
        # check max_train_games
        if max_train_games_list is not None:
            if str(row["max_train_games"]) not in max_train_games_list:
                continue
        if debugging:
            print(2)
         # check num_epochs
        if num_epochs_list is not None:
            if int(row["num_epochs"]) not in num_epochs_list:
                continue
        if debugging:
            print(3)
        # check probe_dataset
        if probe_datasets is not None:
            if row["probe_dataset"] not in probe_datasets:
                continue
        # check training_config
        if debugging:
            print(4)
        if training_configs is not None:
            if row["training_config"] not in training_configs:
                continue
        
        # parse out the iteration
        iters = extract_iterations(row["model_name"])
        if iters is None:
            continue  # or skip if we can't parse
        row["iterations"] = iters
        
        filtered_data.append(row)

    # --- 3) Group data by model_prefix for plotting lines in different colors
    #     Then within each group, we'll split by selection_arg to vary markers
    grouped_by_prefix = {}
    print(f"length of filtered {len(filtered_data)}")
    for row in filtered_data:
        # find which prefix (from the user-specified list) actually matches
        prefix_match = None
        for pref in model_prefixes:
            if row["model_name"].startswith(pref):
                prefix_match = pref
                break
        if prefix_match is None:
            continue
        
        if prefix_match not in grouped_by_prefix:
            grouped_by_prefix[prefix_match] = []
        grouped_by_prefix[prefix_match].append(row)

    # --- 4) We will want a distinct marker shape for each distinct value of selection_arg
    #     Let's gather them first
    selection_values = set()
    for row in filtered_data:
        selection_values.add(str(row[selection_arg]))
    
    # Provide some markers (extend as needed)
    markers = ["o", "s", "x", "D", "^", "*", "P", "v", ">"]
    # Map each selection value to a marker
    selection_value_to_marker = {}
    for i, val in enumerate(sorted(selection_values)):
        selection_value_to_marker[val] = markers[i % len(markers)]
    
    # --- 5) We'll define a color cycle for the prefixes
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    prefix_colors = {}
    for i, pref in enumerate(model_prefixes):
        prefix_colors[pref] = color_cycle[i % len(color_cycle)]
    
    # --- 6) Start plotting
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # For legend-handling, we will track lines we create so we can label them, 
    # and also track marker shapes we create so we can label them.
    line_handles = {}
    marker_handles = {}
    
    for prefix in grouped_by_prefix:
        # Sort rows by iteration to connect them in ascending order
        rows_for_prefix = sorted(grouped_by_prefix[prefix], key=lambda r: r["iterations"])
        
        # We'll store (x, y) points for each selection-arg-value so that we can do
        # one "plot" call per selection-arg-value. 
        # This way each plot call gets a distinct marker, but the same color for the prefix.
        # Then the line is continuous (since sorted by iteration).
        data_by_sel_val = {}
        for r in rows_for_prefix:
            val = str(r[selection_arg])
            if val not in data_by_sel_val:
                data_by_sel_val[val] = {"iterations": [], "accuracies": []}
            data_by_sel_val[val]["iterations"].append(r["iterations"])
            data_by_sel_val[val]["accuracies"].append(r["accuracy"])
        
        # Now plot each selection-value subset
        for sel_val, subdata in data_by_sel_val.items():
            mk = selection_value_to_marker[sel_val]
            col = prefix_colors[prefix]
            
            line_obj, = ax.plot(
                subdata["iterations"],
                subdata["accuracies"],
                color=col,
                marker=mk,
                label=f"{prefix}, {selection_arg}={sel_val}"
            )
            
            # Keep references for legend
            # We'll only label each prefix once in the "line legend" sense,
            # and only label each selection value once in the "marker legend" sense.
            # However, a simpler approach is to let each combination have a single legend entry.
            # We'll do that simpler approach here.
    
    # If you'd rather have separate legends for prefix vs selection arg, you can do that, 
    # but it requires custom legend handling. 
    # By default, each line above gets a legend entry with the combined label.

    ax.set_xlabel("Iterations")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs. Iterations")
    ax.grid(True)
    ax.legend(loc="best")
    
    plt.tight_layout()
    plt.show()



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
