import matplotlib.pyplot as plt
from collections import defaultdict
import re
import json
from itertools import cycle, product
from matplotlib.lines import Line2D

def plot_accuracy_by_iterations(
    results_datapath,
    model_prefixes,
    max_train_games_list=None,
    num_epochs_list=None,
    probe_datasets=None,
    training_configs=None,
    accuracy_types=None,
    color_means=None,
    line_means=None,
    marker_means=None,
    debugging=-1
):
    """
    Plots accuracy vs. iterations, filtered by the given criteria.

    Args:
        results_datapath (str): Path to the JSON data file.
        model_prefixes (list of str): Which model prefixes to plot.
        max_train_games_list (list of str or int, optional): Filter by max_train_games.
        num_epochs_list (list of str or int, optional): Filter by num_epochs.
        probe_datasets (list of str, optional): Filter by probe_dataset.
        training_configs (list of str, optional): Filter by training_config.
        accuracy_types (list of str, optional): List of accuracy types to plot. Can include "training_set".
        color_means (str, optional): Variable name to map to colors.
        line_means (str, optional): Variable name to map to line styles.
        marker_means (str, optional): Variable name to map to markers.
        debugging (int, optional): Debugging level. Set to >=0 to enable print statements.
    """
    
    # Validate mapping arguments
    valid_means = {"model_prefix", "num_epochs", "max_train_games", "probe_dataset", "training_config", "accuracy_type"}
    for arg, value in zip(["color_means", "line_means", "marker_means"], [color_means, line_means, marker_means]):
        if value is not None and value not in valid_means:
            raise ValueError(f"{arg} must be one of {valid_means}, got '{value}'.")

    # Set default accuracy_types if not provided
    if accuracy_types is None:
        accuracy_types = ["train"]
    elif isinstance(accuracy_types, str):
        accuracy_types = [accuracy_types]
    elif not isinstance(accuracy_types, list):
        raise ValueError("accuracy_types must be a string or a list of strings.")
    
    # Load data
    with open(results_datapath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # --- 0) Normalize optional lists to sets for quick membership checks
    if max_train_games_list is not None:
        max_train_games_list = set(str(x) for x in max_train_games_list)
    if probe_datasets is not None:
        probe_datasets = set(probe_datasets)
    if training_configs is not None:
        training_configs = set(training_configs)
    
    # --- 1) Helper to extract iterations from model_name
    def extract_iterations(model_name):
        match = re.search(r"_(\d+)K$", model_name)
        if match:
            return int(match.group(1)) * 1000
        return None
    
    # --- 2) Filter the data
    filtered_data = []
    for row in data:
        model_name = row.get("model_name", "Unknown")
        if debugging >= 0:
            print(f"Processing model: {model_name}, training_config {row.get("training_config","Unknown")}")
        
        # Filter by model_prefixes
        if not any(model_name.startswith(pref) for pref in model_prefixes):
            if debugging >= 1:
                print(f"Skipping {model_name}: Prefix mismatch.")
            continue
        
        # Filter by max_train_games
        if max_train_games_list is not None:
            if str(row.get("max_train_games", "")) not in max_train_games_list:
                if debugging >= 1:
                    print(f"Skipping {model_name}: max_train_games filter.")
                continue
        
        # Filter by num_epochs
        if num_epochs_list is not None:
            if int(row.get("num_epochs", 0)) not in num_epochs_list:
                if debugging >= 1:
                    print(f"Skipping {model_name}: num_epochs filter.")
                continue
        
        # Filter by probe_datasets
        if probe_datasets is not None:
            if row.get("probe_dataset", "") not in probe_datasets:
                if debugging >= 1:
                    print(f"Skipping {model_name}: probe_dataset filter.")
                continue
        
        # Filter by training_configs
        if training_configs is not None:
            if row.get("training_config", "") not in training_configs:
                if debugging >= 1:
                    print(f"Skipping {model_name}: training_config filter.")
                continue
        
        # Ensure test_results exist
        if "test_results" not in row:
            if debugging >= 1:
                print(f"Skipping {model_name}: 'test_results' missing.")
            continue
        
        # Check accuracy_types
        if accuracy_types is not None:
            missing_types = []
            for acc_type in accuracy_types:
                if acc_type == "train":
                    if "accuracy" not in row:
                        missing_types.append("train")
                elif acc_type == "training_set":
                    # Will handle later based on probe_dataset
                    continue
                else:
                    if acc_type not in row["test_results"]:
                        missing_types.append(acc_type)
            if missing_types:
                if debugging >= 1:
                    print(f"Skipping {model_name}: Missing accuracy types {missing_types}.")
                continue
        
        # Extract iterations
        iters = extract_iterations(model_name)
        if iters is None:
            if debugging >= 1:
                print(f"Skipping {model_name}: Unable to parse iterations.")
            continue
        row["iterations"] = iters
        
        filtered_data.append(row)
    
    if debugging >= 0:
        print(f"Filtered data points: {len(filtered_data)}")
    
    # --- 3) Group data by model_prefix
    grouped_by_prefix = {}
    for row in filtered_data:
        model_name = row["model_name"]
        prefix_match = None
        for pref in model_prefixes:
            if model_name.startswith(pref):
                prefix_match = pref
                break
        if prefix_match is None:
            continue  # Should not happen
        
        if prefix_match not in grouped_by_prefix:
            grouped_by_prefix[prefix_match] = []
        grouped_by_prefix[prefix_match].append(row)
    
    # --- 4) Determine unique values for color_means, line_means, marker_means
    color_values = set()
    line_values = set()
    marker_values = set()
    
    for row in filtered_data:
        if color_means:
            model_name = str(row.get("model_name", "Unknown"))
            color_key = model_name.rsplit('_', 1)[0]
            color_values.add(color_key)
        if line_means:
            line_values.add(str(row.get(line_means, "Unknown")))
        if marker_means:
            marker_values.add(str(row.get(marker_means, "Unknown")))
        # Additionally, accuracy_types can influence markers
        # But since marker_means is separate, we'll handle accuracy_types as part of marker_means if needed
    
    # --- 5) Assign colors, line styles, markers
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_cycler = cycle(color_cycle)
    color_mapping = {val: next(color_cycler) for val in sorted(color_values)}
    
    line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 5)), (0, (1, 10))]
    line_cycler = cycle(line_styles)
    line_mapping = {val: next(line_cycler) for val in sorted(line_values)}
    
    markers = ["o", "s", "x", "D", "^", "*", "P", "v", ">", "<", "h", "H", "d", "p"]
    marker_cycler = cycle(markers)
    marker_mapping = {val: next(marker_cycler) for val in sorted(marker_values)}
    
    # Additionally, map accuracy_types to markers if multiple
    if len(accuracy_types) > 1:
        accuracy_marker_mapping = {}
        for acc_type in accuracy_types:
            accuracy_marker_mapping[acc_type] = next(marker_cycler)
    else:
        accuracy_marker_mapping = {}
    
    if debugging >= 0:
        print(f"Color mapping: {color_mapping}")
        print(f"Line style mapping: {line_mapping}")
        print(f"Marker mapping: {marker_mapping}")
        if accuracy_marker_mapping:
            print(f"Accuracy type marker mapping: {accuracy_marker_mapping}")
    
    # --- 6) Start plotting
    fig, ax = plt.subplots(figsize=(12, 8))

    # Collect legend handles
    color_handles = []
    line_handles = []
    marker_handles = []
    if debugging >= -1:
        print(f"grouped by prefix {grouped_by_prefix.keys()}")


    def third_level_structure():
        return {
            "iterations": [],
            "accuracies": [],
            "color": "",
            "line_style": "",
            "marker": "",
            "label": ""
        }

    # Create a defaultdict for nested structure
    all_info = defaultdict(lambda: defaultdict(lambda: defaultdict(third_level_structure)))

    for prefix, probe_dataset, acc_type in product(model_prefixes, probe_datasets, accuracy_types):
    # Accessing the structure initializes it
        _ = all_info[prefix][probe_dataset][acc_type]


    for prefix, probe_dataset, acc_type in product(model_prefixes, probe_datasets, accuracy_types):
        # Accessing the structure initializes it
        _ = all_info[prefix][probe_dataset][acc_type]
    for prefix, rows in grouped_by_prefix.items():
        # Sort rows by iterations
        sorted_rows = sorted(rows, key=lambda r: r["iterations"])
        
        for row in sorted_rows:
            iterations = row["iterations"]
            probe_dataset = row.get("probe_dataset", "Unknown")
            for acc_type in accuracy_types:
                # Determine accuracy value
                if acc_type == "train":
                    accuracy = row.get("accuracy", None)
                elif acc_type == "training_set":
                    # Map to test_results based on probe_dataset
                    accuracy = row["test_results"].get(probe_dataset, None)
                else:
                    accuracy = row["test_results"].get(acc_type, None)
                
                if accuracy is None:
                    if debugging >= 1:
                        print(f"Missing accuracy for {acc_type} in model {prefix}, probe_dataset {probe_dataset}.")
                    continue
                
                # Determine color
                if color_means:
                    if color_means =="model_prefix":
                        model_name = str(row.get("model_name", "Unknown"))
                        color_key = model_name.rsplit('_', 1)[0]
                        color = color_mapping.get(color_key, 'black')
                    else:
                        color_key = str(row.get(color_means, "Unknown"))
                        color = color_mapping.get(color_key, 'black')
                else:
                    color = 'black'
                
                # Determine line style
                if line_means:
                    line_key = str(row.get(line_means, "Unknown"))
                    line_style = line_mapping.get(line_key, ':')
                else:
                    line_style = '-'
                
                # Determine marker
                if len(accuracy_types) > 1 and acc_type in accuracy_marker_mapping:
                    marker = accuracy_marker_mapping[acc_type]
                elif marker_means:
                    marker_key = str(row.get(marker_means, "Unknown"))
                    marker = marker_mapping.get(marker_key, 'o')
                else:
                    marker = 'o'
                
                # Prepare label
                label_components = []
                if color_means:
                    label_components.append(f"{color_means}={color_key}")
                if line_means:
                    label_components.append(f"{line_means}={line_key}")
                if len(accuracy_types) > 1:
                    label_components.append(f"Accuracy={acc_type}")
                label = ", ".join(label_components)
                all_info[prefix][probe_dataset][acc_type]["iterations"].append(iterations)
                all_info[prefix][probe_dataset][acc_type]["accuracies"].append(accuracy)
                all_info[prefix][probe_dataset][acc_type]["color"] = color
                all_info[prefix][probe_dataset][acc_type]['line_style'] = line_style
                all_info[prefix][probe_dataset][acc_type]["marker"] = marker
                all_info[prefix][probe_dataset][acc_type]["label"] = label

                # Plot
                if debugging >= -1:
                    print(f"ploting {iterations} with {accuracy} color {color} linestyle {line_style} marker {marker} label {label}")
    for prefix, probe_dataset, acc_type in product(model_prefixes, probe_datasets, accuracy_types):
        ax.plot(
            all_info[prefix][probe_dataset][acc_type]["iterations"],
            all_info[prefix][probe_dataset][acc_type]["accuracies"],
            color=all_info[prefix][probe_dataset][acc_type]["color"],
            linestyle=all_info[prefix][probe_dataset][acc_type]["line_style"],
            marker=all_info[prefix][probe_dataset][acc_type]["marker"],
            label=all_info[prefix][probe_dataset][acc_type]["label"]
                    )
        
    # --- 7) Create custom legends
    
    # Legend for Colors
    if color_means:
        color_handles = [
            Line2D([0], [0], color=color_mapping[val], lw=2, label=val)
            for val in sorted(color_values)
        ]
        color_legend = ax.legend(handles=color_handles, title=color_means, loc="upper left")
        ax.add_artist(color_legend)
    
    # Legend for Line Styles
    if line_means:
        line_handles = [
            Line2D([0], [0], color='black', linestyle=line_mapping[val], lw=2, label=val)
            for val in sorted(line_values)
        ]
        line_legend = ax.legend(handles=line_handles, title=line_means, loc="upper right")
        ax.add_artist(line_legend)
    
    # Legend for Markers
    if len(accuracy_types) > 1:
        marker_handles = [
            Line2D([0], [0], marker=accuracy_marker_mapping[acc_type], color='w', label=acc_type,
                   markerfacecolor='gray', markersize=8)
            for acc_type in accuracy_types
        ]
        marker_legend = ax.legend(handles=marker_handles, title="Accuracy Types", loc="lower right")
    elif marker_means:
        marker_handles = [
            Line2D([0], [0], marker=marker_mapping[val], color='w', label=val,
                   markerfacecolor='gray', markersize=8)
            for val in sorted(marker_values)
        ]
        marker_legend = ax.legend(handles=marker_handles, title=marker_means, loc="lower right")
    
    # --- 8) Final plot adjustments
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs. Iterations")
    ax.grid(True)
    
    plt.tight_layout()
    plt.show()
