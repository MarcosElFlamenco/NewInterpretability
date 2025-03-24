import matplotlib.pyplot as plt
from collections import defaultdict
import re
import json
from itertools import cycle, product
from matplotlib.lines import Line2D
import torch
import os
from typing import List

def plot_probe_accuracies_by_layer(
    model_names: List[str],
    probe_type: str,
    train_set: str,
    num_layers: int = 8,
    piece_type: str = "chess_piece"
) -> None:
    """
    Plot linear probe accuracies across layers for multiple models.
    
    Args:
        model_names: List of model names to analyze
        probe_type: Type of probe (e.g., 'classic')
        train_set: Training set used (e.g., 'lichess')
        num_layers: Number of layers to analyze
        piece_type: Type of piece probe
    """
    plt.figure(figsize=(10, 6))
    labels = {
        "random_karvhypNSNR_600K" : 'small_random',
        "random_karvhypNSNR_300K" : 'small_random_300',
        "karvmodel_600K" : "karvonen original model",
        "big_random16M_vocab32_300K" : "big_random",
        "gm_karvhyp_300K" : "grandmaster games"
    }
    colors = {
        "random_karvhypNSNR_600K" : "#6ebf06",
        "random_karvhypNSNR_300K" : "#e89915",
        "karvmodel_600K" : "#e31609",
        "big_random16M_vocab32_300K" : "#2292a4",
        "gm_karvhyp_300K" : "#714955", 
    }
    for model_name in model_names:
        accuracies = []
        
        for layer in range(num_layers):
            # Construct checkpoint filename
            checkpoint_name = f"{model_name}_{probe_type}_{train_set}_10000_{piece_type}_probe_layer{layer}.pth"
            checkpoint_path = os.path.join("../linear_probes", checkpoint_name)
            
            try:
                # Load checkpoint and extract accuracy
                checkpoint = torch.load(checkpoint_path)
                accuracy = checkpoint["accuracy"].item()
                accuracies.append(accuracy * 100)  # Convert to percentage
            except Exception as e:
                print(f"Error loading {checkpoint_path}: {e}")
                accuracies.append(None)
        
        # Plot accuracies for this model
        plt.plot(range(num_layers), accuracies, color=colors[model_name], marker='o', label=labels[model_name])
    
    plt.xlabel("Layer")
    plt.ylabel("Accuracy (%)")
    plt.title(f"Linear Probe Accuracies") #({probe_type} - {train_set})")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"probe_accuracies_{probe_type}_{train_set}.png")
    plt.show()


def plot_accuracy_by_iterations(
    results_datapath,
    model_prefixes,
    probe_datasets,
    training_configs,
    accuracy_types,
    color_list=None,
    skip_0=True,
    legend_probe_dataset=True,
    legend_accuracy_type=True,
    grid=False
):
    """
    Plots iteration vs. accuracy for the given model prefixes, probe datasets,
    training configs, and accuracy types, based on data in a JSON file.

    :param results_datapath: Path to the JSON file containing results.
    :param model_prefixes: List of model prefix strings to include.
    :param probe_datasets: List of probe datasets to include.
    :param training_configs: List of training configs to include.
    :param accuracy_types: List of accuracy types to plot (e.g. ["train", "random", "lichess", "training set"]).
    :param color_list: Optional list of colors for each model prefix (must be at least as long as model_prefixes).
    """

    # 1. Read the JSON file
    with open(results_datapath, 'r') as f:
        data = json.load(f)

    # Convert single-record JSON into list if necessary
    if isinstance(data, dict):
        data = [data]

    # 3. Helper: parse model name into (base_prefix, iteration_label).
    #    For example, "random_karvhypNSNR_300K" -> ("random_karvhypNSNR", "300K")
    #    If there's no underscore, treat entire string as prefix, iteration=None
    def split_model_name(model_name):
        if "_" not in model_name:
            return model_name, None
        parts = model_name.rsplit("_", 1)
        prefix_part = parts[0]
        iteration_part = parts[1]
        return prefix_part, iteration_part

    # 4. Helper: parse iteration label like "300K", "1M" into an integer (for sorting)
    #    If you prefer to keep them as strings, skip integer parsing,
    #    but then you'll need to sort them lexicographically or map them to an order.
    def parse_iteration(iter_str):
        """Convert '300K' -> 300000, '1M' -> 1000000, '500k' -> 500000, etc.
           Fall back to try int(), else return None."""
        if not iter_str:
            return None
        # Common patterns: 300K, 500k, 1M, 2m, 10000, ...
        # Try a quick regex:
        m = re.match(r"(\d+(?:\.\d+)?)([kKmM]?)", iter_str)
        if not m:
            # Last fallback: try to parse as int
            try:
                return int(iter_str)
            except ValueError:
                return None
        number_str, suffix = m.groups()
        number_val = float(number_str)
        suffix = suffix.lower()
        if suffix == 'k':
            return int(number_val * 1000)
        elif suffix == 'm':
            return int(number_val * 1000000)
        else:
            return int(number_val)

    # 5. Helper: get the correct accuracy for a given record & accuracy type
    def get_accuracy(record, acc_type):
        """
        "train" -> record["accuracy"]
        "training set" -> record["test_results"][record["probe_dataset"]]
        Otherwise, assume acc_type is a key in test_results, e.g. "random", "lichess"
        """
        if acc_type == "train":
            return record.get("accuracy", None)
        elif acc_type == "training_set":
            pd = record["probe_dataset"]
            return record["test_results"].get(pd, None)
        else:
            return record["test_results"].get(acc_type, None)


    # 2. Filter data by the given arguments
    filtered_data = []
    for record in data:

        # The record must match in all of these:
        if record.get("probe_dataset") in probe_datasets \
           and record.get("training_config") in training_configs:
            # We only keep if the model prefix also matches
            # We'll parse out the actual prefix vs iteration below
            # but let's do a coarse check that the record's model_name
            # starts with one of the prefixes (before final underscore).
            # The function below extracts the prefix (base) and iteration
            prefix, iteration_str = split_model_name(record["model_name"])
            if prefix in model_prefixes:
                filtered_data.append(record)


        # 6. Collect all points we want to plot:
    #    We'll store them in a structure keyed by (prefix, probe_dataset, accuracy_type)
    #    Then each entry is a list of (iteration, accuracy).
    plot_dict = {}
    for record in filtered_data:
        prefix, iteration_str = split_model_name(record["model_name"])
        iteration_val = parse_iteration(iteration_str)
        if iteration_val is None:
            # If you prefer to skip records without valid iteration, you can continue
            # or store them at iteration=0, etc. We'll skip them here:
            continue
        if skip_0 and iteration_val == 0:
            continue

        pd = record["probe_dataset"]
        for acc_type in accuracy_types:
            acc_val = get_accuracy(record, acc_type)
            if acc_val is not None:
                key = (prefix, pd, acc_type)
                if key not in plot_dict:
                    plot_dict[key] = []
                plot_dict[key].append((iteration_val, acc_val))

    # Sort each list of (iteration, accuracy) by iteration
    for key in plot_dict:
        plot_dict[key].sort(key=lambda x: x[0])

    # 7. Prepare line styles, markers, and colors
    # Make sure you have enough styles for however many you have in the lists
    default_colors = ["#6ebf06", "#e89915", "#e31609", "#2292a4", "#714955", 
                      "brown", "pink", "gray", "olive", "cyan"]
    if color_list is None:
        color_list = default_colors

    # Assign color per model_prefix (order as in model_prefixes)
    color_map = {}
    for i, mp in enumerate(model_prefixes):
        # If user-supplied color_list is too short, wrap around with modulo
        color_map[mp] = color_list[i % len(color_list)]

    # Assign line style per probe dataset
    line_styles = ["-", "--", "-.", ":"]
    line_map = {}
    for i, ds in enumerate(probe_datasets):
        line_map[ds] = line_styles[i % len(line_styles)]

    # Assign marker per accuracy type
    marker_styles = ["o", "s", "D", "^", "v", "<", ">", "p", "*", "x"]
    marker_map = {}
    for i, acc_type in enumerate(accuracy_types):
        marker_map[acc_type] = marker_styles[i % len(marker_styles)]

    # 8. Create a single figure + axis
    fig, ax = plt.subplots(figsize=(8,6))

    # Plot all lines
    for (prefix, ds, acc_type), vals in plot_dict.items():
        if not vals:
            continue
        iterations = [v[0] for v in vals]
        accuracies = [v[1] for v in vals]

        ax.plot(
            iterations,
            accuracies,
            color=color_map[prefix],
            linestyle=line_map[ds],
            marker=marker_map[acc_type],
            label=(prefix, ds, acc_type)  # raw label (not used directly in final legends)
        )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs. Iterations by model")

    # 9. Construct three separate legends
    model_names = {
        "lichess_karvhyp" : "lichess",
        "random_karvhypNSNR" : "small_random",
        "big_random16M_vocab32" : "big_random",
        "gm_karvhyp" : "grandmaster",
        "2_random_600" : "big_random2"
    }
    # Legend 1: Model Prefix (colors)
    legend1_handles = []
    legend1_labels = []
    for mp in model_prefixes:
        # A dummy line2D for each prefix color
        handle = Line2D([0], [0], color=color_map[mp], linewidth=3)
        legend1_handles.append(handle)
        legend1_labels.append(model_names[mp])
    legend1 = ax.legend(legend1_handles, legend1_labels, title="Model dataset", loc="upper left")
    ax.add_artist(legend1)

    # Legend 2: Probe Dataset (line styles)
    legend2_handles = []
    legend2_labels = []
    for ds in probe_datasets:
        handle = Line2D([0], [0], color="black", linestyle=line_map[ds], linewidth=2)
        legend2_handles.append(handle)
        legend2_labels.append(ds)
    if legend_probe_dataset:
        legend2 = ax.legend(legend2_handles, legend2_labels, title="Probe training dataset", loc="upper center")
        ax.add_artist(legend2)

    # Legend 3: Accuracy Type (markers)
    legend3_handles = []
    legend3_labels = []
    for acc_type in accuracy_types:
        handle = Line2D([0], [0], color="black", marker=marker_map[acc_type],
                        linewidth=0, markersize=8)
        legend3_handles.append(handle)
        legend3_labels.append(acc_type)
    if legend_accuracy_type:
        legend3 = ax.legend(legend3_handles, legend3_labels, title="Accuracy Type", loc="upper right")
        ax.add_artist(legend3)

    plt.tight_layout()
    plt.grid(grid)
    plt.show()
