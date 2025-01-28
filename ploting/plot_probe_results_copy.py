import matplotlib.pyplot as plt
import re
import json
from itertools import cycle
from matplotlib.lines import Line2D

def plot_accuracy_by_iterations(
    results_datapath,
    model_prefixes,
    max_train_games_list=None,
    num_epochs_list=None,
    probe_datasets=None,
    training_configs=None,
    selection_arg="probe_dataset",  # Changed to "probe_dataset"
    accuracy_types=None,  # Changed to accept a list
    debugging=-1
):
    """
    Plots accuracy vs. iterations, filtered by the given criteria.

    Args:
        results_datapath (str): Path to the JSON data file.
        model_prefixes (list of str): Which model prefixes to plot. Each prefix gets its own color.
        max_train_games_list (list of str or int, optional): Only include rows if max_train_games is in this list.
        num_epochs_list (list of str or int, optional): Only include rows if num_epochs is in this list.
        probe_datasets (list of str, optional): Only include rows if probe_dataset is in this list.
        training_configs (list of str, optional): Only include rows if training_config is in this list.
        selection_arg (str): Must be one of {"num_epochs", "max_train_games", "probe_dataset", "training_config"}.
                            Controls which column is used to vary the line style.
        accuracy_types (list of str, optional): List of accuracy types to plot. Each type gets its own curve.
        debugging (int, optional): Debugging level. Set to >=0 to enable print statements.
    """

    # Set default accuracy_types if not provided
    if accuracy_types is None:
        accuracy_types = ["train"]
    elif isinstance(accuracy_types, str):
        accuracy_types = [accuracy_types]
    elif not isinstance(accuracy_types, list):
        raise ValueError("accuracy_types must be a string or a list of strings.")

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
        if debugging >= 0:
            print(f"Processing model: {row.get('model_name', 'Unknown')}")
        # Skip if doesn't start with any of the chosen model_prefixes
        if not any(row.get("model_name", "").startswith(pref) for pref in model_prefixes):
            if debugging >= 1:
                print(f"Skipping model {row.get('model_name', 'Unknown')} due to prefix mismatch.")
            continue
        # Check max_train_games
        if max_train_games_list is not None:
            if str(row.get("max_train_games", "")) not in max_train_games_list:
                if debugging >= 1:
                    print(f"Skipping model {row.get('model_name', 'Unknown')} due to max_train_games filter.")
                continue
        # Check num_epochs
        if num_epochs_list is not None:
            if int(row.get("num_epochs", 0)) not in num_epochs_list:
                if debugging >= 1:
                    print(f"Skipping model {row.get('model_name', 'Unknown')} due to num_epochs filter.")
                continue
        # Check probe_dataset
        if probe_datasets is not None:
            if row.get("probe_dataset", "") not in probe_datasets:
                if debugging >= 1:
                    print(f"Skipping model {row.get('model_name', 'Unknown')} due to probe_dataset filter.")
                continue
        # Check training_config
        if training_configs is not None:
            if row.get("training_config", "") not in training_configs:
                if debugging >= 1:
                    print(f"Skipping model {row.get('model_name', 'Unknown')} due to training_config filter.")
                continue
        # Ensure test_results exist
        if "test_results" not in row:
            if debugging >= 1:
                print(f"Skipping model {row.get('model_name', 'Unknown')} because 'test_results' is missing.")
            continue
        # Check accuracy_types
        if accuracy_types is not None:
            # Ensure all specified accuracy_types are present in test_results or 'accuracy' key
            missing_types = []
            for acc_type in accuracy_types:
                if acc_type == "train":
                    # Assume 'accuracy' key for 'train'
                    if "accuracy" not in row:
                        missing_types.append("train")
                else:
                    if acc_type not in row["test_results"]:
                        missing_types.append(acc_type)
            if missing_types:
                if debugging >= 1:
                    print(f"Skipping model {row.get('model_name', 'Unknown')} due to missing accuracy types: {missing_types}")
                continue
        # Parse out the iteration
        iters = extract_iterations(row.get("model_name", ""))
        if iters is None:
            if debugging >= 1:
                print(f"Skipping model {row.get('model_name', 'Unknown')} due to inability to parse iterations.")
            continue  # or skip if we can't parse
        row["iterations"] = iters

        filtered_data.append(row)

    if debugging >= 0:
        print(f"Number of filtered data points: {len(filtered_data)}")

    # --- 3) Group data by model_prefix for plotting lines in different colors
    #     Then within each group, we'll split by selection_arg to vary line styles
    grouped_by_prefix = {}
    for row in filtered_data:
        # Find which prefix (from the user-specified list) actually matches
        prefix_match = None
        for pref in model_prefixes:
            if row["model_name"].startswith(pref):
                prefix_match = pref
                break
        if prefix_match is None:
            continue  # This should not happen due to previous filtering

        if prefix_match not in grouped_by_prefix:
            grouped_by_prefix[prefix_match] = []
        grouped_by_prefix[prefix_match].append(row)

    # --- 4) We will want a distinct line style for each distinct value of selection_arg
    #     Let's gather them first
    selection_values = set()
    for row in filtered_data:
        selection_values.add(str(row.get(selection_arg, "Unknown")))

    # Provide some line styles (extend as needed)
    line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 5))]
    line_style_cycle = cycle(line_styles)
    selection_value_to_linestyle = {}
    for val in sorted(selection_values):
        selection_value_to_linestyle[val] = next(line_style_cycle)

    if debugging >= 0:
        print(f"Selection values and their line styles: {selection_value_to_linestyle}")

    # --- 5) Assign colors to model_prefixes
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    prefix_colors = {}
    color_cycler = cycle(color_cycle)
    for pref in model_prefixes:
        prefix_colors[pref] = next(color_cycler)

    if debugging >= 0:
        print(f"Prefix colors: {prefix_colors}")

    # --- 6) Start plotting
    fig, ax = plt.subplots(figsize=(10, 7))

    for prefix in grouped_by_prefix:
        rows_for_prefix = sorted(grouped_by_prefix[prefix], key=lambda r: r["iterations"])

        # Organize data by selection_val and accuracy_type
        data_by_sel_val = {}
        for r in rows_for_prefix:
            sel_val = str(r.get(selection_arg, "Unknown"))
            if sel_val not in data_by_sel_val:
                data_by_sel_val[sel_val] = {}
                for acc_type in accuracy_types:
                    data_by_sel_val[sel_val][acc_type] = {"iterations": [], "accuracies": []}
            for acc_type in accuracy_types:
                data_by_sel_val[sel_val][acc_type]["iterations"].append(r["iterations"])
                if acc_type == "train":
                    data_by_sel_val[sel_val][acc_type]["accuracies"].append(r.get("accuracy", 0))
                else:
                    data_by_sel_val[sel_val][acc_type]["accuracies"].append(r["test_results"].get(acc_type, 0))

        # Plot each selection_val and accuracy_type
        for sel_val, acc_dict in data_by_sel_val.items():
            linestyle = selection_value_to_linestyle.get(sel_val, '-')
            for acc_type, data_points in acc_dict.items():
                color = prefix_colors[prefix]
                label = f"{prefix}, {selection_arg}={sel_val}, {acc_type}"
                ax.plot(
                    data_points["iterations"],
                    data_points["accuracies"],
                    color=color,
                    linestyle=linestyle,
                    marker='o',  # Optional: You can remove or customize markers
                    label=label
                )

    # --- 7) Create custom legends

    # Legend for Model Prefixes (Colors)
    prefix_handles = [
        Line2D([0], [0], color=prefix_colors[pref], lw=2, label=pref)
        for pref in model_prefixes
    ]

    # Legend for Selection Argument (Line Styles)
    line_handles = [
        Line2D([0], [0], color='black', linestyle=selection_value_to_linestyle[val], lw=2, label=val)
        for val in sorted(selection_values)
    ]

    # Add legends to the plot
    first_legend = ax.legend(handles=prefix_handles, title="Model Prefixes", loc="upper left")
    second_legend = ax.legend(handles=line_handles, title=selection_arg, loc="upper right")
    ax.add_artist(first_legend)  # Add the first legend back

    # --- 8) Final plot adjustments
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs. Iterations")
    ax.grid(True)

    plt.tight_layout()
    plt.show()
