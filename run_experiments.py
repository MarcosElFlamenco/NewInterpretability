#!/usr/bin/env python3
"""
run_experiments.py

This script automates:
1. Converting PyTorch models into Transformer Lens models (if needed).
2. Training probes on each model + dataset + config combination.
3. Optionally testing the trained probes if a --test flag is provided.
4. Tracking and logging experiment results (accuracy) into a file.

Author: Your Name
"""
import argparse
import os
import subprocess
import itertools
import json
import torch

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
MODEL_FOLDER = "models"
TRANSFORMER_LENS_PREFIX = "tf_lens_"

# Where we expect to find / save our linear probe checkpoints
LINEAR_PROBE_FOLDER = "linear_probes"

# Tracking file to store results
TRACKING_FILE = "experiment_tracking.json"


def parse_arguments():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Automate probe training and optional testing.")
    
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="List of model names. Example: --models modelA modelB"
    )
    parser.add_argument(
        "--probe_datasets",
        nargs="+",
        required=True,
        help="List of probe datasets. Example: --probe_datasets dataset1 dataset2"
    )
    parser.add_argument(
        "--training_configs",
        nargs="+",
        required=True,
        help="List of training config files or config names. Example: --training_configs config1.yaml config2.yaml"
    )
    parser.add_argument(
        "--test_games_datasets",
        nargs="+",
        default=[],
        help="List of test game datasets (only used if --test is provided)."
    )
    parser.add_argument(
        "--max_iters",
        default=50000,
        help="List of test game datasets (only used if --test is provided)."
    )
    parser.add_argument(
        "--max_train_games",
        default=50000,
        help="List of test game datasets (only used if --test is provided)."
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="If set, also run test probe after training."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="If set, print verbose output to the terminal."
    )

    return parser.parse_args()


def load_experiment_tracking(file_path=TRACKING_FILE):
    """
    Loads the JSON tracking file if it exists and returns a list of experiment entries.
    If the file does not exist or is invalid, returns an empty list.
    """
    if os.path.isfile(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # If it's not a list, coerce to a list
                if not isinstance(data, list):
                    data = [data]
                return data
        except (json.JSONDecodeError, OSError):
            # If file is corrupted or unreadable, return an empty list
            return []
    else:
        return []

def append_experiment_entry(entry_dict, file_path=TRACKING_FILE):
    """
    Appends a single experiment dictionary to the existing JSON file.
    Preserves all previously stored experiments by loading them first,
    appending in memory, then overwriting the file.
    """
    # 1) Load existing data
    existing_data = load_experiment_tracking(file_path)
    
    # 2) Append the new entry (a dict) to the list
    existing_data.append(entry_dict)
    
    # 3) Write back the entire list in "w" mode
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, indent=2)


def save_experiment_tracking(data):
    """
    Saves the experiment tracking data (list of dicts) to JSON.
    """
    with open(TRACKING_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def model_exists_as_transformer_lens(model_name_pth):
    """
    Checks if a Transformer Lens version of the model exists.
    E.g. checks for `models/tf_lens_modelName`.
    """
    tf_lens_path = os.path.join(MODEL_FOLDER, TRANSFORMER_LENS_PREFIX + model_name_pth)
    return os.path.isdir(tf_lens_path) or os.path.isfile(tf_lens_path)


def create_transformer_lens_model(model_name_pth, verbose=False):
    """
    Calls model_setup.py with --model_name_pth to create the Transformer Lens version
    if it doesn't already exist.
    """
    if verbose:
        print(f"[INFO] Creating Transformer Lens model for {model_name_pth}...")
    cmd = [
        "python",
        "model_setup.py",
        "--model_name",
        model_name_pth
    ]
    if verbose:
        print(f"[CMD] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def run_train_probe(model_name, probe_dataset, training_config, max_iters, max_train_games, verbose=False):
    """
    Runs the training command for a given (model, probe_dataset, training_config).
    We always train with:
      --mode train
      --probe piece
      --probe_dataset probe_dataset
      --model_name tf_lens_{model_name}
      --training_config training_config
    """
    # The "tf_lens_" prefix is appended to the original model name
    #tf_model_name = TRANSFORMER_LENS_PREFIX + model_name
    
    cmd = [
        "python",
        "train_test_chess.py", 
        "--mode", "train",
        "--probe", "piece",
        "--probe_dataset", probe_dataset,
        "--model_name", model_name,
        "--training_config", training_config,
        "--max_iters", max_iters,
        "--max_train_games", max_train_games
    ]
    
    if verbose:
        print(f"[INFO] Training probe for model={model_name}, dataset={probe_dataset}, config={training_config}")
        print(f"[CMD] {' '.join(cmd)}")
    
    subprocess.run(cmd, check=True)


def run_test_probe(model_name, probe_dataset, test_games_dataset, training_config, verbose=False):
    """
    Runs the test command for a given (model, probe_dataset, test_games_dataset, training_config).
    We always test with:
      --mode test
      --probe piece
      --probe_dataset probe_dataset
      --model_name tf_lens_{model_name}
      --test_games_dataset test_games_dataset
    """
    tf_model_name = TRANSFORMER_LENS_PREFIX + model_name

    cmd = [
        "python",
        "test.py",
        "--mode", "test",
        "--probe", "piece",
        "--probe_dataset", probe_dataset,
        "--model_name", tf_model_name,
        "--test_games_dataset", test_games_dataset
    ]
    if verbose:
        print(f"[INFO] Testing probe for model={model_name}, dataset={probe_dataset}, test_games_dataset={test_games_dataset}, config={training_config}")
        print(f"[CMD] {' '.join(cmd)}")
    
    subprocess.run(cmd, check=True)


def build_checkpoint_filename(model_name, probe_dataset):
    """
    Builds the expected checkpoint filename, e.g.:
    linear_probes/tf_lens_big_random16M_vocab32_150K_chess_piece_probe_type_vanilla_layer_5_test_random.pth

    The user wants the name to be something like:
      linear_probes/<tf_lens_model_name>_chess_piece_probe_type_vanilla_layer_5_test_<probe_dataset>.pth
    
    Adjust as needed for your naming convention. For now, let's do something literal:
    """
    tf_lens_model_name = TRANSFORMER_LENS_PREFIX + model_name
    filename = f"{tf_lens_model_name}_chess_piece_probe_type_vanilla_layer_5_test_{probe_dataset}.pth"
    return os.path.join(LINEAR_PROBE_FOLDER, filename)


def main():
    args = parse_arguments()

    # Load existing experiment logs
    experiment_tracking = load_experiment_tracking()

    # For easy checking, let's keep experiment_tracking as a list of dicts.
    # Each dict might have keys like:
    #  {
    #     "model_name": <model_name>,
    #     "probe_dataset": <probe_dataset>,
    #     "training_config": <training_config>,
    #     "test_games_dataset": <test_games_dataset (optional)>,
    #     "accuracy": float
    #  }

    # Create a set of (model, probe_dataset, training_config) that are completed
    # for faster checking. We'll ignore test_game_dataset for the "train done" check.
    completed_set = set(
        (
            d["model_name"],
            d["probe_dataset"],
            d["training_config"],
            d["max_iters"],
            d["max_train_games"]
        )
        for d in experiment_tracking
        if "accuracy" in d  # or any condition you want
    )

    # Generate all combinations to run
    combos = itertools.product(
        args.models,
        args.probe_datasets,
        args.training_configs
    )

    # TRAINING SECTION
    for model_name, probe_dataset, training_config in combos:
        model_name_pth = model_name + ".pth"
        if args.verbose:
            print("\n----------------------------------------------------")
            print(f"[COMBO] model={model_name}, probe_dataset={probe_dataset}, config={training_config}")
            print("----------------------------------------------------\n")

        # 1) Check if the transformer lens model exists
        if not model_exists_as_transformer_lens(model_name_pth):
            create_transformer_lens_model(model_name_pth, verbose=args.verbose)

        # 2) Check if this combination is already done
        if (model_name, probe_dataset, training_config, args.max_iters, args.max_train_games) in completed_set:
            if args.verbose:
                print("[INFO] This combination already trained. Skipping training step.")
        else:
            # 3) Train the probe
            run_train_probe(model_name, probe_dataset, training_config, args.max_iters, args.max_train_games,verbose=args.verbose)
            
            # 4) Load the resulting checkpoint to get accuracy
            checkpoint_path = build_checkpoint_filename(model_name, probe_dataset)
            if os.path.isfile(checkpoint_path):
                ckpt = torch.load(checkpoint_path, map_location="cpu")
                # Suppose there's an "accuracy" key in the checkpoint

                accuracy = ckpt.get("acc", None).item()
                print(type(accuracy))
                
                if accuracy is not None:
                    if args.verbose:
                        print(f"[INFO] Loaded checkpoint. Accuracy = {accuracy}")
                    # Store the result in our tracking list
                    
                    experiment_tracked = {
                        "model_name": model_name,
                        "probe_dataset": probe_dataset,
                        "training_config": training_config,
                        "max_iters": args.max_iters,
                        "max_train_games": args.max_train_games,
                        "accuracy": accuracy
                    }
                    experiment_tracking.append(experiment_tracked)
                else:
                    if args.verbose:
                        print("[WARNING] 'accuracy' key not found in checkpoint. Skipping logging.")
            else:
                if args.verbose:
                    print(f"[ERROR] Checkpoint file not found: {checkpoint_path}")
        append_experiment_entry(experiment_tracked)

    # Save the tracking data
    #save_experiment_tracking(experiment_tracking)

    # TEST SECTION (if --test is provided)
    if args.test and len(args.test_games_datasets) > 0:
        # We run test for all combinations of model x probe_dataset x training_config x test_games_dataset
        test_combos = itertools.product(
            args.models,
            args.probe_datasets,
            args.training_configs,
            args.test_games_datasets
        )
        for model_name, probe_dataset, training_config, test_dataset in test_combos:
            if args.verbose:
                print("\n----------------------------------------------------")
                print(f"[TEST] model={model_name}, probe_dataset={probe_dataset}, config={training_config}, test_dataset={test_dataset}")
                print("----------------------------------------------------\n")

            run_test_probe(model_name, probe_dataset, test_dataset, training_config, verbose=args.verbose)

    elif args.test and len(args.test_games_datasets) == 0:
        print("[WARNING] --test was set but no --test_games_datasets provided. Nothing to test.")
    
    if args.verbose:
        print("[INFO] All tasks completed. Goodbye!")


if __name__ == "__main__":
    main()