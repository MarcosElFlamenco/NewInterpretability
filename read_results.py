#!/usr/bin/env python3
"""
read_results.py

This script reads the experiment_tracking.json file generated by run_experiments.py
and filters the data by command-line arguments to display the results in a meaningful way.

Author: Your Name
"""
import argparse
import json
import os

TRACKING_FILE = "experiment_tracking.json"


def parse_arguments():
    parser = argparse.ArgumentParser(description="Read and filter experiment tracking data.")
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="List of model names to filter on. If omitted, show all models."
    )
    parser.add_argument(
        "--probe_datasets",
        nargs="*",
        default=None,
        help="List of probe datasets to filter on. If omitted, show all."
    )
    parser.add_argument(
        "--training_configs",
        nargs="*",
        default=None,
        help="List of training configs to filter on. If omitted, show all."
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    if not os.path.isfile(TRACKING_FILE):
        print(f"[ERROR] Tracking file '{TRACKING_FILE}' does not exist. Exiting.")
        return
    
    with open(TRACKING_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # data should be a list of dicts, each with keys:
    #   model_name, probe_dataset, training_config, accuracy, ...
    # Filter based on command-line arguments
    filtered_data = []
    for entry in data:
        model_matches = True
        probe_matches = True
        config_matches = True
        
        if args.models is not None and len(args.models) > 0:
            model_matches = entry.get("model_name") in args.models
        if args.probe_datasets is not None and len(args.probe_datasets) > 0:
            probe_matches = entry.get("probe_dataset") in args.probe_datasets
        if args.training_configs is not None and len(args.training_configs) > 0:
            config_matches = entry.get("training_config") in args.training_configs
        
        if model_matches and probe_matches and config_matches:
            filtered_data.append(entry)
    
    if not filtered_data:
        print("[INFO] No results match the specified filters.")
        return
    
    # Print results in a meaningful way
    print("Filtered Experiment Results:")
    print("---------------------------------------------------------")
    for entry in filtered_data:
        model_name = entry.get("model_name")
        probe_dataset = entry.get("probe_dataset")
        training_config = entry.get("training_config")
        accuracy = entry.get("accuracy")
        print(f"Model: {model_name}, Probe Dataset: {probe_dataset}, Training Config: {training_config}, Accuracy: {accuracy}")
    print("---------------------------------------------------------")
    print(f"Total results found: {len(filtered_data)}")


if __name__ == "__main__":
    main()