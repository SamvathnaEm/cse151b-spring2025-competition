#!/usr/bin/env python3
"""
analyze_optuna_results.py

Script to analyze the results of an Optuna hyperparameter optimization sweep.
Run this script after completing a multirun sweep to see the best hyperparameters.

Usage:
    python analyze_optuna_results.py [multirun_dir]

If multirun_dir is not provided, it will look for the most recent multirun directory.
"""

import argparse
import glob
import json
import os
import pandas as pd
from pathlib import Path
import yaml


def find_latest_multirun_dir(base_dir="multirun"):
    """Find the most recent multirun directory."""
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Multirun directory {base_dir} not found.")
    
    dirs = sorted(glob.glob(f"{base_dir}/*/*/*"), key=os.path.getctime, reverse=True)
    if not dirs:
        raise FileNotFoundError(f"No multirun directories found in {base_dir}.")
    
    return dirs[0]


def load_yaml_file(file_path):
    """Load a YAML file and return its contents."""
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)


def find_config_files(multirun_dir):
    """Find all config.yaml files in the multirun directory."""
    return list(glob.glob(f"{multirun_dir}/*/.hydra/config.yaml"))


def find_optimization_metric_files(multirun_dir):
    """Find all optimization_metric.json files in the multirun directory."""
    return list(glob.glob(f"{multirun_dir}/*/optimization_metric.json"))


def extract_hyperparameters(config_yaml):
    """Extract hyperparameters from a config.yaml file."""
    config = load_yaml_file(config_yaml)
    
    # Extract the hyperparameters you care about
    hyperparams = {
        "data.sequence_length": config.get("data", {}).get("sequence_length"),
        "data.batch_size": config.get("data", {}).get("batch_size"),
        "model.lstm_hidden_dim": config.get("model", {}).get("lstm_hidden_dim"),
        "model.n_lstm_layers": config.get("model", {}).get("n_lstm_layers"),
        "model.lstm_dropout": config.get("model", {}).get("lstm_dropout"),
        "model.cnn_init_dim": config.get("model", {}).get("cnn_init_dim"),
        "model.cnn_depth": config.get("model", {}).get("cnn_depth"),
        "model.cnn_dropout_rate": config.get("model", {}).get("cnn_dropout_rate"),
        "training.lr": config.get("training", {}).get("lr"),
    }
    
    return hyperparams


def load_metric(metric_file):
    """Load the optimization metric from a JSON file."""
    try:
        with open(metric_file, 'r') as f:
            data = json.load(f)
            return data.get("value", float('inf'))
    except (json.JSONDecodeError, FileNotFoundError):
        return float('inf')


def main():
    parser = argparse.ArgumentParser(description="Analyze Optuna sweep results.")
    parser.add_argument("multirun_dir", nargs="?", help="Path to the multirun directory.")
    args = parser.parse_args()
    
    # Find the multirun directory if not provided
    multirun_dir = args.multirun_dir or find_latest_multirun_dir()
    print(f"Analyzing Optuna results in: {multirun_dir}")
    
    # Get all config and metric files
    config_files = find_config_files(multirun_dir)
    metric_files = find_optimization_metric_files(multirun_dir)
    
    # Map each job to its parent directory for matching
    config_dirs = {os.path.dirname(os.path.dirname(cf)): cf for cf in config_files}
    metric_dirs = {os.path.dirname(mf): mf for mf in metric_files}
    
    # Collect data for each trial
    trials = []
    for job_dir in config_dirs:
        if job_dir in metric_dirs:
            config_file = config_dirs[job_dir]
            metric_file = metric_dirs[job_dir]
            
            hyperparams = extract_hyperparameters(config_file)
            metric_value = load_metric(metric_file)
            
            trial_data = {
                "job_dir": job_dir,
                "val_loss": metric_value,
                **hyperparams
            }
            trials.append(trial_data)
    
    # Convert to DataFrame and sort by metric value
    if trials:
        df = pd.DataFrame(trials).sort_values("val_loss")
        
        # Print the best trial
        print("\n=== Best Trial ===")
        best_trial = df.iloc[0].to_dict()
        for k, v in best_trial.items():
            if k != "job_dir":
                print(f"{k}: {v}")
        
        # Print the table of all trials
        print("\n=== All Trials ===")
        print(df.to_string(index=False))
        
        # Save to CSV for later analysis
        csv_path = os.path.join(multirun_dir, "optuna_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to: {csv_path}")
    else:
        print("No trials found or trials didn't complete successfully.")


if __name__ == "__main__":
    main() 