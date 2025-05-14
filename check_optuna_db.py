#!/usr/bin/env python3
"""
check_optuna_db.py - Script to check the Optuna database and show trial information

This script connects to the Optuna database and displays information about the trials,
including the best trials found so far.
"""

import argparse
import os
import sys
from pathlib import Path

try:
    import optuna
except ImportError:
    print("Optuna is not installed. Please install it with 'pip install optuna'")
    sys.exit(1)

def check_database(db_name="optuna_climate.db", study_name="climate_optimization"):
    """Check the Optuna database and show trial information."""
    
    # Check if database file exists
    db_path = Path(db_name)
    if not db_path.exists():
        print(f"Database file {db_path} does not exist.")
        print("No trials have been run yet, or the database is located elsewhere.")
        return
    
    storage_uri = f"sqlite:///{db_path}"
    
    try:
        # Load the study
        study = optuna.load_study(study_name=study_name, storage=storage_uri)
    except Exception as e:
        print(f"Error loading study: {e}")
        return
    
    # Print basic study information
    print(f"\n{'='*50}")
    print(f"Study: {study_name}")
    print(f"Database: {db_path}")
    print(f"Direction: {study.direction.name}")
    print(f"Number of completed trials: {len(study.trials)}")
    if len(study.trials) == 0:
        print("No trials have been completed yet.")
        return
    
    # Print best trial information
    print(f"\n{'='*50}")
    print("BEST TRIAL:")
    best_trial = study.best_trial
    print(f"  Value: {best_trial.value:.6f}")
    print(f"  Params:")
    
    # Print parameters in a readable format
    for param_name, param_value in best_trial.params.items():
        print(f"    {param_name}: {param_value}")
    
    # Print top 5 trials
    print(f"\n{'='*50}")
    print("TOP 5 TRIALS:")
    
    # Sort trials by value (best first)
    sorted_trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else float('inf'))
    
    # Print top 5 trials (or fewer if less than 5 trials exist)
    for i, trial in enumerate(sorted_trials[:5]):
        if trial.value is None:
            continue
        print(f"\n  Trial {i+1} - Value: {trial.value:.6f}")
        print("  Params:")
        # Print each parameter
        for param_name, param_value in trial.params.items():
            print(f"    {param_name}: {param_value}")
    
    print(f"\n{'='*50}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check Optuna database and show trial information")
    parser.add_argument("--db", type=str, default="optuna_climate.db", help="Database filename")
    parser.add_argument("--study", type=str, default="climate_optimization", help="Optuna study name")
    args = parser.parse_args()
    
    check_database(db_name=args.db, study_name=args.study) 