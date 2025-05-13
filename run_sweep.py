#!/usr/bin/env python3
"""
run_sweep.py - Script to run Optuna hyperparameter sweeps properly

This script runs an Optuna hyperparameter sweep using Hydra's multirun feature,
ensuring that the sweep parameters correctly override the default values.
"""

# Force matplotlib to use non-interactive backend
# This MUST be done before any other imports that might use matplotlib
import os
os.environ['MPLBACKEND'] = 'Agg'  # Force non-interactive backend
os.environ['PYTHONUNBUFFERED'] = '1'  # Ensure output is immediately flushed

import argparse
import subprocess
import sys


def run_sweep(n_trials=30, max_epochs=10):
    """Run an Optuna hyperparameter sweep with the specified number of trials and epochs."""
    
    # Build the sweep command with direct parameter overrides
    command = [
        "python", "main.py", "-m",
        "hydra/sweeper=optuna",
        f"hydra.sweeper.n_trials={n_trials}",
        f"trainer.max_epochs={max_epochs}",
        # Force matplotlib to use Agg backend in subprocess too
        "+MPLBACKEND=Agg" 
    ]
    
    # Parameters to sweep over
    sweep_params = [
        # Data Parameters
        "data.sequence_length=range(3, 37)",
        "data.batch_size=range(2, 17)",
        
        # LSTM Parameters
        "model.lstm_hidden_dim=range(8, 513)",
        "model.n_lstm_layers=1,2,3,4",
        "model.lstm_dropout=0.1,0.2,0.3,0.5",
        
        # CNN Parameters
        "model.cnn_init_dim=range(10, 200)",
        "model.cnn_depth=1,2,3,4",
        "model.cnn_dropout_rate=0.1,0.2,0.3,0.5",
        
        # Training Parameters
        "training.lr=1e-5,5e-5,1e-4,5e-4,1e-3,5e-3,1e-2,5e-2"
    ]
    
    # Add sweep parameters to command
    command.extend(sweep_params)
    
    # Print the command
    print("Running Optuna sweep with command:")
    print(" ".join(command))
    
    # Run the command
    try:
        # Set environment variables for the subprocess
        env = os.environ.copy()
        env['MPLBACKEND'] = 'Agg'
        
        subprocess.run(command, check=True, env=env)
    except subprocess.CalledProcessError as e:
        print(f"Sweep failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run hyperparameter optimization sweep")
    parser.add_argument("--trials", type=int, default=20, help="Number of trials to run")
    parser.add_argument("--epochs", type=int, default=3, help="Maximum epochs per trial")
    args = parser.parse_args()
    
    run_sweep(n_trials=args.trials, max_epochs=args.epochs) 