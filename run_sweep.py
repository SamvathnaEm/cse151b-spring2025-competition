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


def run_sweep(n_trials=50, max_epochs=20):
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
        # Data Parameters - For climate, sequence length is particularly important 
        "data.sequence_length=choice(3,6,12,18,24,36)",  # Climate-relevant timescales (months)
        "data.batch_size=choice(2,4,8)",  # Smaller batches to avoid OOM
        
        # LSTM Parameters - For temporal patterns
        "model.lstm_hidden_dim=choice(32,64,128,256)",  # Powers of 2 for efficiency
        "model.n_lstm_layers=range(1,3)",  # Stack depth for temporal processing
        "model.lstm_dropout=interval(0.1, 0.5)",  # Continuous dropout range
        
        # CNN Parameters - For spatial patterns
        "model.cnn_init_dim=choice(16,32,64,128)",  # Powers of 2 for efficiency
        "model.cnn_depth=choice(1,2,3)",  # Reduced to avoid OOM
        "model.cnn_dropout_rate=interval(0.1, 0.5)",  # Continuous dropout
        "model.cnn_kernel_size=choice(3,5,7)",  # New! Different kernel sizes for spatial context
        
        # Training Parameters
        "training.lr=interval(1e-6, 1e-2)",  # Log-scale for learning rate
        "trainer.callbacks.2.patience=range(2,11)",  # New! Early stopping patience
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
    parser.add_argument("--trials", type=int, default=100, help="Number of trials to run")
    parser.add_argument("--epochs", type=int, default=15, help="Maximum epochs per trial")
    args = parser.parse_args()
    
    run_sweep(n_trials=args.trials, max_epochs=args.epochs) 