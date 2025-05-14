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
from pathlib import Path
import sqlite3
import shutil


def check_optuna_compatibility(db_path):
    """
    Check if the database is compatible with the installed Optuna version.
    If the database exists but has schema compatibility issues, back it up and create a fresh one.
    """
    if not Path(db_path).exists():
        print(f"Database {db_path} does not exist, will create a new one.")
        return True  # No conflict if file doesn't exist
    
    try:
        # Try to open and query the database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check version table
        try:
            cursor.execute("SELECT version_num FROM alembic_version")
            db_version = cursor.fetchone()
            print(f"Database schema version: {db_version[0] if db_version else 'unknown'}")
            conn.close()
            return True  # Seems compatible
        except sqlite3.OperationalError:
            # Table doesn't exist or other issue
            conn.close()
            return True  # Continue and hope Optuna handles it
        
    except Exception as e:
        print(f"Error checking database: {str(e)}")
        
        # Backup the old database and create a new one
        backup_path = f"{db_path}.backup"
        print(f"⚠️ WARNING: Creating backup of existing database at {backup_path}")
        print(f"This is needed due to Optuna version compatibility issues.")
        
        try:
            shutil.copy2(db_path, backup_path)
            print(f"Backup created successfully. Removing old database to create a fresh one.")
            os.remove(db_path)
            return True  # Ready for a fresh database
        except Exception as backup_error:
            print(f"Failed to backup database: {str(backup_error)}")
            return False


def initialize_optuna_database(db_path):
    """Initialize the Optuna database with required tables to prevent schema errors."""
    try:
        # Connect to the database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create the version table that Optuna expects
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS alembic_version (version_num VARCHAR(32) NOT NULL)"
        )
        
        # Commit changes and close connection
        conn.commit()
        conn.close()
        print(f"Successfully initialized database at {db_path}")
        
    except Exception as e:
        print(f"Error initializing database: {e}")
        # Continue execution - let Optuna handle any remaining setup


def run_sweep(n_trials=1, max_epochs=1, db_name="optuna_climate.db", study_name="climate_optimization", force_new_db=False):
    """Run an Optuna hyperparameter sweep with the specified number of trials and epochs."""
    
    # Ensure database directory exists
    db_path = Path(db_name).resolve()
    db_dir = db_path.parent
    db_dir.mkdir(exist_ok=True)
    
    # Initialize the database to prevent schema errors
    initialize_optuna_database(db_path)
    
    # Database connection string (no quotes to avoid command line parsing issues)
    storage_uri = f"sqlite:///{db_path}"
    
    # Build the sweep command with direct parameter overrides
    command = [
        "python", "main.py", "-m",
        "hydra/sweeper=optuna",
        f"hydra.sweeper.n_trials={n_trials}",
        f"trainer.max_epochs={max_epochs}",
        f"hydra.sweeper.storage={storage_uri}",
        f"hydra.sweeper.study_name={study_name}",
        # Force matplotlib to use Agg backend in subprocess too
        "+MPLBACKEND=Agg" 
    ]
    
    # Parameters to sweep over
    sweep_params = [
        # Data Parameters - For climate, sequence length is particularly important 
        "data.sequence_length=range(3,24)",  # Climate-relevant timescales (months)
        "data.batch_size=choice(2,4)",  # Smaller batches to avoid OOM
        
        # LSTM Parameters - For temporal patterns
        "model.lstm_hidden_dim=range(16,256)",  # Powers of 2 for efficiency
        "model.n_lstm_layers=range(1,3)",  # Stack depth for temporal processing
        "model.lstm_dropout=interval(0.05, 0.95)",  # Continuous dropout range
        
        # CNN Parameters - For spatial patterns
        "model.cnn_init_dim=range(16,128)",  # Powers of 2 for efficiency
        "model.cnn_depth=choice(1,2,3)",  # Reduced to avoid OOM
        "model.cnn_dropout_rate=interval(0.05, 0.95)",  # Continuous dropout
        "model.cnn_kernel_size=choice(3,5,7)",  # New! Different kernel sizes for spatial context
        
        # Training Parameters - For log-scale in Hydra, use tags
        "training.lr=tag(log, interval(1e-6, 1e-2))",  # Log-scale for learning rate using proper Hydra syntax
        "++training.weight_decay=tag(log, interval(1e-6, 1e-3))", # Added Weight Decay
        "++trainer.callbacks.2.patience=choice(3,5,7,9)",  # Early stopping patience
    ]
    
    # Add sweep parameters to command
    command.extend(sweep_params)
    
    # Print the command
    print("Running Optuna sweep with command:")
    print(" ".join(command))
    print(f"\nUsing database: {storage_uri}")
    print(f"Study name: {study_name}")
    print(f"This will allow knowledge to persist between multiple runs\n")
    
    # Run the command
    try:
        # Set environment variables for the subprocess
        env = os.environ.copy()
        env['MPLBACKEND'] = 'Agg'
        
        subprocess.run(command, check=True, env=env)
    except subprocess.CalledProcessError as e:
        print(f"Sweep failed with error: {e}")
        
        if "no longer compatible with the table schema" in str(e):
            print("\n" + "="*80)
            print("OPTUNA VERSION COMPATIBILITY ERROR DETECTED")
            print("="*80)
            print("The database schema is incompatible with your current Optuna version.")
            print("\nOptions to fix this:")
            print("1. Use a new database file: python run_sweep.py --db new_optuna.db")
            print("2. Force creation of a new database: python run_sweep.py --force-new")
            print("3. Downgrade Optuna: pip install optuna==2.10.1")
            print("="*80 + "\n")
        
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run hyperparameter optimization sweep")
    parser.add_argument("--trials", type=int, default=77, help="Number of trials to run")
    parser.add_argument("--epochs", type=int, default=17, help="Maximum epochs per trial")
    parser.add_argument("--db", type=str, default="optuna_fresh.db", help="Database filename for storing results")
    parser.add_argument("--study", type=str, default="climate_optimization", help="Optuna study name")
    parser.add_argument("--force-new", action="store_true", help="Force creation of a new database, removing existing one")
    args = parser.parse_args()
    
    run_sweep(n_trials=args.trials, max_epochs=args.epochs, db_name=args.db, study_name=args.study, force_new_db=args.force_new) 