#!/usr/bin/env python3
"""
fix_matplotlib.py - Sets up matplotlib to use non-interactive backend

This script creates/modifies the matplotlib configuration file to always use
the 'Agg' backend, which prevents Tkinter errors in headless environments.
"""

import os
import matplotlib
from pathlib import Path

def setup_matplotlib_config():
    """Set up matplotlib to use 'Agg' backend by default."""
    
    # Find the configuration directory
    config_dir = Path(matplotlib.get_configdir())
    
    # Ensure the directory exists
    config_dir.mkdir(exist_ok=True, parents=True)
    
    # Path to matplotlib's configuration file
    matplotlibrc_path = config_dir / "matplotlibrc"
    
    # Check if the file exists and read its contents
    lines = []
    if matplotlibrc_path.exists():
        with open(matplotlibrc_path, 'r') as f:
            lines = f.readlines()
        
        # Filter out any existing backend line
        lines = [line for line in lines if not line.strip().startswith('backend:')]
    
    # Add the backend line
    lines.append('backend: Agg\n')
    
    # Write the modified or new config file
    with open(matplotlibrc_path, 'w') as f:
        f.writelines(lines)
    
    print(f"Set matplotlib backend to 'Agg' in {matplotlibrc_path}")
    print("This should prevent Tkinter errors during parallel execution.")

if __name__ == "__main__":
    setup_matplotlib_config() 