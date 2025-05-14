import sqlite3
import pandas as pd

# Connect to the database
conn = sqlite3.connect('optuna_fresh.db')

# Get trial information
trials = pd.read_sql_query("SELECT * FROM trials ORDER BY value", conn)
print(f"Total trials: {len(trials)}")
print("Best 5 trials:")
print(trials.head(5)[['number', 'value', 'state']])

# Get parameter information
params = pd.read_sql_query("SELECT * FROM trial_params", conn)
print(f"\nTotal parameters: {len(params)}")
print("Sample parameters:")
print(params.head(10)[['param_name', 'param_value']])

conn.close()