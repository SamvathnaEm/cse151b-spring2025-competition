# CSE 151B Competition Spring 2025 - Climate Emulation

This repository contains a starting point for the [CSE 151B](https://sites.google.com/view/cse151b-251b/151b-info) competition on climate emulation.
It includes a basic PyTorch Lightning training script, a simple CNN model, a data loader for the provided Zarr dataset, a configuration system using Hydra, and a logging system using Weights & Biases.
Its structure follows what we find to be useful in our own research projects, but you are free to modify it as needed.

## | Kaggle Competition Website
    
[CSE 151B Competition - Climate Emulation](https://www.kaggle.com/t/6f53c429d53099dc7cc590f9bf390b10)

## | Overview

This competition challenges participants to develop machine learning models that can accurately emulate a physics-based climate model to project future climate patterns under varying emissions scenarios. Your models will be evaluated on their ability to capture both spatial patterns and temporal variability - key requirements for actionable climate predictions.

  ### Description
  Climate models are essential tools for understanding Earth's future climate, but they are computationally expensive to run. Machine learning approaches offer a promising alternative that 
  could dramatically reduce computational costs while maintaining prediction accuracy. In this competition, you'll work with data from CMIP6 climate model simulations under different Shared 
  Socioeconomic Pathway (SSP) scenarios.

  The training data consists of monthly climate variables (precipitation and temperature) from multiple SSP scenarios. 
  Your task is to develop models that can predict these variables given various input variables including greenhouse gas concentrations and aerosols under new SSP scenarios. 
  Success in this competition requires models that can:

  1. Capture complex spatial patterns of climate variables across the globe
  2. Accurately represent both mean climate states and temporal variability
  3. Learn the physical relationships between input climate forcings and climate responses

  This challenge simulates a real-world problem in climate science: using data from existing scenarios to predict climate under new scenarios, thereby reducing the need for expensive 
  simulation runs.

  ### Evaluation
  Submissions are evaluated using a combination of area-weighted metrics that account for the different sizes of grid cells at different latitudes (cells near the equator cover more area than those near the poles):

  1. **Monthly Area-Weighted RMSE**: Measures the accuracy of your model's monthly predictions. Calculated as: √(weighted_mean((prediction - actual)²))

  2. **Decadal Mean Area-Weighted RMSE**: Specifically evaluates how well your model captures the spatial patterns in the time-averaged climate. This metric is particularly important for 
  capturing long-term climate change signals. This metric is calculated as: √(weighted_mean((time_mean(predictions) - time_mean(actuals))²)), where time_mean is the mean over a 10-year period

  3. **Decadal Standard Deviation Area-Weighted MAE**: Assesses how well your model represents the temporal variability at each location. This metric ensures models don't just predict the mean 
  state correctly but also capture climate variability. This metric is calculated as: weighted_mean(abs(time_std(predictions) - time_std(actuals))), where time_std is the standard deviation over a 10-year period.

  The final score is a weighted combination of these metrics across precipitation and temperature variables. Note that an important consideration for climate emulators is their computational efficiency (i.e., how quickly they can make predictions at inference time). We encourage you to consider this when designing your models, although this competition does not explicitly evaluate that.

  ## | Dataset Details

  For computational efficiency, the data have been coarsened to a (48, 72) lat-lon grid. 

  Input Variables (also called Forcings):
  - ``CO2`` - Carbon dioxide concentrations
  - ``SO2`` - Sulfur dioxide emissions
  - ``CH4`` - Methane concentrations
  - ``BC`` - Black carbon emissions
  - ``rsdt`` - Incoming solar radiation at top of atmosphere (can be useful to inject knowledge of the season/time of year)

  Output Variables to Predict:
  - ``tas`` - Surface air temperature (in Kelvin)
  - ``pr`` - Precipitation rate (in mm/day)
   
   **Note:** You are free to use any or all of the input variables to make your predictions. 
   Similarly, it is up to you how to predict the output variables (e.g. predict both tas and pr together, or predict them separately).

  ### Data Structure

  The dataset is stored in Zarr format, which efficiently handles large multidimensional arrays. The data includes:

  - Spatial dimensions: Global grid with latitude (y) and longitude (x) coordinates
  - Time dimension: Monthly climate data
  - Member ID dimension: Each scenario was simulated three times (i.e. a 3-member ensemble). This is done to account for the internal variability of the climate system (i.e. the fact that the climate system can evolve differently even under the same external forcings). Thus, given any snapshot of monthly forcings, any of the corresponding monthly climate responses from any of the three ensemble members is a valid target.
  - Multiple scenarios: Data from different Shared Socioeconomic Pathways (SSPs)
    - Training: SSP126 (low emissions), SSP370 (high emissions), SSP585 (very high emissions)
    - Validation: Last 10 years of SSP370
    - Testing: SSP245 (intermediate emissions)
  
  Note: By default, the provided code uses a single ensemble as target for training and validation. It is up to you to decide if and how to use the other ensemble members.

  ### Data Preparation

  In the ``main.py`` script, the data is preprocessed with:
  - Normalization of input and output variables (Z-score normalization)
  - Handling of global input variables by broadcasting them to match spatial dimensions
   
  You can modify this preprocessing pipeline to suit your model architecture and requirements. 
  For example, precipitation data follows a skewed distribution, which may benefit from alternative normalization methods.
 
### Data Visualization

See the [notebooks/data-exploration-basic.ipynb](notebooks/data-exploration-basic.ipynb) notebook for a basic data exploration and visualization of the dataset.


## | Getting Started

1. Create a fresh virtual environment (we recommend python >= 3.10) and install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Download the zarr data files from the competition page and place them in a directory of your choice (you'll later need to specify it with the ``data.path`` argument).

### Configuration

This project uses Hydra for configuration management. The main configuration files are in the `configs/` directory:

- `configs/main_config.yaml`: Main configuration file that includes other configuration files
- `configs/data/default.yaml`: Dataset and data-loading related settings (e.g. data path and batch size)
- `configs/model/simple_cnn.yaml`: Model architecture settings (e.g. architecture type, number of layers)
- `configs/training/default.yaml`: Training parameters (e.g. learning rate)
- `configs/trainer/default.yaml`: PyTorch Lightning Trainer settings (e.g. number of GPUs, precision)

### Running the Model

This codebase uses PyTorch Lightning for training. It is meant to be a starting point for your own model development.
You may use any (or none) of the code provided here and are free to modify it as needed.

To train the model with default settings:

```bash
python main.py data.path=/path/to/your/data.zarr
```

#### Logging

It is recommended to use Weights & Biases for logging.
To enable logging, set `use_wandb=true` and specify your W&B (team) username with `wandb_entity=<your-wandb-username>`.
You will need to create a project `cse-151b-competition` on Weights & Biases. 
When logging is enabled, the training script will automatically log metrics, and hyperparameters to your W&B project.
This will allow you to monitor your training runs and compare different experiments more conveniently from the W&B dashboard.

#### Common Configuration Options

Override configuration options from the command line:

```bash
# Use Weights & Biases for logging (recommended). Be sure to first create a project ``cse-151b-competition`` on wandb.
python main.py data.path=/path/to/your/data.zarr use_wandb=true wandb_entity=<your-wandb-username>

# Change batch size and learning rate and use different batch size for validation
python main.py data.path=/path/to/your/data.zarr data.batch_size=64 data.eval_batch_size=32 training.lr=1e-3

# Change the number of epochs
python main.py data.path=/path/to/your/data.zarr trainer.max_epochs=200

# Train on 4 GPUs with DistributedDataParallel (DDP) mode
python main.py data.path=/path/to/your/data.zarr trainer.strategy="ddp_find_unused_parameters_false" trainer.devices=4 

# Resume training from (or evaluate) a specific checkpoint
python main.py data.path=/path/to/your/data.zarr ckpt_path=/path/to/your/checkpoint.ckpt
```
