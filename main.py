import os
from datetime import datetime

import dask.array as da
import hydra
import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import xarray as xr
from hydra.utils import to_absolute_path
from lightning.pytorch import LightningDataModule
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset
import pandas as pd


try:
    import wandb  # Optional, for logging to Weights & Biases
except ImportError:
    wandb = None

from src.models import get_model
from src.utils import (
    Normalizer,
    calculate_weighted_metric,
    convert_predictions_to_kaggle_format,
    create_climate_data_array,
    create_comparison_plots,
    get_lat_weights,
    get_logger,
    get_trainer_config,
)


# Setup logging
log = get_logger(__name__)


# --- Data Handling ---


# Dataset to precompute all tensors during initialization
class ClimateDataset(Dataset):
    def __init__(self, inputs_norm_dask, outputs_dask, seq_len=1, output_is_normalized=True, training = False):
        self.seq_len = seq_len
        total_time = inputs_norm_dask.shape[0]
        self.training = training
        self.size = total_time - seq_len + 1 if seq_len > 1 else total_time
        if self.size <= 0:
            raise ValueError("Sequence length exceeds available time steps.")

        log.info(
            f"Creating dataset: {self.size} samples, input shape: {inputs_norm_dask.shape}, seq_len: {seq_len}, normalized output: {output_is_normalized}"
        )

        # Precompute tensors
        inputs_np = inputs_norm_dask.compute()
        outputs_np = outputs_dask.compute()
        self.input_tensors = torch.from_numpy(inputs_np).float()  # (total_time, channels, y, x)
        self.output_tensors = torch.from_numpy(outputs_np).float()  # (total_time, output_channels, y, x)

        if torch.isnan(self.input_tensors).any() or torch.isnan(self.output_tensors).any():
            raise ValueError("NaN values detected in dataset tensors")

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if self.seq_len == 1:
            input_data = self.input_tensors[idx].unsqueeze(1)  # (channels, 1, y, x)
            output_data = self.output_tensors[idx]  # (output_channels, y, x)
        else:
            start_idx = idx
            end_idx = start_idx + self.seq_len
            input_seq = self.input_tensors[start_idx:end_idx]  # (seq_len, channels, y, x)
            input_data = input_seq.permute(1, 0, 2, 3)  # (channels, seq_len, y, x)
            output_data = self.output_tensors[end_idx - 1]  # (output_channels, y, x)
        # Grok gaussian noise
        if self.training:
            input_data += torch.randn_like(input_data) * 0.01
        return input_data, output_data


def _load_process_ssp_data(ds, ssp, input_variables, output_variables, member_id, spatial_template):
    """
    Loads and processes input and output variables for a single SSP using Dask.

    Args:
        ds (xr.Dataset): The opened xarray dataset.
        ssp (str): The SSP identifier (e.g., 'ssp126').
        input_variables (list): List of input variable names, including 'sin_month' and 'cos_month' as placeholders.
        output_variables (list): List of output variable names.
        member_id (int): The member ID to select.
        spatial_template (xr.DataArray): A template DataArray with ('y', 'x') dimensions.

    Returns:
        tuple: (input_dask_array, output_dask_array)
               - input_dask_array: Stacked dask array of inputs (time, channels, y, x).
               - output_dask_array: Stacked dask array of outputs (time, channels, y, x).
    """
    # Filter input variables to only those in the dataset
    dataset_vars = [var for var in input_variables if var not in ['sin_month', 'cos_month']]
    ssp_input_dasks = []

    # Load actual dataset variables
    for var in dataset_vars:
        da_var = ds[var].sel(ssp=ssp)
        # Rename spatial dims if needed
        if "latitude" in da_var.dims:
            da_var = da_var.rename({"latitude": "y", "longitude": "x"})
        # Select member if applicable
        if "member_id" in da_var.dims:
            da_var = da_var.sel(member_id=member_id)

        # Process based on dimensions
        if set(da_var.dims) == {"time"}:  # Global variable, broadcast to spatial dims
            da_var_expanded = da_var.broadcast_like(spatial_template).transpose("time", "y", "x")
            ssp_input_dasks.append(da_var_expanded.data)
        elif set(da_var.dims) == {"time", "y", "x"}:  # Spatially resolved
            ssp_input_dasks.append(da_var.data)
        else:
            raise ValueError(f"Unexpected dimensions for variable {var} in SSP {ssp}: {da_var.dims}")

    # Add seasonal encoding channels
    # Use time coordinates from a variable after SSP selection
    time_coords = ds[dataset_vars[0]].sel(ssp=ssp).time.values  # e.g., ds['CO2'].sel(ssp=ssp).time
    # Extract month (1–12) from cftime objects
    months = np.array([t.month for t in time_coords])
    # Cyclic encoding: sin and cos of month
    month_rad = 2 * np.pi * (months - 1) / 12  # Map months 1–12 to 0–2π
    sin_month = np.sin(month_rad)  # Shape: (time,)
    cos_month = np.cos(month_rad)  # Shape: (time,)
    
    # Create DataArray with 1D data and time dimension, then broadcast
    sin_month_da = xr.DataArray(
        sin_month,
        dims=["time"],
        coords={"time": time_coords}
    ).broadcast_like(spatial_template).transpose("time", "y", "x").data
    cos_month_da = xr.DataArray(
        cos_month,
        dims=["time"],
        coords={"time": time_coords}
    ).broadcast_like(spatial_template).transpose("time", "y", "x").data
    
    # Append seasonal channels
    ssp_input_dasks.extend([sin_month_da, cos_month_da])

    # Stack inputs along channel dimension
    stacked_input_dask = da.stack(ssp_input_dasks, axis=1)

    # Prepare output dask arrays
    output_dasks = []
    for var in output_variables:
        da_output = ds[var].sel(ssp=ssp, member_id=member_id)
        if "latitude" in da_output.dims:
            da_output = da_output.rename({"latitude": "y", "longitude": "x"})
        output_dasks.append(da_output.data)

    # Stack outputs along channel dimension
    stacked_output_dask = da.stack(output_dasks, axis=1)
    return stacked_input_dask, stacked_output_dask


class ClimateEmulationDataModule(LightningDataModule):
    def __init__(
        self,
        path: str,
        input_vars: list,
        output_vars: list,
        train_ssps: list,
        test_ssp: str,
        target_member_id: int,
        test_months: int = 360,
        batch_size: int = 32,
        eval_batch_size: int = None,
        num_workers: int = 0,
        seed: int = 42,
        seq_len: int = 1,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.hparams.path = to_absolute_path(path)
        self.normalizer = Normalizer()

        # Set evaluation batch size to training batch size if not specified
        if eval_batch_size is None:
            self.hparams.eval_batch_size = batch_size

        # Placeholders
        self.train_dataset, self.val_dataset, self.test_dataset = None, None, None
        self.lat_coords, self.lon_coords, self._lat_weights_da = None, None, None

    def prepare_data(self):
        if not os.path.exists(self.hparams.path):
            raise FileNotFoundError(f"Data path not found: {self.hparams.path}")
        log.info(f"Data found at: {self.hparams.path}")

    def setup(self, stage: str | None = None):
        log.info(f"Setting up data module for stage: {stage} from {self.hparams.path}")

        with xr.open_zarr(self.hparams.path, consolidated=True, chunks={"time": 24}) as ds:
            spatial_template_da = ds["rsdt"].isel(time=0, ssp=0, drop=True)

            train_inputs_dask_list, train_outputs_dask_list = [], []
            val_input_dask, val_output_dask = None, None
            val_ssp = "ssp370"
            val_months = 120

            log.info(f"Loading data from SSPs: {self.hparams.train_ssps}")
            for ssp in self.hparams.train_ssps:
                ssp_input_dask, ssp_output_dask = _load_process_ssp_data(
                    ds,
                    ssp,
                    self.hparams.input_vars,
                    self.hparams.output_vars,
                    self.hparams.target_member_id,
                    spatial_template_da,
                )

                if ssp == val_ssp:
                    val_input_dask = ssp_input_dask[-val_months:]
                    val_output_dask = ssp_output_dask[-val_months:]
                    train_inputs_dask_list.append(ssp_input_dask[:-val_months])
                    train_outputs_dask_list.append(ssp_output_dask[:-val_months])
                else:
                    train_inputs_dask_list.append(ssp_input_dask)
                    train_outputs_dask_list.append(ssp_output_dask)

            train_input_dask = da.concatenate(train_inputs_dask_list, axis=0)
            train_output_dask = da.concatenate(train_outputs_dask_list, axis=0)

            # Compute normalization statistics
            input_mean = da.nanmean(train_input_dask, axis=(0, 2, 3), keepdims=True).compute()
            input_std = da.nanstd(train_input_dask, axis=(0, 2, 3), keepdims=True).compute()
            # Avoid division by zero for seasonal channels (sin_month, cos_month)
            input_std = np.where(input_std == 0, 1.0, input_std)  # Set std=1 for constant channels
            output_mean = da.nanmean(train_output_dask, axis=(0, 2, 3), keepdims=True).compute()
            output_std = da.nanstd(train_output_dask, axis=(0, 2, 3), keepdims=True).compute()

            self.normalizer.set_input_statistics(mean=input_mean, std=input_std)
            self.normalizer.set_output_statistics(mean=output_mean, std=output_std)

            train_input_norm_dask = self.normalizer.normalize(train_input_dask, data_type="input")
            train_output_norm_dask = self.normalizer.normalize(train_output_dask, data_type="output")

            val_input_norm_dask = self.normalizer.normalize(val_input_dask, data_type="input")
            val_output_norm_dask = self.normalizer.normalize(val_output_dask, data_type="output")

            full_test_input_dask, full_test_output_dask = _load_process_ssp_data(
                ds,
                self.hparams.test_ssp,
                self.hparams.input_vars,
                self.hparams.output_vars,
                self.hparams.target_member_id,
                spatial_template_da,
            )

            test_slice = slice(-self.hparams.test_months, None)
            sliced_test_input_dask = full_test_input_dask[test_slice]
            sliced_test_output_raw_dask = full_test_output_dask[test_slice]

            test_input_norm_dask = self.normalizer.normalize(sliced_test_input_dask, data_type="input")
            test_output_raw_dask = sliced_test_output_raw_dask

        self.train_dataset = ClimateDataset(train_input_norm_dask, train_output_norm_dask, seq_len=self.hparams.seq_len, output_is_normalized=True)
        self.train_dataset.training = True
        self.val_dataset = ClimateDataset(val_input_norm_dask, val_output_norm_dask, seq_len=self.hparams.seq_len, output_is_normalized=True)
        self.test_dataset = ClimateDataset(test_input_norm_dask, test_output_raw_dask, seq_len=self.hparams.seq_len, output_is_normalized=False)

        log.info(
            f"Datasets created. Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)} (last months of {val_ssp}), Test: {len(self.test_dataset)}"
        )

    # Common DataLoader configuration
    def _get_dataloader_kwargs(self, is_train=False):
        """Return common DataLoader configuration as a dictionary"""
        return {
            "batch_size": self.hparams.batch_size if is_train else self.hparams.eval_batch_size,
            "shuffle": is_train,  # Only shuffle training data
            "num_workers": self.hparams.num_workers,
            "persistent_workers": self.hparams.num_workers > 0,
            "pin_memory": True,
        }

    def train_dataloader(self):
        return DataLoader(self.train_dataset, **self._get_dataloader_kwargs(is_train=True))

    def val_dataloader(self):
        return DataLoader(self.val_dataset, **self._get_dataloader_kwargs(is_train=False))

    def test_dataloader(self):
        return DataLoader(self.test_dataset, **self._get_dataloader_kwargs(is_train=False))

    def get_lat_weights(self):
        """
        Returns area weights for the latitude dimension as an xarray DataArray.
        The weights can be used with xarray's weighted method for proper spatial averaging.
        """
        if self._lat_weights_da is None:
            with xr.open_zarr(self.hparams.path, consolidated=True) as ds:
                template = ds["rsdt"].isel(time=0, ssp=0)
                y_coords = template.y.values

                # Calculate weights based on cosine of latitude
                weights = get_lat_weights(y_coords)

                # Create DataArray with proper dimensions
                self._lat_weights_da = xr.DataArray(weights, dims=["y"], coords={"y": y_coords}, name="area_weights")

        return self._lat_weights_da

    def get_coords(self):
        """
        Returns the y and x coordinates (representing latitude and longitude).

        Returns:
            tuple: (y array, x array)
        """
        if self.lat_coords is None or self.lon_coords is None:
            # Get coordinates if they haven't been stored yet
            with xr.open_zarr(self.hparams.path, consolidated=True) as ds:
                template = ds["rsdt"].isel(time=0, ssp=0, drop=True)
                self.lat_coords = template.y.values
                self.lon_coords = template.x.values

        return self.lat_coords, self.lon_coords


# --- PyTorch Lightning Module ---
class ClimateEmulationModule(pl.LightningModule):
    def __init__(self, model: nn.Module, learning_rate: float, weight_decay: float = 1e-4, min_lr: float = 1e-5):
        super().__init__()
        self.model = model
        self.save_hyperparameters(ignore=["model"])  # Saves learning_rate, weight_decay, min_lr
        self.criterion = nn.MSELoss()
        self.normalizer = None
        self.test_step_outputs = []
        self.validation_step_outputs = []


    def forward(self, x):
        return self.model(x)

    def on_fit_start(self) -> None:
        self.normalizer = self.trainer.datamodule.normalizer  # Access the normalizer from the datamodule

    def training_step(self, batch, batch_idx):
        x, y_true_norm = batch  # x: (batch_size, channels, seq_len, y, x), y_true_norm: (batch_size, output_channels, y, x)
        y_pred_seq = self(x)  # (batch_size, output_channels, seq_len, y, x)
        y_pred_last = y_pred_seq[:, :, -1, :, :]  # (batch_size, output_channels, y, x)
        loss = self.criterion(y_pred_last, y_true_norm)
        self.log("train/loss", loss, prog_bar=True, batch_size=x.size(0))
        return loss
    

    def validation_step(self, batch, batch_idx):
        x, y_true_norm = batch
        y_pred_seq = self(x)
        y_pred_last = y_pred_seq[:, :, -1, :, :]
        loss = self.criterion(y_pred_last, y_true_norm)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=x.size(0), sync_dist=True)
        y_pred_denorm = self.normalizer.inverse_transform_output(y_pred_last.cpu().numpy())
        y_true_denorm = self.normalizer.inverse_transform_output(y_true_norm.cpu().numpy())
        self.validation_step_outputs.append((y_pred_denorm, y_true_denorm))
        return loss

    def _evaluate_predictions(self, predictions, targets, is_test=False):
        """
        Helper method to evaluate predictions against targets with climate metrics.

        Args:
            predictions (np.ndarray): Prediction array with shape (time, channels, y, x)
            targets (np.ndarray): Target array with shape (time, channels, y, x)
            is_test (bool): Whether this is being called from test phase (vs validation)
        """
        phase = "test" if is_test else "val"
        log_kwargs = {"prog_bar": not is_test, "sync_dist": not is_test}

        # Get number of evaluation timesteps
        n_timesteps = predictions.shape[0]

        # Get area weights for proper spatial averaging
        area_weights = self.trainer.datamodule.get_lat_weights()

        # Get coordinates
        lat_coords, lon_coords = self.trainer.datamodule.get_coords()
        time_coords = np.arange(n_timesteps)
        output_vars = self.trainer.datamodule.hparams.output_vars

        # Process each output variable
        for i, var_name in enumerate(output_vars):
            # Extract channel data
            preds_var = predictions[:, i, :, :]
            trues_var = targets[:, i, :, :]

            var_unit = "mm/day" if var_name == "pr" else "K" if var_name == "tas" else "unknown"

            # Create xarray objects for weighted calculations
            preds_xr = create_climate_data_array(
                preds_var, time_coords, lat_coords, lon_coords, var_name=var_name, var_unit=var_unit
            )
            trues_xr = create_climate_data_array(
                trues_var, time_coords, lat_coords, lon_coords, var_name=var_name, var_unit=var_unit
            )

            # 1. Calculate weighted month-by-month RMSE over all samples
            diff_squared = (preds_xr - trues_xr) ** 2
            overall_rmse = calculate_weighted_metric(diff_squared, area_weights, ("time", "y", "x"), "rmse")
            self.log(f"{phase}/{var_name}/avg/monthly_rmse", float(overall_rmse), **log_kwargs)

            # 2. Calculate time-mean (i.e. decadal, 120 months average) and calculate area-weighted RMSE for time means
            pred_time_mean = preds_xr.mean(dim="time")
            true_time_mean = trues_xr.mean(dim="time")
            mean_diff_squared = (pred_time_mean - true_time_mean) ** 2
            time_mean_rmse = calculate_weighted_metric(mean_diff_squared, area_weights, ("y", "x"), "rmse")
            self.log(f"{phase}/{var_name}/time_mean_rmse", float(time_mean_rmse), **log_kwargs)

            # 3. Calculate time-stddev (temporal variability) and calculate area-weighted MAE for time stddevs
            pred_time_std = preds_xr.std(dim="time")
            true_time_std = trues_xr.std(dim="time")
            std_abs_diff = np.abs(pred_time_std - true_time_std)
            time_std_mae = calculate_weighted_metric(std_abs_diff, area_weights, ("y", "x"), "mae")
            self.log(f"{phase}/{var_name}/time_stddev_mae", float(time_std_mae), **log_kwargs)

            # Extra logging of sample predictions/images to wandb for test phase (feel free to use this for validation)
            if is_test:
                # Generate visualizations for test phase when using wandb
                if isinstance(self.logger, WandbLogger):
                    # Time mean visualization
                    fig = create_comparison_plots(
                        true_time_mean,
                        pred_time_mean,
                        title_prefix=f"{var_name} Mean",
                        metric_value=time_mean_rmse,
                        metric_name="Weighted RMSE",
                    )
                    self.logger.experiment.log({f"img/{var_name}/time_mean": wandb.Image(fig)})
                    plt.close(fig)

                    # Time standard deviation visualization
                    fig = create_comparison_plots(
                        true_time_std,
                        pred_time_std,
                        title_prefix=f"{var_name} Stddev",
                        metric_value=time_std_mae,
                        metric_name="Weighted MAE",
                        cmap="plasma",
                    )
                    self.logger.experiment.log({f"img/{var_name}/time_Stddev": wandb.Image(fig)})
                    plt.close(fig)

                    # Sample timesteps visualization
                    if n_timesteps > 3:
                        timesteps = np.random.choice(n_timesteps, 3, replace=False)
                        for t in timesteps:
                            true_t = trues_xr.isel(time=t)
                            pred_t = preds_xr.isel(time=t)
                            fig = create_comparison_plots(true_t, pred_t, title_prefix=f"{var_name} Timestep {t}")
                            self.logger.experiment.log({f"img/{var_name}/month_idx_{t}": wandb.Image(fig)})
                            plt.close(fig)

    def on_validation_epoch_end(self):
        # Compute time-mean and time-stddev errors using all validation months
        if not self.validation_step_outputs:
            return

        # Stack all predictions and ground truths
        all_preds_np = np.concatenate([pred for pred, _ in self.validation_step_outputs], axis=0)
        all_trues_np = np.concatenate([true for _, true in self.validation_step_outputs], axis=0)

        # Use the helper method to evaluate predictions
        self._evaluate_predictions(all_preds_np, all_trues_np, is_test=False)

        self.validation_step_outputs.clear()  # Clear the outputs list for next epoch

    def test_step(self, batch, batch_idx):
        x, y_true_denorm = batch
        y_pred_seq = self(x)
        y_pred_last = y_pred_seq[:, :, -1, :, :]
        y_pred_denorm = self.normalizer.inverse_transform_output(y_pred_last.cpu().numpy())
        y_true_denorm_np = y_true_denorm.cpu().numpy()
        self.test_step_outputs.append((y_pred_denorm, y_true_denorm_np))

    def on_test_epoch_end(self):
        # Concatenate all predictions and ground truths from each test step/batch into one array
        all_preds_denorm = np.concatenate([pred for pred, true in self.test_step_outputs], axis=0)
        all_trues_denorm = np.concatenate([true for pred, true in self.test_step_outputs], axis=0)

        # Use the helper method to evaluate predictions
        self._evaluate_predictions(all_preds_denorm, all_trues_denorm, is_test=True)

        # Save predictions for Kaggle submission. This is the file that should be uploaded to Kaggle.
        log.info("Saving Kaggle submission...")
        self._save_kaggle_submission(all_preds_denorm)

        self.test_step_outputs.clear()  # Clear the outputs list

    def _save_kaggle_submission(self, predictions, suffix=""):
        """
        Create a Kaggle submission file from the model predictions.

        Args:
            predictions (np.ndarray): Predicted values with shape (time, channels, y, x)
        """
        # Get coordinates
        lat_coords, lon_coords = self.trainer.datamodule.get_coords()
        output_vars = self.trainer.datamodule.hparams.output_vars
        n_times = predictions.shape[0]
        time_coords = np.arange(n_times)

        # Convert predictions to Kaggle format
        submission_df = convert_predictions_to_kaggle_format(
            predictions, time_coords, lat_coords, lon_coords, output_vars
        )

        # Save with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = to_absolute_path(f"submissions/kaggle_submission{suffix}_{timestamp}.csv")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)  # Ensure directory exists
        submission_df.to_csv(filepath, index=False)

        if wandb is not None and isinstance(self.logger, WandbLogger):
            pass
            # Optionally, uncomment the following line to save the submission to the wandb cloud
            # self.logger.experiment.log_artifact(filepath)  # Log to wandb if available

        log.info(f"Kaggle submission saved to {filepath}")

    def configure_optimizers(self):
        # Use AdamW optimizer with weight decay
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.get('weight_decay', 1e-4)  # Default to 1e-4 if not specified
        )
        
        # Configure ReduceLROnPlateau scheduler
        scheduler = {
            'scheduler': optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',           # Minimize validation loss
                factor=0.5,           # Reduce LR by half
                patience=5,           # Wait 5 epochs without improvement
                min_lr=self.hparams.get('min_lr', 1e-5),  # Minimum LR
                verbose=True          # Print LR changes
            ),
            'monitor': 'val/loss',    # Metric to monitor (validation loss)
            'interval': 'epoch',      # Check after each epoch
            'frequency': 1            # Check every epoch
        }
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler
        }


# --- Main Execution with Hydra ---
@hydra.main(version_base=None, config_path="configs", config_name="main_config.yaml")
def main(cfg: DictConfig):
    # Print resolved configs
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # Set random seed
    pl.seed_everything(cfg.seed, workers=True)

    # Create data module with parameters from configs
    datamodule = ClimateEmulationDataModule(seed=cfg.seed, **cfg.data)
    model = get_model(cfg)

    # Create lightning module
    lightning_module = ClimateEmulationModule(
        model,
        learning_rate=cfg.training.lr,
        weight_decay=cfg.training.get('weight_decay', 1e-4),  # Pass weight_decay
        min_lr=cfg.training.get('min_lr', 1e-5)              # Pass min_lr for scheduler
    )

    # Create lightning trainer with early stopping
    trainer_config = get_trainer_config(cfg, model=model)
    
    # Add early stopping callback
    early_stopping_callback = pl.callbacks.EarlyStopping(
        monitor='val/loss',       # Monitor validation loss
        patience=15,              # Stop after 10 epochs without improvement
        mode='min',               # Minimize the monitored metric
        verbose=True              # Print when stopping
    )
    
    # Append early stopping to callbacks
    if 'callbacks' not in trainer_config:
        trainer_config['callbacks'] = []
    trainer_config['callbacks'].append(early_stopping_callback)
    
    trainer = pl.Trainer(**trainer_config)

    if cfg.ckpt_path and isinstance(cfg.ckpt_path, str):
        cfg.ckpt_path = to_absolute_path(cfg.ckpt_path)

    # Train model
    trainer.fit(lightning_module, datamodule=datamodule, ckpt_path=cfg.ckpt_path)
    log.info("Training finished.")

    # Test model
    trainer_config["devices"] = 1  # Make sure you test on 1 GPU only
    eval_trainer = pl.Trainer(**trainer_config)
    eval_trainer.test(lightning_module, datamodule=datamodule, ckpt_path=cfg.ckpt_path)

    if cfg.use_wandb and isinstance(trainer_config.get("logger"), WandbLogger):
        wandb.finish()  # Finish the run if using wandb


if __name__ == "__main__":
    main()
