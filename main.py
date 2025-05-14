import os
from datetime import datetime

# Force matplotlib to use non-interactive backend to prevent Tkinter errors
import os
os.environ['MPLBACKEND'] = 'Agg'

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
#UPDATED BY GEMINI 2.5-pro-experimental
class ClimateDataset(Dataset):
    def __init__(self, inputs_norm_dask, outputs_dask, sequence_length, output_is_normalized=True): # Added sequence_length
        self.sequence_length = sequence_length # Store sequence_length
        self.output_is_normalized = output_is_normalized
        # Store dataset size (total number of individual time steps)
        self.total_timesteps = inputs_norm_dask.shape[0]

        log.info(
            f"Creating dataset: {self.total_timesteps} total time steps, sequence_length: {self.sequence_length}, "
            f"input shape (per step): {inputs_norm_dask.shape[1:]}, normalized output: {output_is_normalized}"
        )

        # Precompute all tensors in one go
        inputs_np = inputs_norm_dask.compute()
        outputs_np = outputs_dask.compute()

        self.input_tensors = torch.from_numpy(inputs_np).float()
        self.output_tensors = torch.from_numpy(outputs_np).float()

        if torch.isnan(self.input_tensors).any() or torch.isnan(self.output_tensors).any():
            raise ValueError("NaN values detected in dataset tensors")

    def __len__(self):
        # Number of possible start points for a sequence
        return self.total_timesteps - self.sequence_length + 1

    def __getitem__(self, idx):
        # idx is the starting point of the sequence
        start_idx = idx
        end_idx = idx + self.sequence_length

        # Get sequences of inputs and outputs
        input_seq = self.input_tensors[start_idx:end_idx]     # (seq_len, C_in, H, W)
        output_seq = self.output_tensors[start_idx:end_idx]   # (seq_len, C_out, H, W)
        
        return input_seq, output_seq


def _load_process_ssp_data(ds, ssp, input_variables, output_variables, member_id, spatial_template):
    """
    Loads and processes input and output variables for a single SSP using Dask.

    Args:
        ds (xr.Dataset): The opened xarray dataset.
        ssp (str): The SSP identifier (e.g., 'ssp126').
        input_variables (list): List of input variable names.
        output_variables (list): List of output variable names.
        member_id (int): The member ID to select.
        spatial_template (xr.DataArray): A template DataArray with ('y', 'x') dimensions
                                          for broadcasting global variables.

    Returns:
        tuple: (input_dask_array, output_dask_array)
               - input_dask_array: Stacked dask array of inputs (time, channels, y, x).
               - output_dask_array: Stacked dask array of outputs (time, channels, y, x).
    """
    ssp_input_dasks = []
    for var in input_variables:
        da_var = ds[var].sel(ssp=ssp)
        # Rename spatial dims if needed
        if "latitude" in da_var.dims:
            da_var = da_var.rename({"latitude": "y", "longitude": "x"})
        # Select member if applicable
        if "member_id" in da_var.dims:
            da_var = da_var.sel(member_id=member_id)

        # Process based on dimensions
        if set(da_var.dims) == {"time"}:  # Global variable, broadcast to spatial dims:
            # Broadcast like template, then transpose to ensure ('time', 'y', 'x')
            da_var_expanded = da_var.broadcast_like(spatial_template).transpose("time", "y", "x")
            ssp_input_dasks.append(da_var_expanded.data)
        elif set(da_var.dims) == {"time", "y", "x"}:  # Spatially resolved
            ssp_input_dasks.append(da_var.data)
        else:
            raise ValueError(f"Unexpected dimensions for variable {var} in SSP {ssp}: {da_var.dims}")

    # Stack inputs along channel dimension -> dask array (time, channels, y, x)
    stacked_input_dask = da.stack(ssp_input_dasks, axis=1)

    # Prepare output dask arrays for each output variable
    output_dasks = []
    for var in output_variables:
        da_output = ds[var].sel(ssp=ssp, member_id=member_id)
        # Ensure output also uses y, x if necessary
        if "latitude" in da_output.dims:
            da_output = da_output.rename({"latitude": "y", "longitude": "x"})

        # Add time, y, x dimensions as a dask array
        output_dasks.append(da_output.data)

    # Stack outputs along channel dimension -> dask array (time, channels, y, x)
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
        sequence_length: int, 
        test_months: int = 360,
        batch_size: int = 32,
        eval_batch_size: int = None,
        num_workers: int = 0,
        seed: int = 42,
        output_height: int | None = None,
        output_width: int | None = None,
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

        # Use context manager for opening dataset
        with xr.open_zarr(self.hparams.path, consolidated=True, chunks={"time": 24}) as ds:
            # Create a spatial template ONCE using a variable guaranteed to have y, x
            # Extract the template DataArray before renaming for coordinate access
            spatial_template_da = ds["rsdt"].isel(time=0, ssp=0, drop=True)  # drop time/ssp dims

            # --- Prepare Training and Validation Data ---
            train_inputs_dask_list, train_outputs_dask_list = [], []
            val_input_dask, val_output_dask = None, None
            val_ssp = "ssp370"
            val_months = 120 # This val_months is for splitting ssp370, not sequence_length

            # Process all SSPs
            log.info(f"Loading data from SSPs: {self.hparams.train_ssps}")
            for ssp in self.hparams.train_ssps:
                # Load the data for this SSP
                ssp_input_dask, ssp_output_dask = _load_process_ssp_data(
                    ds,
                    ssp,
                    self.hparams.input_vars,
                    self.hparams.output_vars,
                    self.hparams.target_member_id,
                    spatial_template_da,
                )

                if ssp == val_ssp:
                    # Special handling for SSP 370: split into training and validation
                    # Last 120 months go to validation
                    val_input_dask = ssp_input_dask[-val_months:]
                    val_output_dask = ssp_output_dask[-val_months:]
                    # Early months go to training if there are any
                    if ssp_input_dask.shape[0] > val_months: # Ensure there's training data left
                        train_inputs_dask_list.append(ssp_input_dask[:-val_months])
                        train_outputs_dask_list.append(ssp_output_dask[:-val_months])
                else:
                    # All other SSPs go entirely to training
                    train_inputs_dask_list.append(ssp_input_dask)
                    train_outputs_dask_list.append(ssp_output_dask)

            # Concatenate training data only
            train_input_dask = da.concatenate(train_inputs_dask_list, axis=0)
            train_output_dask = da.concatenate(train_outputs_dask_list, axis=0)

            # Compute z-score normalization statistics using the training data
            input_mean = da.nanmean(train_input_dask, axis=(0, 2, 3), keepdims=True).compute()
            input_std = da.nanstd(train_input_dask, axis=(0, 2, 3), keepdims=True).compute()
            output_mean = da.nanmean(train_output_dask, axis=(0, 2, 3), keepdims=True).compute()
            output_std = da.nanstd(train_output_dask, axis=(0, 2, 3), keepdims=True).compute()

            self.normalizer.set_input_statistics(mean=input_mean, std=input_std)
            self.normalizer.set_output_statistics(mean=output_mean, std=output_std)

            # --- Define Normalized Training Dask Arrays ---
            train_input_norm_dask = self.normalizer.normalize(train_input_dask, data_type="input")
            train_output_norm_dask = self.normalizer.normalize(train_output_dask, data_type="output")

            # --- Define Normalized Validation Dask Arrays ---
            val_input_norm_dask = self.normalizer.normalize(val_input_dask, data_type="input")
            val_output_norm_dask = self.normalizer.normalize(val_output_dask, data_type="output")

            # --- Prepare Test Data ---
            full_test_input_dask, full_test_output_dask = _load_process_ssp_data(
                ds,
                self.hparams.test_ssp,
                self.hparams.input_vars,
                self.hparams.output_vars,
                self.hparams.target_member_id,
                spatial_template_da,
            )

            # --- Slice Test Data ---
            # Note: test_months is the total duration of the test set, not sequence_length
            test_slice = slice(-self.hparams.test_months, None)  

            sliced_test_input_dask = full_test_input_dask[test_slice]
            sliced_test_output_raw_dask = full_test_output_dask[test_slice]

            # --- Define Normalized Test Input Dask Array ---
            test_input_norm_dask = self.normalizer.normalize(sliced_test_input_dask, data_type="input")
            test_output_raw_dask = sliced_test_output_raw_dask  # Keep unnormed for evaluation

        # Create datasets, passing sequence_length
        self.train_dataset = ClimateDataset(
            train_input_norm_dask, train_output_norm_dask, 
            self.hparams.sequence_length, output_is_normalized=True
        )
        self.val_dataset = ClimateDataset(
            val_input_norm_dask, val_output_norm_dask, 
            self.hparams.sequence_length, output_is_normalized=True
        )
        self.test_dataset = ClimateDataset(
            test_input_norm_dask, test_output_raw_dask, 
            self.hparams.sequence_length, output_is_normalized=False
        )

        # Log dataset sizes in a single message
        log.info(
            f"Datasets created. Train: {len(self.train_dataset)} sequences, Val: {len(self.val_dataset)} sequences (last months of {val_ssp}), Test: {len(self.test_dataset)} sequences"
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
    def __init__(self, model: nn.Module, learning_rate: float, weight_decay: float = 0.0):
        super().__init__()
        self.model = model
        self.save_hyperparameters(ignore=["model"])
        self.criterion = nn.MSELoss()
        self.normalizer = None
        # Store evaluation outputs for time-mean calculation
        self.test_step_outputs = []
        self.validation_step_outputs = []

    def forward(self, x):
        return self.model(x)

    def on_fit_start(self) -> None:
        self.normalizer = self.trainer.datamodule.normalizer  # Access the normalizer from the datamodule
        # Store sequence_length if needed, or access from datamodule.hparams directly
        self.sequence_length = self.trainer.datamodule.hparams.sequence_length

    def training_step(self, batch, batch_idx):
        # x_seq: (B, S, C_in, H, W), y_true_norm_seq: (B, S, C_out, H, W)
        x_seq, y_true_norm_seq = batch
        
        # y_pred_norm_seq: (B, S, C_out, H, W)
        y_pred_norm_seq = self(x_seq) 
        
        batch_size, seq_len, c_out, h, w = y_pred_norm_seq.shape
        
        # Reshape for loss calculation: (B*S, C_out, H, W)
        loss = self.criterion(
            y_pred_norm_seq.reshape(batch_size * seq_len, c_out, h, w),
            y_true_norm_seq.reshape(batch_size * seq_len, c_out, h, w)
        )
        # Log with batch_size representing number of sequences
        self.log("train/loss", loss, prog_bar=True, batch_size=batch_size) 
        return loss

    def validation_step(self, batch, batch_idx):
        # x_seq: (B, S, C_in, H, W), y_true_norm_seq: (B, S, C_out, H, W)
        x_seq, y_true_norm_seq = batch
        
        # y_pred_norm_seq: (B, S, C_out, H, W)
        y_pred_norm_seq = self(x_seq)

        batch_size, seq_len, c_out, h, w = y_pred_norm_seq.shape

        # Reshape for loss calculation
        loss = self.criterion(
            y_pred_norm_seq.reshape(batch_size * seq_len, c_out, h, w),
            y_true_norm_seq.reshape(batch_size * seq_len, c_out, h, w),
        )
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True)

        # Denormalize and reshape for storage
        # Flatten for normalizer: (B*S, C_out, H, W)
        y_pred_norm_flat = y_pred_norm_seq.reshape(batch_size * seq_len, c_out, h, w).cpu().numpy()
        y_true_norm_flat = y_true_norm_seq.reshape(batch_size * seq_len, c_out, h, w).cpu().numpy()

        y_pred_unnorm_flat = self.normalizer.inverse_transform_output(y_pred_norm_flat)
        y_true_unnorm_flat = self.normalizer.inverse_transform_output(y_true_norm_flat)
        
        # Append the flat (non-sequential) denormalized arrays for epoch end calculation
        self.validation_step_outputs.append((y_pred_unnorm_flat, y_true_unnorm_flat))

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

        # Concatenate all predictions and ground truths
        # Each element in validation_step_outputs is ( (B*S,C,H,W), (B*S,C,H,W) )
        all_preds_np = np.concatenate([pred for pred, _ in self.validation_step_outputs], axis=0)
        all_trues_np = np.concatenate([true for _, true in self.validation_step_outputs], axis=0)
        
        # Ensure that the total number of samples matches what _evaluate_predictions expects.
        # This should now be (Total_Individual_Timesteps, C, H, W)

        # Use the helper method to evaluate predictions
        self._evaluate_predictions(all_preds_np, all_trues_np, is_test=False)

        self.validation_step_outputs.clear()  # Clear the outputs list for next epoch

    def test_step(self, batch, batch_idx):
        # x_seq: (B, S, C_in, H, W), y_true_denorm_seq: (B, S, C_out, H, W)
        x_seq, y_true_denorm_seq = batch
        
        # y_pred_norm_seq: (B, S, C_out, H, W)
        y_pred_norm_seq = self(x_seq) 

        batch_size, seq_len, c_out, h, w = y_pred_norm_seq.shape
        
        # Denormalize the predictions
        # Flatten for normalizer: (B*S, C_out, H, W)
        y_pred_norm_flat = y_pred_norm_seq.reshape(batch_size * seq_len, c_out, h, w).cpu().numpy()
        y_pred_denorm_flat = self.normalizer.inverse_transform_output(y_pred_norm_flat)
        
        # Reshape y_true_denorm_seq for consistency if needed, or it might already be (B*S, C,H,W)
        # depending on how test_dataset yields it (output_is_normalized=False means raw outputs)
        y_true_denorm_flat = y_true_denorm_seq.reshape(batch_size * seq_len, c_out, h, w).cpu().numpy()
        
        # Append the flat (non-sequential) denormalized arrays
        self.test_step_outputs.append((y_pred_denorm_flat, y_true_denorm_flat))

    def on_test_epoch_end(self):
        if not self.test_step_outputs:
            log.info("No test outputs to process.")
            return

        # Concatenate all predictions and ground truths from each test step/batch into one array
        # Each element in test_step_outputs is ( (B*S,C,H,W), (B*S,C,H,W) )
        all_preds_denorm = np.concatenate([pred for pred, true in self.test_step_outputs], axis=0)
        all_trues_denorm = np.concatenate([true for pred, true in self.test_step_outputs], axis=0)
        
        # This should now be (Total_Individual_Timesteps, C, H, W)

        # Use the helper method to evaluate predictions
        self._evaluate_predictions(all_preds_denorm, all_trues_denorm, is_test=True)

        # Save predictions for Kaggle submission.
        log.info("Saving Kaggle submission...")
        # _save_kaggle_submission expects (time, channels, y, x)
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
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',      # Reduce LR when the monitored quantity has stopped decreasing
            factor=0.5,      # Factor by which the learning rate will be reduced. new_lr = lr * factor
            patience=3,      # Number of epochs with no improvement after which learning rate will be reduced
            verbose=True     # If True, prints a message to stdout for each update
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss", # Metric to monitor
                "interval": "epoch",
                "frequency": 1,
            },
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
        weight_decay=cfg.training.weight_decay
    )

    # Create lightning trainer
    trainer_config = get_trainer_config(cfg, model=model)
    trainer = pl.Trainer(**trainer_config)

    if cfg.ckpt_path and isinstance(cfg.ckpt_path, str):
        cfg.ckpt_path = to_absolute_path(cfg.ckpt_path)

    # Train model
    trainer.fit(lightning_module, datamodule=datamodule, ckpt_path=cfg.ckpt_path)
    log.info("Training finished.")

    # Get validation metrics
    results = trainer.callback_metrics
    
    # Test model
    # IMPORTANT: Please note that the test metrics will be bad because the test targets have been corrupted on the public Kaggle dataset.
    # The purpose of testing below is to generate the Kaggle submission file based on your model's predictions.
    trainer_config["devices"] = 1  # Make sure you test on 1 GPU only to avoid synchronization issues with DDP
    eval_trainer = pl.Trainer(**trainer_config)
    eval_trainer.test(lightning_module, datamodule=datamodule, ckpt_path=cfg.ckpt_path)

    if cfg.use_wandb and isinstance(trainer_config.get("logger"), WandbLogger):
        wandb.finish()  # Finish the run if using wandb
    
    # Return validation metric for Optuna optimization
    optimized_metric = results.get("val/loss")
    if optimized_metric is not None:
        log.info(f"Optimization metric (val/loss): {optimized_metric:.6f}")
        return float(optimized_metric)
    else:
        log.warning("val/loss metric not found in results. Returning placeholder value for Optuna.")
        return float("inf")  # If val/loss doesn't exist, return infinity to discourage this setting


if __name__ == "__main__":
    main()
