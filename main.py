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
from torch.utils.data import DataLoader, Dataset, ConcatDataset


try:
    import wandb
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


log = get_logger(__name__)


# --- Data Handling ---

class ClimateSequenceDataset(Dataset):
    def __init__(self, inputs_dask, outputs_dask, sequence_length, output_start_idx, output_end_idx, output_is_normalized=True):
        """
        Dataset to provide sequences of climate data.
        
        Args:
            inputs_dask (dask.array): Input data of shape (time, C_in, H, W)
            outputs_dask (dask.array): Output data of shape (time, C_out, H, W)
            sequence_length (int): Length of the input sequence
            output_start_idx (int): Starting index for output time steps
            output_end_idx (int): Ending index for output time steps
            output_is_normalized (bool): Whether outputs are normalized
        """
        self.sequence_length = sequence_length
        self.inputs_dask = inputs_dask
        self.outputs_dask = outputs_dask
        self.output_is_normalized = output_is_normalized
        self.output_start_idx = output_start_idx
        self.output_end_idx = output_end_idx
        self.size = output_end_idx - output_start_idx + 1

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        t = self.output_start_idx + idx
        input_seq = self.inputs_dask[t - self.sequence_length:t].compute()
        output = self.outputs_dask[t].compute()
        input_seq_tensor = torch.from_numpy(input_seq).float()
        output_tensor = torch.from_numpy(output).float()
        return input_seq_tensor, output_tensor


def _load_process_ssp_data(ds, ssp, input_variables, output_variables, member_id, spatial_template):
    ssp_input_dasks = []
    for var in input_variables:
        da_var = ds[var].sel(ssp=ssp)
        if "latitude" in da_var.dims:
            da_var = da_var.rename({"latitude": "y", "longitude": "x"})
        if "member_id" in da_var.dims:
            da_var = da_var.sel(member_id=member_id)
        if set(da_var.dims) == {"time"}:
            da_var_expanded = da_var.broadcast_like(spatial_template).transpose("time", "y", "x")
            ssp_input_dasks.append(da_var_expanded.data)
        elif set(da_var.dims) == {"time", "y", "x"}:
            ssp_input_dasks.append(da_var.data)
        else:
            raise ValueError(f"Unexpected dimensions for variable {var} in SSP {ssp}: {da_var.dims}")
    stacked_input_dask = da.stack(ssp_input_dasks, axis=1)
    output_dasks = []
    for var in output_variables:
        da_output = ds[var].sel(ssp=ssp, member_id=member_id)
        if "latitude" in da_output.dims:
            da_output = da_output.rename({"latitude": "y", "longitude": "x"})
        output_dasks.append(da_output.data)
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
        sequence_length: int = 12,  # Added sequence_length parameter
    ):
        super().__init__()
        self.save_hyperparameters()
        self.hparams.path = to_absolute_path(path)
        self.normalizer = Normalizer()
        self.sequence_length = sequence_length
        if eval_batch_size is None:
            self.hparams.eval_batch_size = batch_size
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
            all_ssps = self.hparams.train_ssps + [self.hparams.test_ssp]
            full_inputs_dask_dict = {}
            full_outputs_dask_dict = {}
            for ssp in all_ssps:
                ssp_input_dask, ssp_output_dask = _load_process_ssp_data(
                    ds,
                    ssp,
                    self.hparams.input_vars,
                    self.hparams.output_vars,
                    self.hparams.target_member_id,
                    spatial_template_da,
                )
                full_inputs_dask_dict[ssp] = ssp_input_dask
                full_outputs_dask_dict[ssp] = ssp_output_dask

            # Prepare training data
            train_inputs_dask_list = []
            train_outputs_dask_list = []
            val_ssp = "ssp370"
            val_months = 120
            for ssp in self.hparams.train_ssps:
                if ssp == val_ssp:
                    train_inputs_dask_list.append(full_inputs_dask_dict[ssp][:-val_months])
                    train_outputs_dask_list.append(full_outputs_dask_dict[ssp][:-val_months])
                else:
                    train_inputs_dask_list.append(full_inputs_dask_dict[ssp])
                    train_outputs_dask_list.append(full_outputs_dask_dict[ssp])

            train_input_dask = da.concatenate(train_inputs_dask_list, axis=0)
            train_output_dask = da.concatenate(train_outputs_dask_list, axis=0)

            # Compute normalization statistics
            input_mean = da.nanmean(train_input_dask, axis=(0, 2, 3), keepdims=True).compute()
            input_std = da.nanstd(train_input_dask, axis=(0, 2, 3), keepdims=True).compute()
            output_mean = da.nanmean(train_output_dask, axis=(0, 2, 3), keepdims=True).compute()
            output_std = da.nanstd(train_output_dask, axis=(0, 2, 3), keepdims=True).compute()
            self.normalizer.set_input_statistics(mean=input_mean, std=input_std)
            self.normalizer.set_output_statistics(mean=output_mean, std=output_std)

            # Normalize full data
            normalized_inputs_dask_dict = {}
            normalized_outputs_dask_dict = {}
            for ssp in all_ssps:
                normalized_inputs_dask_dict[ssp] = self.normalizer.normalize(full_inputs_dask_dict[ssp], data_type="input")
                normalized_outputs_dask_dict[ssp] = self.normalizer.normalize(full_outputs_dask_dict[ssp], data_type="output")

            # Training datasets
            train_datasets = []
            for ssp in self.hparams.train_ssps:
                inputs_dask = normalized_inputs_dask_dict[ssp]
                outputs_dask = normalized_outputs_dask_dict[ssp]
                T = inputs_dask.shape[0]
                if ssp == val_ssp:
                    output_end_idx = T - val_months - 1
                else:
                    output_end_idx = T - 1
                output_start_idx = self.sequence_length
                if output_end_idx >= output_start_idx:
                    dataset = ClimateSequenceDataset(
                        inputs_dask, outputs_dask, self.sequence_length, output_start_idx, output_end_idx
                    )
                    train_datasets.append(dataset)
            self.train_dataset = ConcatDataset(train_datasets)

            # Validation dataset
            ssp_val = val_ssp
            inputs_dask_val = normalized_inputs_dask_dict[ssp_val]
            outputs_dask_val = normalized_outputs_dask_dict[ssp_val]
            T_val = inputs_dask_val.shape[0]
            output_start_idx_val = T_val - val_months
            output_end_idx_val = T_val - 1
            self.val_dataset = ClimateSequenceDataset(
                inputs_dask_val, outputs_dask_val, self.sequence_length, output_start_idx_val, output_end_idx_val
            )

            # Test dataset
            ssp_test = self.hparams.test_ssp
            inputs_dask_test = normalized_inputs_dask_dict[ssp_test]
            outputs_dask_test = full_outputs_dask_dict[ssp_test]  # Raw outputs
            T_test = inputs_dask_test.shape[0]
            output_start_idx_test = T_test - self.hparams.test_months
            output_end_idx_test = T_test - 1
            self.test_dataset = ClimateSequenceDataset(
                inputs_dask_test, outputs_dask_test, self.sequence_length, output_start_idx_test, output_end_idx_test,
                output_is_normalized=False
            )

        log.info(
            f"Datasets created. Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}, Test: {len(self.test_dataset)}"
        )

    def _get_dataloader_kwargs(self, is_train=False):
        return {
            "batch_size": self.hparams.batch_size if is_train else self.hparams.eval_batch_size,
            "shuffle": is_train,
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
        if self._lat_weights_da is None:
            with xr.open_zarr(self.hparams.path, consolidated=True) as ds:
                template = ds["rsdt"].isel(time=0, ssp=0)
                y_coords = template.y.values
                weights = get_lat_weights(y_coords)
                self._lat_weights_da = xr.DataArray(weights, dims=["y"], coords={"y": y_coords}, name="area_weights")
        return self._lat_weights_da

    def get_coords(self):
        if self.lat_coords is None or self.lon_coords is None:
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
        self.test_step_outputs = []
        self.validation_step_outputs = []

    def forward(self, x):
        return self.model(x)

    def on_fit_start(self):
        self.normalizer = self.trainer.datamodule.normalizer

    def training_step(self, batch, batch_idx):
        x, y_true_norm = batch
        y_pred_norm = self(x)
        loss = self.criterion(y_pred_norm, y_true_norm)
        self.log("train/loss", loss, prog_bar=True, batch_size=x.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y_true_norm = batch
        y_pred_norm = self(x)
        loss = self.criterion(y_pred_norm, y_true_norm)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=x.size(0), sync_dist=True)
        y_pred_denorm = self.normalizer.inverse_transform_output(y_pred_norm.cpu().numpy())
        y_true_denorm = self.normalizer.inverse_transform_output(y_true_norm.cpu().numpy())
        self.validation_step_outputs.append((y_pred_denorm, y_true_denorm))
        return loss

    def _evaluate_predictions(self, predictions, targets, is_test=False):
        phase = "test" if is_test else "val"
        log_kwargs = {"prog_bar": not is_test, "sync_dist": not is_test}
        n_timesteps = predictions.shape[0]
        area_weights = self.trainer.datamodule.get_lat_weights()
        lat_coords, lon_coords = self.trainer.datamodule.get_coords()
        time_coords = np.arange(n_timesteps)
        output_vars = self.trainer.datamodule.hparams.output_vars

        for i, var_name in enumerate(output_vars):
            preds_var = predictions[:, i, :, :]
            trues_var = targets[:, i, :, :]
            var_unit = "mm/day" if var_name == "pr" else "K" if var_name == "tas" else "unknown"
            preds_xr = create_climate_data_array(preds_var, time_coords, lat_coords, lon_coords, var_name, var_unit)
            trues_xr = create_climate_data_array(trues_var, time_coords, lat_coords, lon_coords, var_name, var_unit)
            diff_squared = (preds_xr - trues_xr) ** 2
            overall_rmse = calculate_weighted_metric(diff_squared, area_weights, ("time", "y", "x"), "rmse")
            self.log(f"{phase}/{var_name}/avg/monthly_rmse", float(overall_rmse), **log_kwargs)
            pred_time_mean = preds_xr.mean(dim="time")
            true_time_mean = trues_xr.mean(dim="time")
            mean_diff_squared = (pred_time_mean - true_time_mean) ** 2
            time_mean_rmse = calculate_weighted_metric(mean_diff_squared, area_weights, ("y", "x"), "rmse")
            self.log(f"{phase}/{var_name}/time_mean_rmse", float(time_mean_rmse), **log_kwargs)
            pred_time_std = preds_xr.std(dim="time")
            true_time_std = trues_xr.std(dim="time")
            std_abs_diff = np.abs(pred_time_std - true_time_std)
            time_std_mae = calculate_weighted_metric(std_abs_diff, area_weights, ("y", "x"), "mae")
            self.log(f"{phase}/{var_name}/time_stddev_mae", float(time_std_mae), **log_kwargs)
            if is_test and isinstance(self.logger, WandbLogger):
                fig = create_comparison_plots(
                    true_time_mean, pred_time_mean, title_prefix=f"{var_name} Mean",
                    metric_value=time_mean_rmse, metric_name="Weighted RMSE"
                )
                self.logger.experiment.log({f"img/{var_name}/time_mean": wandb.Image(fig)})
                plt.close(fig)
                fig = create_comparison_plots(
                    true_time_std, pred_time_std, title_prefix=f"{var_name} Stddev",
                    metric_value=time_std_mae, metric_name="Weighted MAE", cmap="plasma"
                )
                self.logger.experiment.log({f"img/{var_name}/time_Stddev": wandb.Image(fig)})
                plt.close(fig)
                if n_timesteps > 3:
                    timesteps = np.random.choice(n_timesteps, 3, replace=False)
                    for t in timesteps:
                        true_t = trues_xr.isel(time=t)
                        pred_t = preds_xr.isel(time=t)
                        fig = create_comparison_plots(true_t, pred_t, title_prefix=f"{var_name} Timestep {t}")
                        self.logger.experiment.log({f"img/{var_name}/month_idx_{t}": wandb.Image(fig)})
                        plt.close(fig)

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs:
            return
        all_preds_np = np.concatenate([pred for pred, _ in self.validation_step_outputs], axis=0)
        all_trues_np = np.concatenate([true for _, true in self.validation_step_outputs], axis=0)
        self._evaluate_predictions(all_preds_np, all_trues_np, is_test=False)
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        x, y_true_denorm = batch
        y_pred_norm = self(x)
        y_pred_denorm = self.normalizer.inverse_transform_output(y_pred_norm.cpu().numpy())
        y_true_denorm_np = y_true_denorm.cpu().numpy()
        self.test_step_outputs.append((y_pred_denorm, y_true_denorm_np))

    def on_test_epoch_end(self):
        all_preds_denorm = np.concatenate([pred for pred, _ in self.test_step_outputs], axis=0)
        all_trues_denorm = np.concatenate([true for _, true in self.test_step_outputs], axis=0)
        self._evaluate_predictions(all_preds_denorm, all_trues_denorm, is_test=True)
        log.info("Saving Kaggle submission...")
        self._save_kaggle_submission(all_preds_denorm)
        self.test_step_outputs.clear()

    def _save_kaggle_submission(self, predictions, suffix=""):
        lat_coords, lon_coords = self.trainer.datamodule.get_coords()
        output_vars = self.trainer.datamodule.hparams.output_vars
        n_times = predictions.shape[0]
        time_coords = np.arange(n_times)
        submission_df = convert_predictions_to_kaggle_format(predictions, time_coords, lat_coords, lon_coords, output_vars)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = to_absolute_path(f"submissions/kaggle_submission{suffix}_{timestamp}.csv")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        submission_df.to_csv(filepath, index=False)
        if wandb is not None and isinstance(self.logger, WandbLogger):
            pass  # Uncomment to log to wandb: self.logger.experiment.log_artifact(filepath)
        log.info(f"Kaggle submission saved to {filepath}")

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        return optimizer


@hydra.main(version_base=None, config_path="configs", config_name="main_config.yaml")
def main(cfg: DictConfig):
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    pl.seed_everything(cfg.seed, workers=True)
    datamodule = ClimateEmulationDataModule(seed=cfg.seed, **cfg.data)
    model = get_model(cfg)
    # Pass both learning_rate and weight_decay from cfg.training
    lightning_module = ClimateEmulationModule(
        model,
        learning_rate=cfg.training.lr,
        weight_decay=cfg.training.weight_decay
    )
    trainer_config = get_trainer_config(cfg, model=model)
    trainer = pl.Trainer(**trainer_config)
    if cfg.ckpt_path and isinstance(cfg.ckpt_path, str):
        cfg.ckpt_path = to_absolute_path(cfg.ckpt_path)
    trainer.fit(lightning_module, datamodule=datamodule, ckpt_path=cfg.ckpt_path)
    log.info("Training finished.")
    trainer_config["devices"] = 1
    eval_trainer = pl.Trainer(**trainer_config)
    eval_trainer.test(lightning_module, datamodule=datamodule, ckpt_path=cfg.ckpt_path)
    if cfg.use_wandb and isinstance(trainer_config.get("logger"), WandbLogger):
        wandb.finish()


if __name__ == "__main__":
    main()