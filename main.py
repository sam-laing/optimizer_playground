from dataclasses import dataclass
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

from datasets import (
    LinearRegressionDataset,
    LogisticRegressionDataset,
    LinearRegressionSingDataset,
    make_loaders,
)
from models import LinearRegressionModel, LogisticRegressionModel
from utils import plot_training_validation_losses, make_models_and_optimizers
from optimizers import Muon

@dataclass
class Config:
    dataset_type: str = "linear_singular"
    noise_type: str = "gaussian"
    weight_init: str = "gaussian"
    data_seed: int = 22
    dim_input: int = 200
    dim_output: int = 40
    num_classes: int = 10
    n_samples: int = 10_000
    model_seed: int = 9
    snr: float = 100
    condition_number: float = 1000
    sing_dist: str = "normal"
    cov_strength: float = 0.6
    normalize_features: bool = False
    batch_size: int = 10_000
    epochs: int = 15
    max_singular_value: float = 1
    noise_level: float = 0.01  # only used for linear regression datasets
    val_size: float = 0.2
    shuffle: bool = True
    device: str = "cpu"
    scale_up: float = 1.0
    separate_bias: bool = True
    scheduler: bool = True 
    constant_proportion: float = 0.3

def loss_function(W, X, Y):
    return F.mse_loss(X @ W, Y)

def compare_optimizers(
    optimizers_dict: dict,
    dataset,
    config: Config,
):
    train_loader, val_loader = make_loaders(
        dataset,
        batch_size=config.batch_size,
        seed=config.data_seed,
        val_size=config.val_size,
        shuffle=config.shuffle,
    )

    models, optimizers = make_models_and_optimizers(
        dataset,
        optimizers_dict,
        model_seed=config.model_seed,
        device=config.device,
    )

    schedulers = {}
    if hasattr(config, "scheduler") and config.scheduler:
        from utils import ConstantThenCosineLR
        steps_per_epoch = len(train_loader)
        total_steps = config.epochs * steps_per_epoch

        for name, optimizer in optimizers.items():
            schedulers[name] = ConstantThenCosineLR(
                optimizer = optimizer,
                total_steps=total_steps,
                constant_proportion=config.constant_proportion,
            )


    losses = {name: [] for name in models.keys()}
    val_losses = {name: [] for name in models.keys()}
    for epoch in range(config.epochs):
        for i, (X, Y) in enumerate(train_loader):
            X, Y = X.to(config.device), Y.to(config.device)
            current_losses = {}
            for name, model in models.items():
                out = model(X)
                loss = F.mse_loss(out, Y)
                current_losses[name] = loss.item()
                losses[name].append(loss.item())

                optimizers[name].zero_grad()
                loss.backward()
                optimizers[name].step()

                if hasattr(config, "scheduler") and config.scheduler:
                    schedulers[name].step()

            loss_str = ", ".join([f"{name}: {loss:.4f}" for name, loss in current_losses.items()])
            print(f"Epoch {epoch+1}/{config.epochs}, Step {i}, Losses: {loss_str}")

        
        if config.val_size != 0.0:
            for name, model in models.items():
                val_loss = 0
                with torch.no_grad():
                    for X_val, Y_val in val_loader:
                        X_val, Y_val = X_val.to(config.device), Y_val.to(config.device)
                        out = model(X_val)
                        val_loss += F.mse_loss(out, Y_val).item()
                val_loss /= len(val_loader)
                val_losses[name].append(val_loss)
            val_loss_str = ", ".join([f"{name}: {val_losses[name][-1]:.4f}" for name in val_losses.keys()])
            print(f"Epoch {epoch+1}/{config.epochs}, Validation Losses: {val_loss_str}")
        
        if config.val_size == 0.0:
            val_losses = None

    return losses, val_losses

if __name__ == "__main__":
    dim_input = 512
    dim_output = 256
    import math
    config = Config(
        dim_input=dim_input, dim_output=dim_output,
        n_samples=2048,
        data_seed=22, model_seed=1044,
        dataset_type="linear_singular",  
        noise_type="gaussian", 
        epochs=75, 
        batch_size=4096,
        normalize_features=True, 
        scale_up=1.0,
        val_size=0.0, 
        condition_number=100, 
        max_singular_value=  10,
        snr=300,
        scheduler= True,
        constant_proportion=0.3, 
        )

    # Dataset selection
    if config.dataset_type == "linear":
        dataset = LinearRegressionDataset(
            dim_input=config.dim_input,
            dim_output=config.dim_output,
            N_samples=config.n_samples,
            noise_type=config.noise_type,
            noise_level=config.noise_level,
            weight_init=config.weight_init,
            seed=config.data_seed,
            covariance_type="full",
            covariance_strength=config.cov_strength,
            normalize_features=config.normalize_features,
        )
    elif config.dataset_type == "linear_singular":
        dataset = LinearRegressionSingDataset(
            dim_input=config.dim_input,
            dim_output=config.dim_output,
            N_samples=config.n_samples,
            weight_init=config.weight_init,
            noise_type=config.noise_type,
            seed=config.data_seed,
            snr=config.snr,
            sing_dist=config.sing_dist,
            condition_number=config.condition_number,
            normalize_features=config.normalize_features,
            max_singular_value=config.max_singular_value,
            scale_up=config.scale_up
        )
    elif config.dataset_type == "logistic":
        dataset = LogisticRegressionDataset(
            dim_input=config.dim_input,
            num_classes=config.num_classes,
            N_samples=config.n_samples,
            noise_type=config.noise_type,
            noise_level=0.9,
            weight_init=config.weight_init,
            seed=config.data_seed,
            covariance_type="full",
            covariance_strength=config.cov_strength,
            normalize_features=config.normalize_features,
        )
    elif config.dataset_type == "mnist":
        from datasets import MnistDataset
        dataset = MnistDataset(root="./data", train=True)
    else:
        raise ValueError(f"Unknown dataset_type: {config.dataset_type}")

    optimizers_dict = {
        "SGD": (optim.SGD, {
            "lr": 0.8, "weight_decay": 0.1, "momentum": 0.95, 
        }),
        "AdamW": (optim.AdamW, {
            "lr": 0.1, "weight_decay": 0.1, "betas": (0.95, 0.95)
        }),
        "Muon": (Muon, {
            "lr": 0.5, "weight_decay": 0.1, "momentum": 0.9
        }),
    }

    losses, val_losses = compare_optimizers(
        optimizers_dict,
        dataset,
        config,
    )
    extended_title = (
        f"{config.dataset_type} Model Losses\n"
        f"Dataset: {dataset.__class__.__name__}\n"
        f"Optimizer: {', '.join(optimizers_dict.keys())}\n"
        f"Epochs: {config.epochs}, Batch size: {config.batch_size}, N_samples: {config.n_samples}\n"
        f"Noise type: {getattr(dataset, 'noise_type', 'N/A')}, SNR: {getattr(dataset, 'snr', 'N/A')}, max_s_val: {getattr(dataset, 'max_singular_value', 'N/A')}\n"
        f"Weight init: {getattr(dataset, 'weight_init', 'N/A')}, Seed: {getattr(dataset, 'seed', 'N/A')}\n"
    )

    if hasattr(dataset, "condition_number"):
        extended_title += f"Condition number: {dataset.condition_number}, Singular dist: {getattr(dataset, 'sing_dist', 'N/A')}\n"

    SAVE_PATH = f"./plots/{config.dataset_type}/{config.dataset_type}_losses.png"
    import os
    if os.path.exists(SAVE_PATH):
        i = 1
        while os.path.exists(SAVE_PATH):
            SAVE_PATH = f"./plots/{config.dataset_type}/{config.dataset_type}_losses_{i}.png"
            i += 1

    plot_training_validation_losses(
        title=extended_title,
        losses=losses,
        val_losses=val_losses,
        batch_size=config.batch_size,
        n_samples=config.n_samples,
        val_size=config.val_size,
        figsize=(12, 6),
        ylim=None,
        style='default',
        save_path=SAVE_PATH,
        show=False
    )
    X_cov = dataset.X.T @ dataset.X / dataset.X.shape[0]
    eigenvalues = torch.linalg.eigvalsh(X_cov)
    actual_condition = torch.sqrt(eigenvalues.max() / eigenvalues.min())
    print(f"Actual condition number: {actual_condition}")