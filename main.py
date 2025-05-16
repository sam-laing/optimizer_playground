from datasets import LinearRegressionDataset, LogisticRegressionDataset, make_loaders
from models import LinearRegressionModel, LogisticRegressionModel
from utils import plot_training_validation_losses, make_models_and_optimizers


import torch
import torch.nn.functional as F
import torch.nn as nn

import torch.optim as optim
from torch import Tensor
from optimizers import SimpleMuon

#kind of a useless inclusion but whatever
def loss_function(W, X, Y):
    """
    Objective function for linear regression problem
    L(W) = ||Y - XW||^2 + lambda * ||W||^2
    """
    loss = F.mse_loss(X @ W, Y)
    return loss


def compare_optimizers(
        optimizers_dict: dict,  # Format: {'optim_name': (optim_class, optim_kwargs)}
        dataset: LinearRegressionDataset,
        device: str = "cpu",
        batch_size: int = 128,
        epochs: int = 10,
        model_seed: int = 99,
        sampler_seed: int = 99,
        val_size: int = 0.2,
        shuffle: bool = True,

    ):
    """
    Compare multiple optimizers by training linear regression models on dataset.

    Args:
        optimizers_dict: Dictionary where keys are optimizer names and values are tuples of
                        (optimizer_class, optimizer_kwargs)
        dataset: Dataset object
        batch_size: Batch size for data loading
        epochs: Number of training epochs
        model_seed: Random seed for model initialization
        val_size: Size of validation set (if None, use the entire dataset for training)
        shuffle: Whether to shuffle the data
    Returns:   
        Dictionary containing loss histories for each optimizer
    """
    #dataloading 
    if val_size is not None:
        train_loader, val_loader = make_loaders(
            dataset, batch_size=batch_size, seed=sampler_seed, val_size=val_size, shuffle=shuffle
        )
    else:
        raise ValueError("val_size must be in (0,1)... won't accept no val")

    models, optimizers = make_models_and_optimizers(
        dataset,
        optimizers_dict,
        model_seed=model_seed,
        device=device
    )

    losses = {name: [] for name in models.keys()}
    val_losses = {name: [] for name in models.keys()}
    for epoch in range(epochs):
        for i, (X, Y) in enumerate(train_loader):
            X, Y = X.to(device), Y.to(device)
            current_losses = {}
            for name, model in models.items():
                out = model(X)
                loss = loss_function(model.get_weights(), X, Y)
                current_losses[name] = loss.item()
                losses[name].append(loss.item())

                optimizers[name].zero_grad()
                loss.backward()
                optimizers[name].step()

            loss_str = ", ".join([f"{name}: {loss:.4f}" for name, loss in current_losses.items()])
            print(f"Epoch {epoch+1}/{epochs}, Step {i}, Losses: {loss_str}")
        
        #val once per epoch
        for name, model in models.items():
            val_loss = 0
            with torch.no_grad():
                for X_val, Y_val in val_loader:
                    X_val, Y_val = X_val.to(device), Y_val.to(device)
                    out = model(X_val)
                    val_loss += loss_function(model.get_weights(), X_val, Y_val).item()
            
            val_loss /= len(val_loader)
            val_losses[name].append(val_loss)
        #join val losses[name][-1] into one string for all name to display the val loss of each optimizer
        val_loss_str = ", ".join([f"{name}: {val_losses[name][-1]:.4f}" for name in val_losses.keys()])

        print(f"Epoch {epoch+1}/{epochs}, Validation Losses: {val_loss_str}")

    
    

    return losses, val_losses


if __name__ == "__main__":
    from dataclasses import dataclass


    @dataclass
    class Config:
        dim_input: int
        dim_output: int
        N_samples: int
        noise_type: str
        noise_level: float
        weight_init: str
        seed: int
        covariance_type: str = None
        covariance_strength: float = 0.5
        batch_size: int = 128
        epochs: int = 10
        model_seed: int = 99
        val_size: float = 0.2
        shuffle: bool = True
        device: str = "cpu"

    DATASET_TYPE = "linear"  # or "logistic"
    NOISE_TYPE = "laplace"  # or "uniform", "laplace"
    WEIGHT_INIT = "gaussian"  # or "uniform", "laplace"
    DATA_SEED = 33
    DIM_INPUT = 200
    DIM_OUTPUT = 10
    NUM_CLASSES = 10
    N_SAMPLES = 10_000
    COV_STRENGTH = 0.6
    NORMALIZE_FEATURES = True


    if DATASET_TYPE == "linear":
        dataset = LinearRegressionDataset(
            dim_input=200,
            dim_output=10,
            N_samples=N_SAMPLES,
            noise_type=NOISE_TYPE,
            noise_level=0.01,
            weight_init=WEIGHT_INIT,
            seed=DATA_SEED,
            covariance_type="full",
            covariance_strength=COV_STRENGTH,
            normalize_features=NORMALIZE_FEATURES
        )
    elif DATASET_TYPE == "logistic":
        dataset = LogisticRegressionDataset(
            dim_input=200,
            num_classes=NUM_CLASSES,
            N_samples=N_SAMPLES,
            noise_type=NOISE_TYPE,
            noise_level=0.9,
            weight_init=WEIGHT_INIT,
            seed=DATA_SEED, 
            covariance_type="full", 
            covariance_strength=COV_STRENGTH,
            normalize_features=NORMALIZE_FEATURES
        )
    elif DATASET_TYPE == "mnist":
        from datasets import MnistDataset
        dataset = MnistDataset(root = "./data", train=True)
        

    """    
    optimizers_dict = {
        "SGD": (optim.SGD, {
            "lr": 0.05, "weight_decay": 0.1, "momentum": 0.9
            }),
        "AdamW": (optim.AdamW, {
            "lr": 0.01, "weight_decay": 0.1, "betas": (0.9, 0.999)
            }),
        "SimpleMuon": (SimpleMuon, {
            "lr": 0.1, "weight_decay": 0.1, "momentum": 0.8
            }),
    }
    """    
    from optimizers import CustomAdamW 

    betas = (0.95, 0.95)
    optimizers_dict = {
        "BC and ZI": (CustomAdamW, {
            "lr": 0.01, "weight_decay": 0.1, "betas": (betas), "do_bias_correction": True, "zero_init": True
            }),
        "no BC and ZI": (CustomAdamW, {
            "lr": 0.01, "weight_decay": 0.1, "betas": (betas), "do_bias_correction": False,  "zero_init": True
            }),
        "no BC and no ZI": (CustomAdamW, {
            "lr": 0.01, "weight_decay": 0.1, "betas": (betas), "do_bias_correction": False, "zero_init": False
            }),
        "BC and no ZI": (CustomAdamW, {
            "lr": 0.01, "weight_decay": 0.1, "betas": (betas), "do_bias_correction": True, "zero_init": False
            }),
    }



    BATCH_SIZE = 512
    EPOCHS = 5 

    losses, val_losses = compare_optimizers(
        optimizers_dict,
        dataset,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        model_seed=999,
        val_size=0.2,
        shuffle=True, 
  
    )
    extended_title = f"{DATASET_TYPE} Model Losses\n" \
        f"Dataset: {dataset.__class__.__name__}\n" \
        f"Optimizer: {', '.join(optimizers_dict.keys())}\n" \
        f"Epochs: {EPOCHS}, Batch size: {BATCH_SIZE}, N_samples: {dataset.N_samples}\n" \
        f"Noise type: {dataset.noise_type}, Noise level: {dataset.noise_level}\n" \
        f"Weight init: {dataset.weight_init}, Seed: {dataset.seed}\n"
    extended_title += f"Covariance type: {dataset.covariance_type}, Covariance strength: {dataset.covariance_strength}\n"
    SAVE_PATH = f"./plots/{DATASET_TYPE}/{DATASET_TYPE}_losses.png"
    #check if this exact path exists, if so create a new file with a slightly altered name
    #shamefully hacky solution but fine for now
    import os
    if os.path.exists(SAVE_PATH):
        i = 1
        while os.path.exists(SAVE_PATH):
            SAVE_PATH = f"./plots/{DATASET_TYPE}/{DATASET_TYPE}_losses_{i}.png"
            i += 1
        
    plot_training_validation_losses(
        title=extended_title,
        losses=losses,
        val_losses=val_losses,
        batch_size=BATCH_SIZE,
        n_samples=dataset.N_samples,
        val_size=0.2,
        figsize=(12, 6),
        ylim=None,
        style='default',
        save_path=SAVE_PATH, 
        show=False
        )
    