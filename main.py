from datasets import LinearRegressionDataset, LogisticRegressionDataset, make_loaders
from models import LinearRegressionModel, LogisticRegressionModel


import torch
import torch.nn.functional as F
import torch.nn as nn

import torch.optim as optim
from torch import Tensor




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
        batch_size: int = 128,
        epochs: int = 10,
        model_seed: int = 99,
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
    # Create data loaders
    
    if val_size is not None:
        train_loader, val_loader = make_loaders(
            dataset, batch_size=batch_size, seed=model_seed, val_size=val_size, shuffle=shuffle
        )
    else:
        raise ValueError("val_size must be in (0,1)... won't accept no val")
    
    """
    models = {
        name: LinearRegressionModel(
            dim_input=dataset.dim_input,
            dim_output=dataset.dim_output,
            seed=model_seed
        )

        for name in optimizers_dict.keys()
    }
    #get the learning rate, wd and momentum/ beta1,beta2 from the optimizers_dict with some try except 

    """
    """
    optimizers = {
        name: optim_class(params=[models[name].W], **optim_kwargs)
        for name, (optim_class, optim_kwargs) in optimizers_dict.items()
    }
    """

    models = {}
    optimizers = {}
    for name, (optim_class, optim_kwargs) in optimizers_dict.items():
        try:
            lr = optim_kwargs['lr']
        except KeyError:
            lr = None
        try:
            wd = optim_kwargs['weight_decay']
        except KeyError:
            wd = None
        try:
            beta1 = optim_kwargs['betas'][0]
            beta2 = optim_kwargs['betas'][1]
        except KeyError:
            beta1, beta2 = None, None
            try:
                # get momentum instead
                momentum = optim_kwargs['momentum']
            except KeyError:
                momentum = None
        cfg_name = name
        if lr is not None:
            cfg_name += f", lr:{lr}"
        if wd is not None:
            cfg_name += f", wd:{wd}"
        if beta1 is not None and beta2 is not None:
            cfg_name += f", beta1:{beta1}, beta2:{beta2}"
        elif momentum is not None:
            cfg_name += f", momentum:{momentum}"

        model = LinearRegressionModel(
            dim_input=dataset.dim_input,
            dim_output=dataset.dim_output,
            seed=model_seed
        )
        models[cfg_name] = model
        optimizers[cfg_name] = optim_class(params=[model.W], **optim_kwargs)


    losses = {name: [] for name in models.keys()}
    val_losses = {name: [] for name in models.keys()}

    for epoch in range(epochs):
        for i, (X, Y) in enumerate(train_loader):
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
                    out = model(X_val)
                    val_loss += loss_function(model.get_weights(), X_val, Y_val).item()
            
            val_loss /= len(val_loader)
            val_losses[name].append(val_loss)
        #join val losses[name][-1] into one string for all name to display the val loss of each optimizer
        val_loss_str = ", ".join([f"{name}: {val_losses[name][-1]:.4f}" for name in val_losses.keys()])

        print(f"Epoch {epoch+1}/{epochs}, Validation Losses: {val_loss_str}")

    
    

    return losses, val_losses


if __name__ == "__main__":
    # Example usage
    dataset = LinearRegressionDataset(
        dim_input=200,
        dim_output=10,
        N_samples=10_000,
        noise_type="gaussian",
        noise_level=0.01,
        weight_init="gaussian",
        seed=42
    )
    from optimizers import SimpleMuon
    
    optimizers_dict = {
        "SGD": (optim.SGD, {
            "lr": 0.01, "weight_decay": 0.01
            }),
        "AdamW": (optim.AdamW, {
            "lr": 0.001, "weight_decay": 0.01
            }),
        "SimpleMuon": (SimpleMuon, {
            "lr": 0.01, "weight_decay": 0.01, "momentum": 0.9
            }),
    }
    
    losses, val_losses = compare_optimizers(
        optimizers_dict,
        dataset,
        batch_size=128,
        epochs=5,
        model_seed=99,
        val_size=0.2,
        shuffle=True
    )