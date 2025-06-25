import matplotlib.pyplot as plt
import numpy as np
from datasets import LinearRegressionDataset, LinearRegressionSingDataset, LogisticRegressionDataset, MnistDataset, make_loaders
import torch 
import torch.nn.functional as F
from torch import nn
from optimizers import Muon 


def plot_training_validation_losses(
    title, losses, val_losses=None, batch_size=1, n_samples=1, val_size=0.2,
    figsize=(12, 6), ylim=None, style='default', save_path=None, show=False
):
    """
    Plot the training and (optionally) validation losses together.

    Args:
        losses: Dictionary of {optimizer_name: list_of_training_losses}
        val_losses: Dictionary of {optimizer_name: list_of_validation_losses} or None
        batch_size: Batch size used during training
        n_samples: Total number of samples in the dataset
        val_size: Proportion of dataset used for validation (default: 0.2)
        figsize: Figure size (width, height)
        ylim: Y-axis limits (min, max) or None for auto
        style: Matplotlib style
        save_path: Path to save the figure or None
        show: Whether to display the plot
    """
    train_samples = int(n_samples * (1 - val_size))
    steps_per_epoch = max(1, train_samples // batch_size)

    plt.style.use(style)
    fig, ax = plt.subplots(figsize=figsize)

    optimizer_names = list(losses.keys())
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    color_map = {name: colors[i % len(colors)] for i, name in enumerate(optimizer_names)}

    for name in optimizer_names:
        ax.plot(
            losses[name],
            color=color_map[name],
            linewidth=2,
            label=f"{name} (train)",
            alpha=0.8
        )

        # Only plot validation losses if provided and non-empty for this optimizer
        if val_losses is not None and name in val_losses and len(val_losses[name]) > 0:
            val_x = [min((epoch + 1) * steps_per_epoch - 1, len(losses[name]) - 1)
                     for epoch in range(len(val_losses[name]))]
            ax.plot(
                val_x, val_losses[name],
                color=color_map[name],
                linestyle='--',
                linewidth=2,
                marker='o',
                markersize=6,
                label=f"{name} (val)"
            )

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Training Steps", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')

    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(loc='upper right', frameon=True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def make_models_and_optimizers(
        dataset, 
        optimizers_dict,
        model_seed=0,
        device='cpu',
):
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

        if isinstance(dataset, LinearRegressionDataset) or isinstance(dataset, LinearRegressionSingDataset):
            from models import LinearRegressionModel
            model = LinearRegressionModel(
                dim_input=dataset.dim_input,
                dim_output=dataset.dim_output,
                model_seed=model_seed, 
                separate_bias=False
            )
        

        elif isinstance(dataset, LogisticRegressionDataset):   
            from models import LogisticRegressionModel 
            model = LogisticRegressionModel(
                dim_input=dataset.dim_input, 
                num_classes=dataset.num_classes,
                seed=model_seed
            )
        
        elif isinstance(dataset, MnistDataset):
            from models.simpleMLP import SimpleMLP
            model = SimpleMLP(
                dim_input=dataset.dim_input,
                dim_output=dataset.dim_output,
                seed=model_seed
            )
        else:
            raise ValueError("Unknown dataset type")

        models[cfg_name] = model.to(device)
        if optim_class is Muon:
            print(f"muon being used {optim_class} ")
            optimizers[cfg_name] = optim_class(muon_params=[model.W], **optim_kwargs)
        else:
            print(f"standard optimizer being used {optim_class} ")
            optimizers[cfg_name] = optim_class(params=[model.W], **optim_kwargs)

    return models, optimizers

def compare_optimizers(
    optimizers_dict: dict,
    dataset,
    config,
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

            loss_str = ", ".join([f"{name}: {loss:.4f}" for name, loss in current_losses.items()])
            print(f"Epoch {epoch+1}/{config.epochs}, Step {i}, Losses: {loss_str}")

        # Validation
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

    return losses, val_losses

import math
from torch.optim.lr_scheduler import _LRScheduler

class ConstantThenCosineLR(_LRScheduler):
    """
    Starts with constant learning rate for a proportion of steps,
    then decays with cosine annealing to zero.
    
    Args:
        optimizer (Optimizer): Wrapped optimizer
        constant_proportion (float): Proportion of training to keep LR constant (0-1)
        total_steps (int): Total number of training steps
        last_epoch (int): The index of the last step. Default: -1
    """
    def __init__(self, optimizer, constant_proportion, total_steps, last_epoch=-1):
        self.constant_steps = int(constant_proportion * total_steps)
        self.cosine_steps = total_steps - self.constant_steps
        self.total_steps = total_steps
        super(ConstantThenCosineLR, self).__init__(optimizer, last_epoch)
        
    def get_lr(self):
        if self.last_epoch < self.constant_steps:
            # Constant phase
            return [base_lr for base_lr in self.base_lrs]
        else:
            # Cosine decay phase
            current_cosine_step = self.last_epoch - self.constant_steps
            completion = current_cosine_step / self.cosine_steps
            cosine_factor = 0.5 * (1 + math.cos(math.pi * completion))
            return [base_lr * cosine_factor for base_lr in self.base_lrs]