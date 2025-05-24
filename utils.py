import matplotlib.pyplot as plt
import numpy as np
from datasets import LinearRegressionDataset, LogisticRegressionDataset, MnistDataset



def plot_training_validation_losses(
    title, losses, val_losses, batch_size, n_samples, val_size=0.2,
    figsize=(12, 6), ylim=None, style='default', save_path=None, show=False

):
    """
    Plot the training and validation losses together.
    
    Args:
        losses: Dictionary of {optimizer_name: list_of_training_losses}
        val_losses: Dictionary of {optimizer_name: list_of_validation_losses}
        batch_size: Batch size used during training
        n_samples: Total number of samples in the dataset
        val_size: Proportion of dataset used for validation (default: 0.2)
        figsize: Figure size (width, height)
        ylim: Y-axis limits (min, max) or None for auto
        style: Matplotlib style
        save_path: Path to save the figure or None
    """
    #validation only every epoch so need per steps for val and train on same plot
    train_samples = int(n_samples * (1 - val_size))
    steps_per_epoch = train_samples // batch_size
    
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
        
        if name in val_losses and len(val_losses[name]) > 0:
            # Calculate x-coordinates for validation points (end of each epoch)
            val_x = [min((epoch+1) * steps_per_epoch - 1, len(losses[name])-1) 
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

        if isinstance(dataset, LinearRegressionDataset):
            from models import LinearRegressionModel
            model = LinearRegressionModel(
                dim_input=dataset.dim_input,
                dim_output=dataset.dim_output,
                seed=model_seed
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
        optimizers[cfg_name] = optim_class(params=[model.W], **optim_kwargs)

    return models, optimizers

