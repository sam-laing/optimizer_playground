import optuna
import torch
import torch.optim as optim
from typing import Dict, Tuple, List, Any, Callable, Type, Union
from dataclasses import dataclass
import numpy as np

from datasets import LinearRegressionDataset, LogisticRegressionDataset
from models import LinearRegressionModel, LogisticRegressionModel
from main import compare_optimizers
from optimizers import SimpleMuon  

def optimize_hyperparams(
    optimizer_class: Type,
    param_space: Dict[str, Dict[str, Any]],
    dataset,
    n_trials: int = 50,
    epochs: int = 30,
    batch_size: int = 128,
    model_seed: int = 99,
    val_size: float = 0.2,
    study_name: str = None,
    verbose: bool = True
) -> Tuple[Dict[str, Any], float]:
    """
    Generic hyperparameter optimizer for any PyTorch optimizer.
    
    Args:
        optimizer_class: The optimizer class (e.g., optim.SGD, SimpleMuon)
        param_space: Dictionary defining the search space for each parameter
            Format: {
                "param_name": {
                    "type": "float"|"int"|"categorical",
                    "min": min_value,  # For float, int
                    "max": max_value,  # For float, int
                    "log": True|False,  # For float (optional)
                    "choices": [...]  # For categorical
                }
            }
        dataset: Dataset to use for optimization
        n_trials: Number of trials for hyperparameter search
        epochs: Number of training epochs per trial
        batch_size: Training batch size
        model_seed: Random seed for model initialization
        val_size: Validation set size (0-1)
        study_name: Name for the optuna study
        verbose: Whether to print progress
        
    Returns:
        best_params: Dictionary of best hyperparameters
        best_value: Best validation loss achieved
    """
    if study_name is None:
        study_name = f"{optimizer_class.__name__}_optimization"
    
    def objective(trial):
        # Build hyperparameters from the defined search space
        params = {}
        for name, space in param_space.items():
            if space["type"] == "float":
                params[name] = trial.suggest_float(
                    name, space["min"], space["max"], 
                    log=space.get("log", False)
                )
            elif space["type"] == "int":
                params[name] = trial.suggest_int(
                    name, space["min"], space["max"], 
                    log=space.get("log", False)
                )
            elif space["type"] == "categorical":
                params[name] = trial.suggest_categorical(name, space["choices"])
        
        # Handle special cases like betas for Adam
        if "beta1" in params and "beta2" in params:
            beta1 = params.pop("beta1")
            beta2 = params.pop("beta2")
            params["betas"] = (beta1, beta2)
        
        # Configure optimizer with suggested hyperparameters
        optimizer_name = optimizer_class.__name__
        optimizers_dict = {
            optimizer_name: (optimizer_class, params)
        }
        
        # Train and evaluate
        _, val_losses = compare_optimizers(
            optimizers_dict,
            dataset,
            batch_size=batch_size,
            epochs=epochs,
            model_seed=model_seed,
            val_size=val_size,
            shuffle=True,
            verbose=False  # Reduce output noise during optimization
        )
        
        # Get validation loss for the optimizer (the key format depends on your compare_optimizers function)
        key = list(val_losses.keys())[0]
        return val_losses[key][-1]  # Final validation loss
    
    # Create and run optimization study
    study = optuna.create_study(direction="minimize", study_name=study_name)
    study.optimize(objective, n_trials=n_trials)
    
    if verbose:
        print(f"\n{optimizer_class.__name__} Best Hyperparameters:")
        print(f"  {study.best_params}")
        print(f"  Validation Loss: {study.best_value:.6f}")
    
    return study.best_params, study.best_value


# Pre-defined parameter spaces for common optimizers
SGD_PARAM_SPACE = {
    "lr": {"type": "float", "min": 1e-4, "max": 1.0, "log": True},
    "momentum": {"type": "float", "min": 0.5, "max": 0.99},
    "weight_decay": {"type": "float", "min": 1e-5, "max": 1e-1, "log": True}
}

ADAM_PARAM_SPACE = {
    "lr": {"type": "float", "min": 1e-5, "max": 1e-2, "log": True},
    "beta1": {"type": "float", "min": 0.8, "max": 0.99},
    "beta2": {"type": "float", "min": 0.95, "max": 0.9999},
    "weight_decay": {"type": "float", "min": 1e-5, "max": 1e-1, "log": True}
}

ADAMW_PARAM_SPACE = ADAM_PARAM_SPACE  # Same search space

MUON_PARAM_SPACE = {
    "lr": {"type": "float", "min": 1e-4, "max": 1.0, "log": True},
    "momentum": {"type": "float", "min": 0.8, "max": 0.99},
    "weight_decay": {"type": "float", "min": 1e-5, "max": 1e-1, "log": True},
    "ns_steps": {"type": "int", "min": 3, "max": 8}
}


def optimize_multiple_optimizers(
    optimizer_configs: List[Tuple[Type, Dict]],
    dataset,
    n_trials: int = 20,
    epochs: int = 5,
    batch_size: int = 128,
    model_seed: int = 99,
    val_size: float = 0.2
) -> Dict[str, Dict]:
    """
    Optimize multiple optimizers in sequence.
    
    Args:
        optimizer_configs: List of (optimizer_class, param_space) tuples
        dataset: Dataset to use for optimization
        n_trials: Number of trials per optimizer
        epochs: Number of epochs per trial
        batch_size: Batch size for training
        model_seed: Random seed for model initialization
        val_size: Validation set size
        
    Returns:
        Dictionary with optimizer results {optimizer_name: {"params": best_params, "val_loss": best_val_loss}}
    """
    results = {}
    
    for optimizer_class, param_space in optimizer_configs:
        optimizer_name = optimizer_class.__name__
        print(f"\nOptimizing {optimizer_name} hyperparameters ({n_trials} trials)...")
        
        best_params, best_val = optimize_hyperparams(
            optimizer_class,
            param_space,
            dataset,
            n_trials=n_trials,
            epochs=epochs,
            batch_size=batch_size,
            model_seed=model_seed,
            val_size=val_size,
            study_name=f"{optimizer_name}_optimization"
        )
        
        results[optimizer_name] = {
            "params": best_params,
            "val_loss": best_val
        }
    
    return results


def run_comparison_with_optimized_hyperparams(
    optimization_results: Dict[str, Dict],
    dataset,
    epochs: int = 10,
    batch_size: int = 128,
    model_seed: int = 99,
    val_size: float = 0.2
):
    """
    Run a comparison using the optimized hyperparameters.
    
    Args:
        optimization_results: Results from optimize_multiple_optimizers
        dataset: Dataset to use
        epochs: Number of training epochs
        batch_size: Training batch size
        model_seed: Random seed for model initialization
        val_size: Validation set size
        
    Returns:
        train_losses, val_losses: Training and validation losses
    """
    optimizers_dict = {}
    
    for optimizer_name, result in optimization_results.items():
        params = result["params"].copy()
        
        # Handle special parameter reconstruction
        if "beta1" in params and "beta2" in params:
            beta1 = params.pop("beta1")
            beta2 = params.pop("beta2")
            params["betas"] = (beta1, beta2)
        
        # Map optimizer name to class
        if optimizer_name == "SGD":
            optimizer_class = optim.SGD
        elif optimizer_name == "Adam":
            optimizer_class = optim.Adam
        elif optimizer_name == "AdamW":
            optimizer_class = optim.AdamW
        elif optimizer_name == "SimpleMuon":
            optimizer_class = SimpleMuon
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        display_name = f"{optimizer_name} (optimized)"
        optimizers_dict[display_name] = (optimizer_class, params)
    
    print("\nRunning comparison with optimized hyperparameters...")
    train_losses, val_losses = compare_optimizers(
        optimizers_dict,
        dataset,
        batch_size=batch_size,
        epochs=epochs,
        model_seed=model_seed,
        val_size=val_size,
        shuffle=True
    )
    
    return train_losses, val_losses


# Example usage
if __name__ == "__main__":
    # Create dataset
    dataset = LinearRegressionDataset(
        dim_input=100,
        dim_output=10,
        N_samples=5000,
        noise_type="gaussian",
        noise_level=0.1,
        seed=42
    )
    
    # Define optimizers to optimize
    optimizer_configs = [
        (optim.SGD, SGD_PARAM_SPACE),
        (optim.AdamW, ADAMW_PARAM_SPACE),
        (SimpleMuon, MUON_PARAM_SPACE)
    ]
    
    # Run optimization
    results = optimize_multiple_optimizers(
        optimizer_configs,
        dataset,
        n_trials=10,  # Low for quick demo, use 50-100 in practice
        epochs=3,
        batch_size=128
    )
    
    # Compare optimizers with optimized hyperparameters
    train_losses, val_losses = run_comparison_with_optimized_hyperparams(
        results,
        dataset,
        epochs=5,
        batch_size=128
    )
    
    # Print summary of best hyperparameters
    print("\n===== OPTIMIZATION RESULTS =====")
    for optimizer_name, result in results.items():
        print(f"{optimizer_name}:")
        print(f"  Best params: {result['params']}")
        print(f"  Validation loss: {result['val_loss']:.6f}")