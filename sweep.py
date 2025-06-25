import itertools
import json
import numpy as np
from tqdm import tqdm
import torch
from optimizers import Muon
from torch.optim import SGD, AdamW
from datasets import LinearRegressionSingDataset
from models import LinearRegressionModel
from dataclasses import dataclass
from utils import compare_optimizers, make_loaders, make_models_and_optimizers

@dataclass
class Config:
    dataset_type: str = "linear_singular"
    noise_type: str = "gaussian"
    weight_init: str = "gaussian"
    data_seed: int = 33
    dim_input: int = 512
    dim_output: int = 64
    n_samples: int = 4096 
    model_seed: int = 44
    snr: float = 300
    condition_number: float = 100
    sing_dist: str = "normal"
    normalize_features: bool = True
    batch_size: int = 1000
    epochs: int = 5
    max_singular_value: float = 5
    val_size: float = 0.2
    shuffle: bool = True
    device: str = "cpu"
    separate_bias: bool = False
    scheduler: bool = False  
    constant_proportion: float = 0.3

# Hyperparameter settings
lrs = [1e-3, 1e-1]
wds = [0.1]   
moms = [0.9, 0.95]  
condition_numbers = [10, 100, 1000]
snrs = [300]
seeds = [1, 2]

# Dictionary to store all results, organized by condition number
all_results = {cond: [] for cond in condition_numbers}

# Dictionary to store best results for each optimizer at each condition number
best_results = {cond: {} for cond in condition_numbers}

# Run experiments
print(f"Starting sweep with {len(lrs) * len(wds) * len(moms) * len(condition_numbers) * len(snrs)} configurations")

for cond, snr in itertools.product(condition_numbers, snrs):
    print(f"\nProcessing condition number {cond}, SNR {snr}")
    
    for lr, wd, mom in itertools.product(lrs, wds, moms):
        print(f"\nTesting lr={lr}, wd={wd}, mom={mom}")
        
        # Lists to store results across seeds
        all_seed_losses = []
        all_seed_val_losses = []
        
        # Run for each seed
        for seed in tqdm(seeds, desc=f"Seeds"):
            # Configure experiment
            config = Config(
                condition_number=cond,
                snr=snr,
                data_seed=seed,
                model_seed=seed,
            )
            
            # Setup optimizers
            optimizers_dict = {
                "SGD": (SGD, {"lr": lr, "weight_decay": wd, "momentum": mom}),
                "AdamW": (AdamW, {"lr": lr, "weight_decay": wd, "betas": (mom, mom)}),
                "Muon": (Muon, {"lr": lr, "weight_decay": wd, "momentum": mom}),
            }
            
            # Create dataset
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
            )
            
            # Run experiment
            try:
                losses, val_losses = compare_optimizers(
                    optimizers_dict,
                    dataset,
                    config,
                )
                # Print keys to help with debugging
                print(f"  Seed {seed} optimizer keys: {list(losses.keys())}")
                
                all_seed_losses.append(losses)
                if val_losses is not None:
                    all_seed_val_losses.append(val_losses)
            except Exception as e:
                print(f"  Error running seed {seed}: {str(e)}")
                continue
        
        # Skip this configuration if no successful runs
        if not all_seed_losses:
            print("  No successful runs for this configuration, skipping")
            continue
        
        # Get all unique optimizer keys from all seeds
        all_optimizer_keys = set()
        for seed_losses in all_seed_losses:
            all_optimizer_keys.update(seed_losses.keys())
        
        # Extract base optimizer names (everything before any comma)
        base_optimizers = {}
        for key in all_optimizer_keys:
            base_name = key.split(',')[0].strip()
            if base_name not in base_optimizers:
                base_optimizers[base_name] = []
            base_optimizers[base_name].append(key)
        
        # Average losses for each base optimizer across seeds
        mean_losses = {}
        std_losses = {}
        mean_val_losses = {}
        std_val_losses = {}
        
        for base_name, full_keys in base_optimizers.items():
            # Process each full key
            for full_key in full_keys:
                # Get all losses for this key across seeds
                key_losses = []
                for seed_losses in all_seed_losses:
                    if full_key in seed_losses:
                        key_losses.append(seed_losses[full_key])
                
                # Skip if no data
                if not key_losses:
                    continue
                
                # Find minimum length (in case different seeds have different numbers of steps)
                min_length = min(len(losses) for losses in key_losses)
                
                # Trim all to minimum length and convert to numpy array
                trimmed_losses = [np.array(losses[:min_length]) for losses in key_losses]
                stacked_losses = np.stack(trimmed_losses)
                
                # Calculate mean and std
                mean_losses[full_key] = stacked_losses.mean(axis=0).tolist()
                std_losses[full_key] = stacked_losses.std(axis=0).tolist()
                
                # Do the same for validation losses if available
                if all_seed_val_losses:
                    key_val_losses = []
                    for seed_val_losses in all_seed_val_losses:
                        if full_key in seed_val_losses:
                            key_val_losses.append(seed_val_losses[full_key])
                    
                    if key_val_losses:
                        min_val_length = min(len(losses) for losses in key_val_losses)
                        trimmed_val_losses = [np.array(losses[:min_val_length]) for losses in key_val_losses]
                        stacked_val_losses = np.stack(trimmed_val_losses)
                        
                        mean_val_losses[full_key] = stacked_val_losses.mean(axis=0).tolist()
                        std_val_losses[full_key] = stacked_val_losses.std(axis=0).tolist()
        
        # Store results for this configuration
        result = {
            "lr": lr,
            "wd": wd,
            "momentum": mom,
            "snr": snr,
            "condition_number": cond,
            "seeds": seeds,
            "mean_losses": mean_losses,
            "std_losses": std_losses,
        }
        
        if mean_val_losses:
            result["mean_val_losses"] = mean_val_losses
            result["std_val_losses"] = std_val_losses
        
        all_results[cond].append(result)
        
        # Update best results for each base optimizer
        for base_name, full_keys in base_optimizers.items():
            # Initialize if this is the first configuration for this optimizer
            if base_name not in best_results[cond]:
                best_results[cond][base_name] = {
                    "best_final_loss": float('inf'),
                    "config": None
                }
            
            # Check each full key for this base optimizer
            for full_key in full_keys:
                if full_key in mean_losses:
                    # Get the final loss
                    final_loss = mean_losses[full_key][-1]
                    
                    # Update if better than current best
                    if final_loss < best_results[cond][base_name]["best_final_loss"]:
                        best_results[cond][base_name] = {
                            "best_final_loss": final_loss,
                            "config": {
                                "lr": lr,
                                "wd": wd,
                                "momentum": mom,
                                "snr": snr,
                                "key": full_key,
                                "mean_losses": mean_losses[full_key],
                                "std_losses": std_losses[full_key],
                            }
                        }
                        
                        # Add validation losses if available
                        if full_key in mean_val_losses:
                            best_results[cond][base_name]["config"]["mean_val_losses"] = mean_val_losses[full_key]
                            best_results[cond][base_name]["config"]["std_val_losses"] = std_val_losses[full_key]
        
        print(f"  Completed configuration: lr={lr}, wd={wd}, mom={mom}")

# Make numpy arrays JSON serializable
def make_json_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.floating, np.integer)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    else:
        return obj

# Prepare final results dictionary
final_output = {
    "hyperparameters": {
        "lrs": lrs,
        "wds": wds,
        "moms": moms,
        "condition_numbers": condition_numbers,
        "snrs": snrs,
        "seeds": seeds
    },
    "all_results": make_json_serializable(all_results),
    "best_performing": make_json_serializable(best_results)
}

# Save to file
output_filename = f"sweep_results_conds{'-'.join(str(c) for c in condition_numbers)}.json"
with open(output_filename, 'w') as f:
    json.dump(final_output, f, indent=2)

print(f"\nSaved results to {output_filename}")

# Print summary of best results
print("\nBest results summary:")
for cond in condition_numbers:
    print(f"\nCondition number {cond}:")
    for optimizer, result in best_results[cond].items():
        config = result["config"]
        print(f"  {optimizer}: final loss = {result['best_final_loss']:.6f} (lr={config['lr']}, wd={config['wd']}, mom={config['momentum']})")