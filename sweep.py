import itertools
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
    dim_input: int = 200
    dim_output: int = 40
    num_classes: int = 10
    n_samples: int = 10_000
    model_seed: int = 99
    snr: float = 100
    condition_number: float = 100
    sing_dist: str = "normal"
    normalize_features: bool = True
    batch_size: int = 1000
    epochs: int = 5
    max_singular_value: float = 5
    noise_level: float = 0.01  # only used for linear regression datasets
    val_size: float = 0.2
    shuffle: bool = True
    device: str = "cpu"
    separate_bias: bool = True

# Define sweep ranges
lrs = [0.01, 0.05, 0.1]
wds = [0.0, 0.01, 0.1]
moms = [0.8, 0.9, 0.95]
condition_numbers = [10, 100, 1000]
snrs = [10, 100, 1000]

results = []

for cond, snr in itertools.product(condition_numbers, snrs):
    for lr, wd, mom in itertools.product(lrs, wds, moms):
        config = Config(
            condition_number=cond,
            snr=snr,
            # You can override other config fields here if needed
        )

        optimizers_dict = {
            "SGD": (SGD, {
                "lr": lr, "weight_decay": wd, "momentum": mom
            }),
            "AdamW": (AdamW, {
                "lr": lr, "weight_decay": wd, "betas": (mom, mom)
            }),
            "Muon": (Muon, {
                "lr": lr, "weight_decay": wd, "momentum": mom
            }),
        }

        # Dataset selection (example for linear_singular)
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

        losses, val_losses = compare_optimizers(
            optimizers_dict,
            dataset,
            config,
        )

        results.append({
            "config": config,
            "lr": lr,
            "wd": wd,
            "momentum": mom,
            "condition_number": cond,
            "snr": snr,
            "losses": losses,
            "val_losses": val_losses,
        })

        print(f"Finished: lr={lr}, wd={wd}, mom={mom}, cond={cond}, snr={snr}")

# Optionally, save results to disk for later analysis
import pickle
with open("sweep_results.pkl", "wb") as f:
    pickle.dump(results, f)