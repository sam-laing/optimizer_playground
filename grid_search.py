from main import compare_optimizers
from torch.optim import AdamW, SGD
from datasets import LinearRegressionDataset
from optimizers import SimpleMuon

"""    
Basically looking to grid search over a predefined set of HPs for each optimizer in parallel (with the compare_optimizers function)
and then get sensitivity curves for each optimizer parameter and optimizer (with other HPs at best performing values).
want to iterate through a zip of the optimizers HPs so each hp should have the same length in optimizers_sweep_dict
    Run a grid search over a predefined set of hyperparameters for each optimizer.
    
    Args:
        dataset: Dataset to use
        optimizers_sweep_dict: Dictionary of optimizers and their hyperparameter ranges
        epochs: Number of training epochs
        batch_size: Training batch size
        model_seed: Random seed for model initialization
        val_size: Validation set size
        shuffle: Whether to shuffle the data
        n_jobs: Number ofVIBE parallel jobs to run
        n_trials: Number of trials for each hyperparameter combination
"""

if __name__ == "__main__":
    # Example usage
    from datasets import LinearRegressionSingDataset

    DATASET_TYPE = "linear_singular" 
    NOISE_TYPE = "gaussian"  # or "uniform", "laplace"
    WEIGHT_INIT = "gaussian"  # or "uniform", "laplace"
    DATA_SEED = 4
    DIM_INPUT = 200
    DIM_OUTPUT = 30
    NUM_CLASSES = 10
    N_SAMPLES = 3_000
    MODEL_SEED = 99
    SNR = 1
    CONDITION_NUMBER = 1000
    SING_DIST = "uniform"  
    COV_STRENGTH = 0.6
    NORMALIZE_FEATURES = True
    BATCH_SIZE = N_SAMPLES
    EPOCHS = 22 

    dataset = LinearRegressionSingDataset(
        dim_input=DIM_INPUT,
        dim_output=DIM_OUTPUT,
        N_samples=N_SAMPLES,
        weight_init=WEIGHT_INIT,
        noise_type=NOISE_TYPE,
        seed=DATA_SEED,
        snr=SNR, 
        sing_dist=SING_DIST,
        condition_number=CONDITION_NUMBER,
    )


    """    
    dataset = LinearRegressionDataset(
        dim_input=200,
        dim_output=10,
        N_samples=10_000,
        noise_type="gaussian",
        noise_level=0.01,
        weight_init="gaussian",
        seed=42
    )
    
    """    
    #to sweep HPs

    lrs = [0.001, 0.01, 0.1]  #[0.001, 0.005, 0.01, 0.1]
    weight_decays = [0.01, 0.1]

    # make a 1-1 correspondence between betas for adam and momentum for SGD and SimpleMuon

    # SGD and SimpleMuon have momentum
    # AdamW has betas

    moms = [0.9, 0.95]
    #betas = [(0.8, 0.9), (0.9, 0.999), (0.9, 0.95), (0.95, 0.95)]
    #moms_betas = zip(moms, betas)



    # one optimizer at a time full grid search: compare_optimizers with dict only having one optimizer

    # SGD
    import itertools
    possible_combos = itertools.product(lrs, weight_decays, moms)

    SGD_dict = {} #each HP combination as key and value tuple of best val loss and final train loss
    to_compare = {}
    for i, (lr, wd, mom) in enumerate(possible_combos):
        to_compare[f"SGD_{i}"] = (SGD, {
            "lr": lr,
            "weight_decay": wd,
            "momentum": mom
        })
    
    #now do compare_optimizers on all simultaneously
    print("Running SGD with hyperparameters:")
    for key, value in to_compare.items():
        print(f"{key}: {value}")
    
    SGD_losses, SGD_val_losses = compare_optimizers(
        to_compare,
        dataset,
        batch_size=128,
        epochs=10,
        model_seed=999,
        sampler_seed=999,
        val_size=0.2,
        shuffle=True, 
        separate_bias=False,
    )
    best_val_dict = {}
    for (name, losses) in SGD_val_losses.items():
        best_val_dict[name] = min(losses)
        print(f"Final val loss for {name}: {min(losses)}")
        print(f"Best val loss for {name}: {min(losses)}")
    best_val = min(best_val_dict.values())


    
 