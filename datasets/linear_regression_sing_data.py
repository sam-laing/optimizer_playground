import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
from torch import Tensor



class LinearRegressionSingDataset(Dataset):
    """"
    Mock up artificial data which is ideal for a linear regression problem
    Y = XW + noise
    - X is a matrix of shape (N, dim_input + 1) with first column all ones
    - W is a matrix of shape (dim_input + 1, dim_output)
    - Y is a matrix of shape (N, dim_output)
    - noise is a matrix of shape (N, dim_output) with values in [0,1]


    - X is generated from a multivariate normal distribution with optional covariance
    - Y is generated from a linear model with added noise

    Inputs:   
        dim_input: Number of input features (required)
        dim_output: Number of output features (required)
        N_samples: Number of samples to generate (required)
        noise_type: Type of noise to add (gaussian, uniform, laplace)... default is gaussian
        noise_level: Level of noise to add (kind of a percentage of the size of the data).... should make the problem harder
        weight_init: Type of weight initialization (gaussian, uniform, laplace) ... default is gaussian
        seed: Random seed for reproducibility
        snr: Signal to noise ratio (default is 0.05)
        sing_dist: Distribution of singular values (uniform, normal, lognormal) ... default is uniform
        condition_number: Condition number of the feature matrix (default is 1e6)


    """

    def __init__(
        self, dim_input: int, dim_output: int, N_samples: int, 
        noise_type: str = "gaussian", 
        weight_init: str = "gaussian", seed: int = 0, 
        snr: float = 100, sing_dist: str = "uniform", condition_number: float = 1e6, max_singular_value: float = 1.0, 
        normalize_features: bool = False,

    ):
        super(LinearRegressionSingDataset, self).__init__()
        #problem dimensionality
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.N_samples = N_samples
        
        #distribution of noise and weights 
        self.noise_type = noise_type
        self.weight_init = weight_init

        self.snr = snr
        self.condition_number = condition_number
        self.sing_dist = sing_dist
        self.max_singular_value = max_singular_value
        self.condition_number = condition_number
        self.normalize_features = normalize_features
        
        #randomness should be controlled by a generator to not effect global random state
        self.seed = seed
        self.torch_gen = torch.Generator()
        self.torch_gen.manual_seed(self.seed)
        self.np_rng = np.random.RandomState(self.seed)

        self._generate_data()

    def _generate_data(self):
        self._generate_feature_matrix()
        self._generate_true_weights()
        self.Y = self.X @ self.W_true
        self._generate_noise()
        self.Y += self.noise

        if self.normalize_features:
            self.X = (self.X - self.X.mean(dim=0)) / (self.X.std(dim=0) + 1e-8)

        self.data = lambda: {
            "X": self.X,
            "Y": self.Y,
            "W_true": self.W_true,
        }


    
    def _generate_feature_matrix(self) -> Tensor:
        """
        generate a feature matrix with a specified condition number
        make sure that the covariance matrix is positive definite
        and the other singular values are well spread
        """
        N, d = self.N_samples, self.dim_input
        assert N > d, "can't have N <= d ... tall matrix needed"
        
        #basically generating X = U S V^T while explicitely controlling singular value decomposition

        U = torch.randn(N, d, generator=self.torch_gen)
        self.U, _ = torch.linalg.qr(U, 'reduced')

        Vh = torch.randn(d, d, generator=self.torch_gen)
        self.Vh, _ = torch.linalg.qr(Vh) 

        #svd with max singular value of 1.0 and guarantee certain condition number regardless of sing_dist
        max_singular_value = self.max_singular_value if hasattr(self, 'max_singular_value') else 1.0
        min_singular_value = max_singular_value / self.condition_number


        if self.sing_dist == "uniform":
            diag = min_singular_value + (max_singular_value - min_singular_value) * torch.rand(d, generator=self.torch_gen)
            min_idx = torch.randint(0, d, (1,), generator=self.torch_gen).item()
            max_idx = torch.randint(0, d, (1,), generator=self.torch_gen).item()
            while max_idx == min_idx:
                max_idx = torch.randint(0, d, (1,), generator=self.torch_gen).item()

            diag[min_idx] = min_singular_value
            diag[max_idx] = max_singular_value

        elif self.sing_dist == "normal":
            mean = (min_singular_value + max_singular_value) / 2
            std = (max_singular_value - min_singular_value) / 6

            diag = torch.normal(mean, std, size=(d,), generator=self.torch_gen)
            diag = torch.clamp(diag, min=min_singular_value, max=max_singular_value)
            min_idx = torch.randint(0, d, (1,), generator=self.torch_gen).item()
            max_idx = torch.randint(0, d, (1,), generator=self.torch_gen).item()
            while max_idx == min_idx:
                max_idx = torch.randint(0, d, (1,), generator=self.torch_gen).item()
            diag[min_idx] = min_singular_value
            diag[max_idx] = max_singular_value

        elif self.sing_dist == "lognormal":
            mean = torch.log(min_singular_value * max_singular_value)
            std = (torch.log(max_singular_value) - torch.log(min_singular_value)) / 6

            diag = torch.exp(torch.normal(mean, std, size=(d,), generator=self.torch_gen))
            diag = torch.clamp(diag, min=min_singular_value, max=max_singular_value)
            min_idx = torch.randint(0, d, (1,), generator=self.torch_gen).item()
            max_idx = torch.randint(0, d, (1,), generator=self.torch_gen).item()
            while max_idx == min_idx:
                max_idx = torch.randint(0, d, (1,), generator=self.torch_gen).item()
            diag[min_idx] = min_singular_value
            diag[max_idx] = max_singular_value
        else:
            raise ValueError(f"Unknown singular value distribution: {self.sing_dist}")
        
        #make an Nxd matrix with the singular values on the diagonal
        self.S = torch.zeros(d, d)
        index = torch.arange(d)
        self.S[index, index] = diag

        #return feature matrix with jacked up singular values
        self.X = self.U @ self.S @ self.Vh


    def _generate_true_weights(self):
        """
        Generate true weights for the linear regression model
        """
        if self.weight_init == "gaussian":
            self.W_true = torch.randn(self.dim_input, self.dim_output, generator=self.torch_gen)
        elif self.weight_init == "uniform":
            self.W_true = torch.rand(self.dim_input, self.dim_output, generator=self.torch_gen)
        elif self.weight_init == "laplace":
            self.W_true = torch.randn(self.dim_input, self.dim_output, generator=self.torch_gen)
            self.W_true = torch.sign(self.W_true) * torch.abs(self.W_true).sqrt()
        else:
            raise ValueError("weight_init must be one of gaussian, uniform, laplace")
        
    def _generate_noise(self) -> None:
        signal_std = torch.std(self.Y, dim=0)
        noise_std = signal_std / math.sqrt(self.snr)
        if self.noise_type == "gaussian":
            self.noise = torch.randn(self.Y.shape, generator=self.torch_gen) * noise_std
        elif self.noise_type == "uniform":
            self.noise = (torch.rand(self.Y.shape, generator=self.torch_gen) - 0.5) * 2 * noise_std
        elif self.noise_type == "laplace":
            self.noise = torch.sign(torch.randn(self.Y.shape, generator=self.torch_gen)) * \
                         noise_std * torch.log(torch.rand_like(self.Y, generator=self.torch_gen) + 1e-8)
        else:
            raise ValueError(f"Unknown noise type: {self.noise_type}")

    #probably useless but keeping for now
    def _edit_feature_matrix(self, new_condition_number: float, new_sing_dist: str = "uniform") -> None:
        """
        Edit the feature matrix to have a new condition number and singular value distribution
        """
        self.condition_number = new_condition_number
        self.sing_dist = new_sing_dist
        self._generate_feature_matrix(condition_number=self.condition_number, sing_dist=self.sing_dist)
  

    def __len__(self):
        return self.N_samples

    def __getitem__(self, index):
        return self.X[index], self.Y[index]
    

if __name__ == "__main__":
    dataset = LinearRegressionSingDataset(
        dim_input=100, 
        dim_output=15, 
        N_samples=1000, 
        noise_type="gaussian", 
        weight_init="gaussian", 
        seed=46,
    )
    


    X = dataset.X
    XTX = X.T @ X
    feat_inv = torch.linalg.pinv(XTX)

    #get the svd of the feature matrix
    U, S, Vh = torch.svd(X, some=True)

    #get max and min singular values
    max_singular_value = S.max().item()
    min_singular_value = S.min().item()
    print(f"Max singular value: {max_singular_value}")
    print(f"Min singular value: {min_singular_value}")
