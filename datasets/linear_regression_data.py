import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.nn import Parameter
from torch import Tensor
from typing import Tuple, Dict, Any, Optional


class LinearRegressionDataset(Dataset):
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
        covariance_type: Type of covariance matrix (identity, diagonal, full, low_rank, toeplitz)
        covariance_strength: Strength of covariance matrix (0.0 to 1.0)   
        normalize_features: normalize to make sure on same scale (common practice)
    """

    def __init__(
        self, dim_input: int, dim_output: int, N_samples: int, 
        noise_type: str = "gaussian", noise_level: float = 0.01,
        weight_init: str = "gaussian", seed: int = 0, 
        covariance_type: Optional[str] = None, covariance_strength: Optional[float] = None, 
        normalize_features: bool = False,
    ):
        super(LinearRegressionDataset, self).__init__()
        if covariance_strength is None:
            assert covariance_type is None, "covariance_strength must be specified if covariance_type is specified"

        self.dim_input = dim_input
        self.dim_output = dim_output
        self.noise_type = noise_type
        self.noise_level = noise_level
        self.weight_init = weight_init
        self.normalize_features = normalize_features
        self.seed = seed
        self.N_samples = N_samples
        self.covariance_type = covariance_type
        self.covariance_strength = covariance_strength
        
        # Create local random generators instead of modifying global state
        self.torch_gen = torch.Generator()
        self.torch_gen.manual_seed(seed)
        self.np_rng = np.random.RandomState(seed)
        
        self._generate_data()

    def _generate_covariance_matrix(self) -> Tensor:
        d = self.dim_input
        strength = self.covariance_strength
        cov_type = self.covariance_type

        if cov_type == "identity":
            return torch.eye(d)
        elif cov_type == "diagonal":
            diag = 1.0 + strength * torch.rand(d, generator=self.torch_gen)
            return torch.diag(diag)
        elif cov_type == "full":
            A = torch.randn(d, d, generator=self.torch_gen)
            cov = A @ A.T
            cov = cov / cov.norm() * d * strength  # normalize scale
            return cov
        elif cov_type == "low_rank":
            rank = max(1, d // 4)
            A = torch.randn(d, rank, generator=self.torch_gen)
            cov = A @ A.T + strength * torch.eye(d)
            return cov
        elif cov_type == "toeplitz":
            r = strength ** torch.arange(d).float()
            cov = torch.zeros(d, d)
            for i in range(d):
                cov[i] = r.roll(i)
            return cov
        else:
            raise ValueError(f"Unknown covariance_type: {cov_type}")

  



    def _generate_data(self) -> None:
        # add covariance if specified
        if self.covariance_type is not None:
            cov = self._generate_covariance_matrix()
            mean = torch.randn(self.dim_input, generator=self.torch_gen)
            mvn = torch.distributions.MultivariateNormal(mean, covariance_matrix=cov)
            # For MultivariateNormal, we can't directly specify the generator
            # But it will use the global state which we won't modify
            X_features = mvn.sample((self.N_samples,))
        else:
            X_features = torch.randn(self.N_samples, self.dim_input, generator=self.torch_gen)

        if self.normalize_features:
            X_features = (X_features - X_features.mean(dim=0)) / X_features.std(dim=0)

        self.X = torch.cat((torch.ones(self.N_samples, 1), X_features), dim=1)

        # Initialize weights
        if self.weight_init == "gaussian":
            self.W_true = torch.randn(self.dim_input, self.dim_output, generator=self.torch_gen)
        elif self.weight_init == "uniform":
            self.W_true = torch.rand(self.dim_input, self.dim_output, generator=self.torch_gen)
        elif self.weight_init == "laplace":
            self.W_true = torch.randn(self.dim_input, self.dim_output, generator=self.torch_gen)
            self.W_true = torch.sign(self.W_true) * torch.abs(self.W_true).sqrt()
        else:
            raise ValueError("weight_init must be one of gaussian, uniform, laplace")

        self.noise = self.generate_noise(self.N_samples)
        self.Y = self.X @ self.W_true + self.noise

        self.data = lambda: {
            "X": self.X,
            "Y": self.Y,
            "W_true": self.W_true,
        }

    def generate_noise(self, N):
        if self.noise_type == "gaussian":
            noise = torch.randn(N, self.dim_output, generator=self.torch_gen) * self.noise_level
        elif self.noise_type == "uniform":
            noise = torch.rand(N, self.dim_output, generator=self.torch_gen) * self.noise_level
        elif self.noise_type == "laplace":
            noise = torch.randn(N, self.dim_output, generator=self.torch_gen) * self.noise_level
            noise = torch.sign(noise) * torch.abs(noise).sqrt()
        else:
            raise ValueError("noise_type must be one of gaussian, uniform, laplace")
        return noise

    def __len__(self):
        return self.N_samples

    def __getitem__(self, index):
        return self.X[index], self.Y[index]