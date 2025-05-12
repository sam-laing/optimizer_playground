import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.nn import Parameter
from torch import Tensor
from typing import Tuple, Dict, Any

class LinearRegressionDataset(Dataset):
    def __init__(
        self, dim_input: int, dim_output: int, N_samples: int, 
        noise_type: str = "gaussian", noise_level: float = 0.01,
        weight_init: str = "gaussian", seed: int = 0
    ):
        """  
        mock up artificial data which is ideal for a linear regression problem 
        Y = XW + noise 
        - X is a matrix of shape (N, dim_input + 1) with first column all ones
        - W is a matrix of shape (dim_input + 1, dim_output)
        - Y is a matrix of shape (N, dim_output)
        controlling noise level and type and weight initialization 
        """
        super(LinearRegressionDataset, self).__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.noise_type = noise_type
        self.noise_level = noise_level
        self.weight_init = weight_init
        self.seed = seed
        self.N_samples = N_samples            

        self._set_seed(seed)
        self._generate_data()    
    
    def _set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
    
    def _generate_data(self) -> None:
        self._set_seed(self.seed)
        self.X = torch.randn(self.N_samples, self.dim_input)
        self.X = torch.cat((torch.ones(self.N_samples, 1), self.X), dim=1)

        #noisy observations assumed in Linear Regression
        #allowing for different noise types which isn't standard
        if self.weight_init == "gaussian":
            self.W_true = torch.randn(self.dim_input + 1, self.dim_output)
        elif self.weight_init == "uniform":
            self.W_true = torch.rand(self.dim_input + 1, self.dim_output)
        elif self.weight_init == "laplace":
            self.W_true = torch.randn(self.dim_input + 1, self.dim_output)
            self.W_true = torch.sign(self.W) * torch.abs(self.W) ** (1/2)
        else:
            raise ValueError("weight_init must be one of gaussian, uniform, laplace")
        self.noise = self.generate_noise(self.N_samples)

        self.Y = self.X @ self.W_true + self.noise

        self.data = {
            "X": self.X,
            "Y": self.Y,
            "W_true": self.W_true,
        }

    def generate_noise(self, N):
        if self.noise_type == "gaussian":
            noise = torch.randn(N, self.dim_output) * self.noise_level
        elif self.noise_type == "uniform":
            noise = torch.rand(N, self.dim_output) * self.noise_level
        elif self.noise_type == "laplace":
            noise = torch.randn(N, self.dim_output) * self.noise_level
            noise = torch.sign(noise) * torch.abs(noise) ** (1/2)
        else:
            raise ValueError("noise_type must be one of gaussian, uniform, laplace") 
        return noise
    
    def __len__(self):
        return self.N_samples
    
    def __getitem__(self, index):
        return self.X[index], self.Y[index]

        