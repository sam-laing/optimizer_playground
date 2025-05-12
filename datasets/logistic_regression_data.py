import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.nn import Parameter
from torch import Tensor
from typing import Tuple, Dict, Any


class LogisticRegressionDataset(Dataset):
    """     
    Very similar to LinearRegressionDataset but for a classification problem
    using a softmax activation function to generate labels 
    Y = softmax(XW) + noise
    - X is a matrix of shape (N, dim_input + 1) with first column all ones
    - W is a matrix of shape (dim_input + 1, dim_output)
    - Y is a matrix of shape (N, dim_output) with values in [0,1]
    - noise is a matrix of shape (N, dim_output) with values in [0,1]
    - W is initialized using xavier uniform initialization

    """
    def __init__(
        self, dim_input: int, num_classes: int, N_samples: int, 
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
        super(LogisticRegressionDataset, self).__init__()
        self.dim_input = dim_input
        self.num_classes = num_classes
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

        # Step 1: Generate X with bias term
        self.X = torch.randn(self.N_samples, self.dim_input)
        self.X = torch.cat((torch.ones(self.N_samples, 1), self.X), dim=1)  # (N, dim_input + 1)

        # Step 2: Initialize W_true
        if self.weight_init == "gaussian":
            self.W_true = torch.randn(self.dim_input + 1, self.num_classes)
        elif self.weight_init == "uniform":
            self.W_true = torch.rand(self.dim_input + 1, self.num_classes)
        elif self.weight_init == "laplace":
            laplace_noise = torch.distributions.laplace.Laplace(0.0, 1.0)
            self.W_true = laplace_noise.sample((self.dim_input + 1, self.num_classes))
        else:
            raise ValueError("weight_init must be one of 'gaussian', 'uniform', 'laplace'")

        # Step 3: Generate logits and add noise
        logits = self.X @ self.W_true                     # (N, K)
        noise = self._generate_noise()                    # (N, K)
        noisy_logits = logits + noise                     # noisy class scores

        # Step 4: Compute softmax probabilities
        probabilities = F.softmax(noisy_logits, dim=1)    # (N, K), in [0, 1], sum to 1

        # Step 5: Sample class labels from categorical distribution
        class_labels = torch.multinomial(probabilities, num_samples=1).squeeze(1)  # (N,)
        class_labels_onehot = F.one_hot(class_labels, num_classes=self.num_classes).float()

        # Step 6: Store the data
        self.data = {
            "X": self.X,
            "Y": class_labels_onehot,
            "W_true": self.W_true,
        }
    def _generate_noise(self):
        if self.noise_type == "gaussian":
            noise = torch.randn(self.N_samples, self.num_classes) * self.noise_level
        elif self.noise_type == "uniform":
            noise = torch.rand(self.N_samples, self.num_classes) * self.noise_level
        elif self.noise_type == "laplace":
            laplace_noise = torch.distributions.laplace.Laplace(0.0, 1.0)
            noise = laplace_noise.sample((self.N_samples, self.num_classes)) * self.noise_level
        else:
            raise ValueError("noise_type must be one of 'gaussian', 'uniform', 'laplace'") 
        return noise
    
    def __len__(self):
        return self.N_samples
    
    def __getitem__(self, index):
        return self.X[index], self.data["Y"][index]
    