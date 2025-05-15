import torch 
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from typing import Dict, Optional
from torch import Tensor

class LogisticRegressionDataset(Dataset):
    """     
    Very similar to LinearRegressionDataset but for a classification problem
    using a softmax activation function to generate labels 
    Y = softmax(XW) + noise
    
    Also supports generating features with covariance structure.
    """
    def __init__(
        self, dim_input: int, num_classes: int, N_samples: int, 
        noise_type: str = "gaussian", noise_level: float = 0.01,
        weight_init: str = "gaussian", seed: int = 0,
        covariance_type: Optional[str] = None, covariance_strength: Optional[float] = None,
        normalize_features: bool = False
    ):
        super(LogisticRegressionDataset, self).__init__()
        if covariance_strength is None:
            assert covariance_type is None, "covariance_strength must be specified if covariance_type is specified"
            
        self.dim_input = dim_input
        self.num_classes = num_classes
        self.noise_type = noise_type
        self.noise_level = noise_level
        self.weight_init = weight_init
        self.seed = seed
        self.N_samples = N_samples
        self.covariance_type = covariance_type
        self.covariance_strength = covariance_strength
        self.normalize_features = normalize_features

        # local RNG... avoiding changing global state
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
        if self.covariance_type is not None:
            cov = self._generate_covariance_matrix()
            mean = torch.randn(self.dim_input, generator=self.torch_gen) 
            mvn = torch.distributions.MultivariateNormal(mean, covariance_matrix=cov)
            X_features = mvn.sample((self.N_samples,))
        else:
            X_features = torch.randn(self.N_samples, self.dim_input, generator=self.torch_gen)
            
        if self.normalize_features:
            X_features = (X_features - X_features.mean(dim=0)) / X_features.std(dim=0)
            
        ones = torch.ones(self.N_samples, 1)
        self.X = torch.cat((ones, X_features), dim=1)  # (N, dim_input + 1)

        if self.weight_init == "gaussian":
            self.W_true = torch.randn(self.dim_input + 1, self.num_classes, generator=self.torch_gen)
        elif self.weight_init == "uniform":
            self.W_true = torch.rand(self.dim_input + 1, self.num_classes, generator=self.torch_gen)
        elif self.weight_init == "laplace":
            # Create Laplace samples using our generator indirectly
            u = torch.rand(self.dim_input + 1, self.num_classes, generator=self.torch_gen)
            self.W_true = torch.sign(u - 0.5) * -torch.log(1 - 2 * torch.abs(u - 0.5))
        else:
            raise ValueError("weight_init must be one of 'gaussian', 'uniform', 'laplace'")

        logits = self.X @ self.W_true
        noise = self._generate_noise()
        noisy_logits = logits + noise

        probabilities = F.softmax(noisy_logits, dim=1)
        class_labels = torch.multinomial(probabilities, num_samples=1, 
                                         generator=self.torch_gen).squeeze(1)
        class_labels_onehot = F.one_hot(class_labels, num_classes=self.num_classes).float()

        self.data = {
            "X": self.X,
            "Y": class_labels_onehot,
            "W_true": self.W_true,
        }
        
    def _generate_noise(self):
        if self.noise_type == "gaussian":
            noise = torch.randn(self.N_samples, self.num_classes, generator=self.torch_gen)
            return noise * self.noise_level
        elif self.noise_type == "uniform":
            noise = torch.rand(self.N_samples, self.num_classes, generator=self.torch_gen)
            return noise * self.noise_level
        elif self.noise_type == "laplace":
            u = torch.rand(self.N_samples, self.num_classes, generator=self.torch_gen)
            noise = torch.sign(u - 0.5) * -torch.log(1 - 2 * torch.abs(u - 0.5))
            return noise * self.noise_level
        else:
            raise ValueError("noise_type must be one of 'gaussian', 'uniform', 'laplace'")
    
    def __len__(self):
        return self.N_samples
    
    def __getitem__(self, index):
        return self.X[index], self.data["Y"][index]