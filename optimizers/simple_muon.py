import torch
from torch import Tensor

def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
    """
    Newton-Schulz iteration for orthogonalization (CPU-compatible).
    """
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.float()  # Use float32 for CPU compatibility
    
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

class SimpleMuon(torch.optim.Optimizer):
    """
    Simplified Muon optimizer for CPU/local training.
    """
    def __init__(
            self, params, lr=0.02, weight_decay=0.01, momentum=0.95, nesterov=True, ns_steps=5, zero_init=True
        ):
        assert lr > 0, "Learning rate must be positive"
        assert 0 <= weight_decay, "Weight decay must be non-negative"
        assert 0 <= momentum < 1, "Momentum must be in [0, 1)"
        assert ns_steps > 0 and isinstance(ns_steps, int), "Number of steps must be positive"
        defaults = dict(
            lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps, zero_init=zero_init,
            )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]
                
                if 'momentum_buffer' not in state:
                    if group['zero_init']:
                        state['momentum_buffer'] = torch.zeros_like(p)
                    else:
                        state['momentum_buffer'] = grad.clone()
                
                buf = state['momentum_buffer']
                
                #weight decay
                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])
                
                #momentum update
                buf.mul_(group['momentum']).add_(grad, alpha=1 - group['momentum'])
                
                # Nesterov momentum
                if group['nesterov']:
                    grad = grad.add(buf, alpha=group['momentum'])
                else:
                    grad = buf
                
                # Apply orthogonalization for 2D+ parameters
                if grad.dim() >= 2:
                    original_shape = grad.shape
                    if grad.dim() > 2:  # Flatten conv filters
                        grad = grad.view(original_shape[0], -1)
                    
                    grad = zeropower_via_newtonschulz5(grad, steps=group['ns_steps'])
                    
                    if original_shape != grad.shape:  # Restore shape
                        grad = grad.view(original_shape)
                
                # Update parameters
                p.add_(grad, alpha=-group['lr'])