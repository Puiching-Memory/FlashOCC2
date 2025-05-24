import torch
import torch.nn as nn
import torch.nn.functional as F

class AGLU(nn.Module):
    """Unified activation function module from https://github.com/kostas1515/AGLU"""

    def __init__(self, device=None, dtype=None) -> None:
        """Initialize the Unified activation function."""
        super().__init__()
        self.act = nn.Softplus(beta=-1.0)
        self.lambd = nn.Parameter(nn.init.uniform_(torch.empty(1, device=device, dtype=dtype)))  # lambda parameter
        self.kappa = nn.Parameter(nn.init.uniform_(torch.empty(1, device=device, dtype=dtype)))  # kappa parameter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the forward pass of the Unified activation function."""
        lam = torch.clamp(self.lambd, min=0.0001)
        return torch.exp((1 / lam) * self.act((self.kappa * x) - torch.log(lam)))
    
class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        dim,
        hidden_dim,
        multiple_of,
        ffn_dim_multiplier=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False, **factory_kwargs)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False, **factory_kwargs)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False, **factory_kwargs)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))