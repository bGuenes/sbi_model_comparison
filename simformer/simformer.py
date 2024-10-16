import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

# --------------------------------------------------------------------------------------------------

class GaussianFourierEmbedding(nn.Module):
    """Gaussian Fourier embedding module. Mostly used to embed time.

    Args:
        output_dim (int, optional): Output dimesion. Defaults to 128.
    """
    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed 
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        term1 = torch.sin(x_proj)
        term2 = torch.cos(x_proj)
        out = torch.cat([term1, term2], dim=-1)
        return out


# --------------------------------------------------------------------------------------------------
# Stochastic Differential Equations

class BaseSDE():
      """A base class for SDEs. We assume that the SDE is of the form:

    dX_t = f(t, X_t)dt + g(t, X_t)dW_t

    where f and g are the drift and diffusion functions respectively. We assume that the initial distribution is given by p0 at time t=0.

    Args:
        drift (Callable): Drift function
        diffusion (Callable): Diffusion function
        p0 (Distribution): Initial distribution
    """
      def __init__(self, drift, diff):
          self.drift = drift
          self.diff = diff
      
      def diffusion(self, x, t):
        eps = torch.randn_like(x)
        return x + self.drift(t) + self.diff(t) * eps


class VPSDE(BaseSDE):
    def __init__(self, beta_min=0.01, beta_max=10.0):
        """
        Variance Preserving Stochastic Differential Equation (VPSDE) class.
        The VPSDE is defined as:
            Drift     -> f(x,t) = -1/2 * beta_t * x
            Diffusion -> g(t)   = sqrt(beta_t)
        """
        drift = lambda t: -0.5 * self.betas[t]
        diff = lambda t: torch.sqrt(self.betas[t])

        super().__init__(drift, diff)


class VESDE(BaseSDE):
    def __init__(self, sigma_min=0.0001, sigma_max=15.0):
        """
        Variance Exploding Stochastic Differential Equation (VESDE) class.
        The VESDE is defined as:
            Drift     -> f(x,t) = 0
            Diffusion -> g(t)   = sigma^t
        """
        drift = lambda t: torch.zeros(1)
        
        _const = torch.sqrt(2 * torch.log(torch.tensor([sigma_max / sigma_min])))
        diff = lambda t: sigma_min * (sigma_max / sigma_min) ** t * _const

        super().__init__(drift, diff)


# --------------------------------------------------------------------------------------------------

class Simformer(nn.Module):
    def __init__(self, timesteps, sde_type="vesde"):
        super(Simformer, self).__init__()

        self.betas = self.linear_beta_schedule(timesteps=timesteps)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0) # Cumulative product of alphas at each timestep

        self.sde_type = sde_type

        self.time_embedding = GaussianFourierEmbedding(64)

        if sde_type == "vesde":
            self.sde = VESDE()
        elif sde_type == "vpsde":
            self.sde = VPSDE()
        else:
            raise ValueError("Invalid SDE type")

    def linear_beta_schedule(self, timesteps, start=0.0001, end=0.02):
        return torch.linspace(start, end, timesteps)

    def forward_diffusion_sample(self, x_0, t, device="cpu"):
        """ 
        Takes an image and a timestep as input and 
        returns the noisy version of it
        """
        x_1 = self.sde.diffusion(x_0, t)
        return x_1
