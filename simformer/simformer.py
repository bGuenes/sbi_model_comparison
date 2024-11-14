import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Transformer

import numpy as np

# --------------------------------------------------------------------------------------------------

class GaussianFourierEmbedding(nn.Module):
    """Gaussian Fourier embedding module. Mostly used to embed time.

    Args:
        embed_dim (int, optional): Output dimesion. Defaults to 128.
    """
    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x):
        half_dim = self.embed_dim // 2 + 1
        B = torch.randn(half_dim, x.shape[-1])
        x = 2 * np.pi * torch.matmul(x, B.T)
        term1 = torch.cos(x)
        term2 = torch.sin(x)
        out = torch.cat([term1, term2], dim=-1)
        return out[..., : self.embed_dim]


# --------------------------------------------------------------------------------------------------
# Stochastic Differential Equations

class BaseSDE():
    """
    A base class for SDEs. We assume that the SDE is of the form:

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

    def marginal_stddev(self, t):
        # Marginal standard deviation is the square root of marginal variance
        return torch.sqrt(self.marginal_variance(t))
    

class VPSDE(BaseSDE):
    def __init__(self):
        """
        Variance Preserving Stochastic Differential Equation (VPSDE) class.
        The VPSDE is defined as:
            Drift     -> f(x,t) = -1/2 * beta_t * x
            Diffusion -> g(t)   = sqrt(beta_t)
        """
        drift = lambda t: -0.5 * self.betas[t]
        diff = lambda t: torch.sqrt(self.betas[t])

        super().__init__(drift, diff)

    def marginal_variance(self, t):
        # Using the formula for variance in VPSDE with constant beta
        # Variance for VP SDE: sigma^2 (1 - exp(-integral(beta(s))))
        return 1.0 - torch.exp(-self.betas(t) * t)


class VESDE(BaseSDE):
    def __init__(self, sigma_min=0.0001, sigma_max=15.0):
        """
        Variance Exploding Stochastic Differential Equation (VESDE) class.
        The VESDE is defined as:
            Drift     -> f(x,t) = 0
            Diffusion -> g(t)   = sigma^t
        """
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        drift = lambda t: torch.zeros(1)
        
        _const = torch.sqrt(2 * torch.log(torch.tensor([self.sigma_max / self.sigma_min])))
        diff = lambda t: self.sigma_min * (self.sigma_max / self.sigma_min) ** t * _const

        super().__init__(drift, diff)

    def marginal_variance(self, t):
        # For VE SDE, the variance grows with t^2 * sigma^2
        return (self.sigma_min * (self.sigma_max / self.sigma_min) ** t) ** 2


# --------------------------------------------------------------------------------------------------

class Simformer(nn.Module):
    def __init__(self, timesteps, data_shape, sde_type="vesde", dim_value=20, dim_id=20, dim_condition=10, dim_time=64):
        super(Simformer, self).__init__()

        self.betas = self.linear_beta_schedule(timesteps=timesteps)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0) # Cumulative product of alphas at each timestep

        # initialize SDE
        if sde_type == "vesde":
            self.sde = VESDE()
        elif sde_type == "vpsde":
            self.sde = VPSDE()
        else:
            raise ValueError("Invalid SDE type")

        # Gaussian Fourier Embedding for time embedding
        self.time_embedding = GaussianFourierEmbedding(dim_time)

        # Token embedding layers for values, node IDs, and condition masks
        self.nodes_max = data_shape[1]  # Maximum number of nodes
        self.node_ids = torch.arange(self.nodes_max)  # Node IDs
        self.embedding_net_value = lambda x: np.repeat(x, dim_value, axis=-1)  # Value embedding net (here we just repeat the value)
        self.embedding_net_id = nn.Embedding(self.nodes_max, dim_id)  # Embedding for node IDs
        self.condition_embedding = nn.Parameter(torch.randn(1, 1, dim_condition) * 0.5)  # Learnable condition embedding

        # Transformer model
        self.transformer = Transformer(d_model=dim_value + dim_id + dim_condition + dim_time, nhead=2, num_encoder_layers=2, num_decoder_layers=2)

        # Output linear layer
        self.output_layer = nn.Linear(dim_value + dim_id + dim_condition + dim_time, 1)


    def linear_beta_schedule(self, timesteps, start=0.0001, end=0.02):
        return torch.linspace(start, end, timesteps)

    def forward_diffusion_sample(self, x_0, t, device="cpu"):
        # Diffusion process for time t with defined SDE
        x_1 = self.sde.diffusion(x_0, t)
        return x_1
    
    def output_scale_function(self, t, x):
        scale = torch.clamp(self.sde.marginal_stddev(t), min=1e-2)
        return (x/scale).reshape(x.shape)
    
    # ------------------------------------
    # /////////// Forward pass ///////////    

    def forward(self, x, timestep, condition_mask, edge_mask=None):

        # --- Reshape input ---
        # shaping data in the form of (batch_size, sequence_length, values)
        batch_size, seq_len = x.shape
        x = x.reshape(batch_size, seq_len, 1)
        condition_mask = condition_mask.reshape(x.shape)
        batch_node_ids = torch.tensor(np.repeat([self.node_ids], batch_size, axis=0))

        # --- Embedding ---
        # Time embedding
        time_embedded = self.time_embedding(timestep).unsqueeze(1).expand(-1, seq_len, -1)
        # Value embedding
        value_embedded = self.embedding_net_value(x)
        # Node ID embedding
        id_embedded = self.embedding_net_id(batch_node_ids)
        # Condition embedding
        condition_embedded = self.condition_embedding * condition_mask
        
        # --- Create Token ---
        # Concatenate all embeddings to create the input for the Transformer
        x_encoded = torch.cat([value_embedded, id_embedded, condition_embedded, time_embedded], dim=-1)
        x_encoded = x_encoded.permute(1, 0, 2)  # Transformer expects (seq_len, batch_size, d_model)

        # --- Transformer ---
        # Transformer forward pass
        transformer_output = self.transformer(x_encoded, x_encoded, src_key_padding_mask=edge_mask)
        transformer_output = transformer_output.permute(1, 0, 2)  # Reorder back to (batch_size, seq_len, d_model)

        # --- Output decoding ---
        # Score estimate output layer
        out = self.output_layer(transformer_output)
        out = self.output_scale_function(timestep, out)

        return out
    
    # ------------------------------------

    

