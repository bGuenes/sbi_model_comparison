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

class VESDE():
    def __init__(self, sigma=25.0):
        """
        Variance Exploding Stochastic Differential Equation (VESDE) class.
        The VESDE is defined as:
            Drift     -> f(x,t) = 0
            Diffusion -> g(t)   = sigma^t
        """
        self.sigma = sigma

    def marginal_prob_std(self, t):
        """
        Compute the standard deviation of p_{0t}(x(t) | x(0)) for VESDE.

        Args:
            t: A tensor of time steps.
        Returns:
            The standard deviation.
        """
        return torch.sqrt((self.sigma ** (2 * t) - 1.0) / (2 * np.log(self.sigma)))


class VPSDE():
    def __init__(self):
        raise NotImplementedError("VPSDE is not implemented yet.")
    

# --------------------------------------------------------------------------------------------------

class Simformer(nn.Module):
    # ------------------------------------
    # /////////// Initialization ///////////
    # Initialize the Simformer model

    def __init__(self, timesteps, data_shape, sde_type="vesde", dim_value=20, dim_id=20, dim_condition=10, dim_time=64):
        super(Simformer, self).__init__()

        # Time steps in the diffusion process
        self.timesteps = timesteps
        self.t = torch.linspace(0, 1, self.timesteps)

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

    # ------------------------------------
    # /////////// Helper functions ///////////

    def forward_diffusion_sample(self, x_0, t, noise=None):
        # Diffusion process for time t with defined SDE
        if noise is None:
            noise = torch.randn_like(x_0)

        std = self.sde.marginal_prob_std(t).reshape(-1, 1)
        x_1 = x_0 + std * noise
        return x_1
    
    def output_scale_function(self, t, x):
        scale = self.sde.marginal_prob_std(t)
        return x / scale.unsqueeze(1)
    
    # ------------------------------------
    # /////////// Forward pass ///////////
    # Predict the score for the given input data 

    def forward_transformer(self, x, timestep, condition_mask, edge_mask=None):

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

        # --- Normalize output ---
        score = self.output_scale_function(timestep, out)

        return score
    
    # ------------------------------------
    # /////////// Loss function ///////////

    def loss_fn(self, pred, timestep, noise):
        '''
        Loss function for the score prediction task

        Args:
            pred: Predicted score
            target: Target score
                    
        The target is the noise added to the data at a specific timestep 
        Meaning the prediction is the approximation of the noise added to the data
        '''
        std = self.sde.marginal_prob_std(timestep).unsqueeze(1)
        noise = noise.unsqueeze(1)
        loss = torch.mean(torch.sum((pred*std + noise)**2))

        return loss
    
    # ------------------------------------
    # /////////// Training ///////////

    def train(self, data, condition_mask, batch_size=64, epochs=10, lr=1e-3):
        # Define the optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        # Training loop
        for epoch in range(epochs):
            
            loss_epoch = 0

            for i in range(0, data.shape[0], batch_size):
                optimizer.zero_grad()

                x_0 = data[i:i+batch_size]

                # Pick random timesteps in diffusion process
                index_t = torch.randint(0, self.timesteps, (batch_size,))
                timestep = self.t[index_t].reshape(-1, 1)

                noise = torch.randn_like(x_0)

                x_1 = self.forward_diffusion_sample(x_0, timestep, noise)

                score = self.forward_transformer(x_1, timestep, condition_mask[i:i+batch_size])
                loss = self.loss_fn(score, timestep, noise)
                loss_epoch += loss.item()

                loss.backward()
                optimizer.step()

            print(f"Epoch: {epoch}, Loss: {loss_epoch:.4f}")
        
    # ------------------------------------
