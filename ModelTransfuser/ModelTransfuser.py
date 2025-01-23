import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Transformer
from torch.optim.lr_scheduler import ReduceLROnPlateau

import numpy as np

import pickle
import sys
import tqdm
import time

# --------------------------------------------------------------------------------------------------

class GaussianFourierEmbedding(nn.Module):
    """Gaussian Fourier embedding module. Mostly used to embed time.

    Args:
        embed_dim (int, optional): Output dimesion. Defaults to 64.
    """
    def __init__(self, embed_dim=64, scale=30.):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x):
        half_dim = self.embed_dim // 2 + 1
        B = torch.randn(half_dim, x.shape[-1], device=x.device)
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

class ModelTransfuser(nn.Module):
    # ------------------------------------
    # /////////// Initialization ///////////
    # Initialize the ModelTransfuser model

    def __init__(self, timesteps, data_shape, sde_type="vesde", sigma=25.0, dim_value=20, dim_id=20, dim_condition=20, dim_time=64):
        super(ModelTransfuser, self).__init__()

        # Time steps in the diffusion process
        self.timesteps = timesteps
        self.t = torch.linspace(0, 1, self.timesteps)
        self.sigma = sigma

        # initialize SDE
        if sde_type == "vesde":
            self.sde = VESDE(sigma=self.sigma)
        elif sde_type == "vpsde":
            self.sde = VPSDE()
        else:
            raise ValueError("Invalid SDE type")

        # Gaussian Fourier Embedding for time embedding
        self.time_embedding = GaussianFourierEmbedding(dim_time)

        # Token embedding layers for values, node IDs, and condition masks
        self.nodes_max = data_shape[1]  # Maximum number of nodes
        self.node_ids = torch.arange(self.nodes_max)  # Node IDs
        self.dim_value = dim_value
        self.embedding_net_id = nn.Embedding(self.nodes_max, dim_id)  # Embedding for node IDs
        self.condition_embedding = nn.Parameter(torch.randn(1, 1, dim_condition) * 0.5)  # Learnable condition embedding

        # Transformer model
        self.transformer = Transformer(d_model=dim_value + dim_id + dim_condition + dim_time, nhead=2, num_encoder_layers=2, num_decoder_layers=2, batch_first=True)

        # Output linear layer
        self.output_layer = nn.Linear(dim_value + dim_id + dim_condition + dim_time, 1)

    # ------------------------------------
    # /////////// Helper functions ///////////

    def forward_diffusion_sample(self, x_0, t, x_1=None):
        # Diffusion process for time t with defined SDE
        if x_1 is None:
            x_1 = torch.randn_like(x_0)

        std = self.sde.marginal_prob_std(t).reshape(-1, 1).to(x_0.device)
        x_t = x_0 + std * x_1
        return x_t
    
    def embedding_net_value(self, x):
        # Value embedding net (here we just repeat the value)
        dim_value=self.dim_value
        return x.repeat(1,1,dim_value)
    
    def set_normalization(self, data):
        """
        Compute and set normalization parameters (mean and std) from the dataset.
        """
        self.mean = data.mean(dim=0, keepdim=True).to(data.device)
        self.std = data.std(dim=0, keepdim=True).to(data.device)
        print(f"Normalization parameters set.")

    def normalize(self, x):
        """
        Normalize input data using the stored mean and std.
        """
        try:
            return (x - self.mean) / (self.std + 1e-6)  # Add epsilon to avoid division by zero
        except:
            raise ValueError("Normalization parameters are not set. Use `set_normalization` first.")
    
    def output_scale_function(self, t, x):
        scale = self.sde.marginal_prob_std(t).to(x.device)
        return x / scale.unsqueeze(1)
    
    # ------------------------------------
    # /////////// Forward pass ///////////
    # Predict the score for the given input data 


    def forward_transformer(self, x, timestep, condition_mask, edge_mask=None):

        # --- Reshape input ---
        # shaping data in the form of (batch_size, sequence_length, values)
        batch_size, seq_len = x.shape
        x = x.reshape(batch_size, seq_len, 1)
        condition_mask = condition_mask.reshape(x.shape).to(x.device)
        batch_node_ids = torch.tensor(np.repeat([self.node_ids], batch_size, axis=0)).to(x.device)

        # --- Embedding ---
        # Time embedding
        time_embedded = self.time_embedding(timestep).unsqueeze(1).expand(batch_size, seq_len, -1)
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
        transformer_output = self.transformer(x_encoded, x_encoded)
        transformer_output = transformer_output.permute(1, 0, 2)  # Reorder back to (batch_size, seq_len, d_model)

        # --- Output decoding ---
        # Score estimate output layer
        out = self.output_layer(transformer_output)

        return out
    
    # ------------------------------------
    # /////////// Loss function ///////////

    def loss_fn(self, score, timestep, x_1, condition_mask):
        '''
        Loss function for the score prediction task

        Args:
            score: Predicted score
                    
        The target is the noise added to the data at a specific timestep 
        Meaning the prediction is the approximation of the noise added to the data
        '''
        sigma_t = self.sde.marginal_prob_std(timestep).unsqueeze(1).to(score.device)
        x_1 = x_1.unsqueeze(2).to(score.device)
        condition_mask = condition_mask.unsqueeze(2).to(score.device)

        loss = torch.mean(sigma_t**2 * torch.sum((1-condition_mask)*(x_1-sigma_t*score)**2))

        return loss
    
    # ------------------------------------
    # /////////// Training ///////////

    def train(self, data, condition_mask_data=None, batch_size=64, epochs=10, lr=1e-3, device="cpu", val_data=None, condition_mask_val=None):
        start_time = time.time()

        self.to(device)

        # Define the optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=0.1)
        scheduler = ReduceLROnPlateau(optimizer, patience=5, threshold=500)
        self.train_loss = []
        self.val_loss = []

        if condition_mask_data is None:
        # If no condition mask is provided, eg to just train posterior or likelihood
        # the condition mask is sampled randomly over all data points to be able to predict the likelihood or posterior and also be able to work with missing data
            condition_mask_random_data = torch.distributions.bernoulli.Bernoulli(torch.ones_like(data) * 0.33)

        if condition_mask_val is None and val_data is not None:
            condition_mask_random_val = torch.distributions.bernoulli.Bernoulli(torch.ones_like(val_data) * 0.33)

        # Normalize data
        data_normed = self.normalize(data)

        # Training loop
        for epoch in range(epochs):
            idx = torch.randperm(data_normed.nelement())
            data_normed_shuffled = data_normed.view(-1)[idx].view(data_normed.size())

            loss_epoch = 0

            if condition_mask_data is None:
                condition_mask_data = condition_mask_random_data.sample()

            for i in tqdm.tqdm(range(0, data_normed_shuffled.shape[0], batch_size), desc=f'Epoch {epoch+1:{""}{2}}/{epochs}: '):
            #for i in tqdm.tqdm(range(0, 100, batch_size), desc=f'Epoch {epoch+1:{""}{2}}/{epochs}: '):
                optimizer.zero_grad()

                x_0 = data_normed_shuffled[i:i+batch_size].to(device)
                condition_mask_batch = condition_mask_data[i:i+batch_size].to(device)

                # Pick random timesteps in diffusion process
                index_t = torch.randint(0, self.timesteps, (x_0.shape[0],))
                timestep = self.t[index_t].reshape(-1, 1).to(device)

                x_1 = torch.randn_like(x_0)

                x_t = self.forward_diffusion_sample(x_0, timestep, x_1)

                score = self.forward_transformer(x_t, timestep, condition_mask_batch)
                loss = self.loss_fn(score, timestep, x_1, condition_mask_batch)
                loss_epoch += loss.item()

                loss.backward()
                optimizer.step()
                
            
            self.train_loss.append(loss_epoch)
            #torch.cuda.empty_cache()

            # Validation set if provided
            if val_data is not None:
                val_data_normed = self.normalize(val_data)

                if condition_mask_val is None:
                    # If no condition mask is provided, val on random condition mask
                    condition_mask_val = condition_mask_random_val.sample()

                batch_size_val = 1000
                val_loss = 0
                for i in range(0, val_data_normed.shape[0], batch_size_val):
                    x_0 = val_data_normed[i:i+batch_size_val].to(device)
                    condition_mask_val_batch = condition_mask_val[i:i+batch_size_val].to(device)
                    index_t = torch.randint(0, self.timesteps, (x_0.shape[0],))

                    timestep = self.t[index_t].reshape(-1, 1).to(device)
                    noise = torch.randn_like(x_0)

                    x_1 = self.forward_diffusion_sample(x_0, timestep, noise)
                    score = self.forward_transformer(x_1, timestep, condition_mask_val_batch)
                    val_loss += self.loss_fn(score, timestep, noise, condition_mask_val_batch).item()
                    
                self.val_loss.append(val_loss)
                scheduler.step(val_loss)
                #print(scheduler.get_last_lr())
                #torch.cuda.empty_cache()

                print(f'--- Training Loss: {loss_epoch:{""}{11}.3f} --- Validation Loss: {val_loss:{""}{11}.3f} ---')
                print()

            else:
                scheduler.step(loss_epoch)
                print(f'--- Training Loss: {loss_epoch:{""}{11}.3f} ---')
                print()
        
        end_time = time.time()
        time_elapsed = (end_time - start_time) / 60
        print(f"Training finished after {time_elapsed:.1f} minutes")

    # ------------------------------------
    # /////////// Sample ///////////

    def sample(self, data, condition_mask, num_samples=1_000, device="cpu"):

        # Shaping
        # Correct data shape is [Number of unique samples, Number of predicted samples for each unique sample, Number of values]
        if len(data.shape) == 1:
            data = data.unsqueeze(0)
        if len(condition_mask.shape) == 1:
            condition_mask = condition_mask.unsqueeze(0)
        
        self.to(device)

        # Normalize data
        data_normed = self.normalize(data)

        #x = data_normed.repeat(num_samples, 1)
        x = data_normed.unsqueeze(1).repeat(1,num_samples,1).to(device)
        condition_mask_samples = condition_mask.unsqueeze(1).repeat(1,num_samples,1).to(device)
        dt = (1/self.timesteps)
        
        for n in tqdm.tqdm(range(len(data))):

            for t in reversed(self.t):
                timestep = t.reshape(-1, 1).to(device)
                
                score = self.forward_transformer(x[n,:], timestep, condition_mask_samples[n,:]).squeeze(-1).detach()
                dx = 1/2 * self.sigma**(2*timestep)* score * dt

                x[n,:] -= dx * (1-condition_mask_samples[n,:])
            
        # Denormalize data
        x = x * (self.std.to(device) + 1e-6) + self.mean.to(device)

        return x.detach()
    
    # ------------------------------------
    # /////////// Save & Load ///////////

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            model = pickle.load(f)
        return model