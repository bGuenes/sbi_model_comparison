import torch
import torch.nn as nn

import schedulefree

import numpy as np

import pickle
import sys
import tqdm
import time

from src.ConditionTransformer import DiT
from src.Simformer import Transformer
from src.sde import VESDE, VPSDE

####################################################################################################################

class ModelTransfuser(nn.Module):
    ########################################
    # /////////// Initialization ///////////

    def __init__(
            self,
            nodes_size, 
            sde_type="vesde",
            sigma=25.0,
            hidden_size=128,
            depth=6,
            num_heads=16,
            mlp_ratio=4,
            ):
        
        super(ModelTransfuser, self).__init__()

        self.nodes_size = nodes_size

        # initialize SDE
        self.sigma = sigma
        if sde_type == "vesde":
            self.sde = VESDE(sigma=self.sigma)
        elif sde_type == "vpsde":
            self.sde = VPSDE()
        else:
            raise ValueError("Invalid SDE type")

        # define model
        self.model = DiT(nodes_size=self.nodes_size, hidden_size=hidden_size, 
                                       depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio)

        # self.model = Transformer(nodes_size=self.nodes_size, hidden_size=hidden_size,
        #                          depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio)
        
    ##########################################
    # /////////// Helper functions ///////////

    def forward_diffusion_sample(self, x_0, t, x_1=None, condition_mask=None):
        # Diffusion process for time t with defined SDE
        
        if condition_mask is None:
            condition_mask = torch.zeros_like(x_0)

        if x_1 is None:
            x_1 = torch.randn_like(x_0)*(1-condition_mask)+(condition_mask)*x_0

        std = self.sde.marginal_prob_std(t).reshape(-1, 1).to(x_0.device)
        x_t = x_0 + std * x_1 * (1-condition_mask)
        return x_t
    
    def embedding_net_value(self, x):
        # Value embedding net (here we just repeat the value)
        #dim_value=self.dim_value
        return x.repeat(1,1,self.dim_value)
    
    def output_scale_function(self, t, x):
        scale = self.sde.marginal_prob_std(t).to(x.device)
        return x / scale
    
    
    #######################################
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
        score = score.unsqueeze(2)

        loss = torch.mean(sigma_t**2 * torch.sum((1-condition_mask)*(x_1+sigma_t*score)**2))

        return loss
    
    ##################################
    # /////////// Training ///////////
    
    def train(self, data, condition_mask_data=None, 
                batch_size=64, epochs=10, lr=1e-3, device="cpu", 
                val_data=None, condition_mask_val=None, 
                verbose=True, checkpoint_path=None):

        start_time = time.time()

        self.to(device)
        self.model.to(device)
        eps = 1e-3

        # Define the optimizer
        #optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        optimizer = schedulefree.AdamWScheduleFree(self.model.parameters(), lr=lr)

        self.train_loss = []
        self.val_loss = []

        if condition_mask_data is None:
        # If no condition mask is provided, eg to just train posterior or likelihood
        # the condition mask is sampled randomly over all data points to be able to predict the likelihood or posterior and also be able to work with missing data
            condition_mask_random_data = torch.distributions.bernoulli.Bernoulli(torch.ones_like(data) * 0.33)

        if condition_mask_val is None and val_data is not None:
            condition_mask_random_val = torch.distributions.bernoulli.Bernoulli(torch.ones_like(val_data) * 0.33)

        self.best_loss = torch.inf
        
        # --------------------
        # ----- Training -----
        for epoch in range(epochs):
            self.model.train()
            optimizer.train()
            idx = torch.randperm(data.shape[0])
            data_shuffled = data[idx,:]

            loss_epoch = 0

            if condition_mask_data is None:
                condition_mask_data = condition_mask_random_data.sample()
            elif len(condition_mask_data.shape) == 1:
                condition_mask_data = condition_mask_data.unsqueeze(0).repeat(data.shape[0], 1)
            
            for i in tqdm.tqdm(range(0, data_shuffled.shape[0], batch_size), desc=f'Epoch {epoch+1:{""}{2}}/{epochs}: ', disable=not verbose):
                optimizer.zero_grad()

                x_0 = data_shuffled[i:i+batch_size].to(device)
                condition_mask_batch = condition_mask_data[i:i+batch_size].to(device)

                # Pick random timesteps in diffusion process
                timestep = torch.rand(x_0.shape[0],1, device=device)* (1. - eps) + eps

                # Sample x_1 for unobserved data
                x_1 = torch.randn_like(x_0)*(1-condition_mask_batch)+(condition_mask_batch)*x_0

                # Calculate x_t for the sampled timestep
                x_t = self.forward_diffusion_sample(x_0, timestep, x_1, condition_mask_batch)

                # Calculate score
                out = self.model(x=x_t, t=timestep, c=condition_mask_batch)
                score = self.output_scale_function(timestep, out)

                # Calculate loss
                loss = self.loss_fn(score, timestep, x_1, condition_mask_batch)
                loss_epoch += loss.item()

                loss.backward()
                #torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
            
            self.train_loss.append(loss_epoch)
            #torch.cuda.empty_cache()

            # ----------------------
            # ----- Validation -----
            if val_data is not None:
                self.model.eval()
                optimizer.eval()

                if condition_mask_val is None:
                    # If no condition mask is provided, val on random condition mask
                    condition_mask_val = condition_mask_random_val.sample()
                elif len(condition_mask_val.shape) == 1:
                    condition_mask_val = condition_mask_val.unsqueeze(0).repeat(val_data.shape[0], 1)

                batch_size_val = 1000
                val_loss = 0
                for i in range(0, val_data.shape[0], batch_size_val):
                    x_0 = val_data[i:i+batch_size_val].to(device)
                    condition_mask_val_batch = condition_mask_val[i:i+batch_size_val].to(device)

                    # Pick random timesteps in diffusion process
                    timestep = torch.rand(x_0.shape[0],1, device=device)* (1. - eps) + eps

                    # Sample x_1 for unobserved data
                    x_1 = torch.randn_like(x_0)*(1-condition_mask_val_batch)+(condition_mask_val_batch)*x_0

                    # Calculate x_t for the sampled timestep
                    x_t = self.forward_diffusion_sample(x_0, timestep, x_1, condition_mask_val_batch)

                    # Calculate score
                    out = self.model(x=x_t, t=timestep, c=condition_mask_val_batch)
                    score = self.output_scale_function(timestep, out)

                    # Calculate loss
                    loss = self.loss_fn(score, timestep, x_1, condition_mask_val_batch)
                    val_loss += loss.item()

                    if val_loss <= self.best_loss and checkpoint_path is not None:
                        self.best_loss = val_loss
                        self.save(f"{checkpoint_path}/ModelTransfuser_best.pickle")
                    
                self.val_loss.append(val_loss)
                #torch.cuda.empty_cache()

                if verbose:
                    print(f'--- Training Loss: {loss_epoch:{""}{11}.3f} --- Validation Loss: {val_loss:{""}{11}.3f} ---')
                    print()
            # ----------------------

            else:
                if loss_epoch <= self.best_loss and checkpoint_path is not None:
                    self.best_loss = loss_epoch
                    self.save(f"{checkpoint_path}/ModelTransfuser_best.pickle")

                if verbose:
                    print(f'--- Training Loss: {loss_epoch:{""}{11}.3f} ---')
                    print()
        
        end_time = time.time()
        time_elapsed = (end_time - start_time) / 60
        print(f"Training finished after {time_elapsed:.1f} minutes")

    #################################
    # /////////// Sample ///////////

    def sample(self, data, condition_mask, temperature=1, timesteps=50, num_samples=1_000, device="cpu"):

        self.model.eval()
        self.model.to(device)
        #self.to(device)

        # Shaping
        # Correct data shape is [Number of unique samples, Number of predicted samples for each unique sample, Number of values]
        if len(data.shape) == 1:
            data = data.unsqueeze(0)
        if len(condition_mask.shape) == 1:
            condition_mask = condition_mask.unsqueeze(0).repeat(data.shape[0], 1)

        self.timesteps = timesteps

        joint_data = torch.zeros_like(condition_mask)
        if torch.sum(condition_mask==1).item()!=0:
            joint_data[condition_mask==1] = data.flatten()

        condition_mask_samples = condition_mask.unsqueeze(1).repeat(1,num_samples,1).to(device)

        data = joint_data
        
        x = data.unsqueeze(1).repeat(1,num_samples,1).to(device)
        random_t1_samples = self.sde.marginal_prob_std(torch.ones_like(x)) * torch.randn_like(x) * (1-condition_mask_samples.to(device))
        x += random_t1_samples
    
        
        dt = (1/timesteps)
        eps = 1e-3
        time_steps = torch.linspace(1., eps, timesteps, device=device)

        self.x_t = torch.zeros(x.shape[0], timesteps+1, x.shape[1], x.shape[2])
        self.score_t = torch.zeros(x.shape[0], timesteps+1, x.shape[1], x.shape[2])
        self.dx_t = torch.zeros(x.shape[0], timesteps+1, x.shape[1], x.shape[2])
        self.x_t[:,0,:,:] = x
        
        for n in tqdm.tqdm(range(len(data))):

            for i,t in enumerate(time_steps):
                timestep = t.reshape(-1, 1).to(device) * (1. - eps) + eps
                
                out = self.model(x=x[n,:], t=timestep, c=condition_mask_samples[n]).squeeze(-1).detach()
                out = out / temperature
                score = self.output_scale_function(timestep, out)
                dx = self.sigma**(2*timestep)* score * dt

                x[n,:] = x[n,:] + dx * (1-condition_mask_samples[n,:])

                self.x_t[n,i+1] = x[n,:]
                self.dx_t[n,i] = dx
                self.score_t[n,i] = score

        return x.detach()
    
    #####################################
    # /////////// Save & Load ///////////

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            model = pickle.load(f)
        return model