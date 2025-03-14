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

# --------------------------------------------------------------------------------------------------

class ModelTransfuser(nn.Module):
    # ------------------------------------
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
        
    # ------------------------------------
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
        score = score.unsqueeze(2)

        loss = torch.mean(sigma_t**2 * torch.sum((1-condition_mask)*(x_1+sigma_t*score)**2))

        return loss
    
    # ------------------------------------
    # /////////// Training ///////////
    
    def train(self, data, condition_mask_data=None, 
                batch_size=64, max_epochs=500, lr=1e-3, device="cpu", 
                val_data=None, condition_mask_val=None, 
                verbose=True, checkpoint_path=None, early_stopping_patience=20,
                cfg_prob=0.2):

        start_time = time.time()

        self.model.to(device)
        eps = 1e-3

        optimizer = schedulefree.AdamWScheduleFree(self.model.parameters(), lr=lr)

        self.train_loss = []
        self.val_loss = []
        self.best_loss = torch.inf
        
        # --------------------------------------------------------------------------------------------------
        # --- Training ---
        for epoch in range(max_epochs):
            self.model.train()
            optimizer.train()
            loss_epoch = 0

            for batch in tqdm.tqdm(data, desc=f"Epoch {epoch+1}: ", disable=not verbose):
                optimizer.zero_grad()

                # Expecting batch as (x_0, condition_mask) tuple; modify if needed
                if isinstance(batch, (list, tuple)):
                    x_0, condition_mask_batch = batch
                else:
                    x_0 = batch
                    # If no condition mask in your dataloader, create one based on cfg_prob using random sampling
                    condition_mask_batch = torch.distributions.bernoulli.Bernoulli(0.33).sample(x_0.shape)

                x_0 = x_0.to(device)
                condition_mask_batch = condition_mask_batch.to(device)

                # Classifier-free guidance
                if cfg_prob is not None:
                    rand = torch.rand(1).item()
                    if rand < cfg_prob:
                        condition_mask_batch = torch.zeros_like(condition_mask_batch)

                timestep = torch.rand(x_0.shape[0], 1, device=device) * (1. - eps) + eps

                # Sample x_1 from the random noise
                x_1 = torch.randn_like(x_0)*(1-condition_mask_batch) + (condition_mask_batch)*x_0
                # Calculate x at time t in the diffusion process
                x_t = self.forward_diffusion_sample(x_0, timestep, x_1, condition_mask_batch)
                # Calculate the score
                out = self.model(x=x_t, t=timestep, c=condition_mask_batch)
                score = self.output_scale_function(timestep, out)
                # Calculate the loss
                loss = self.loss_fn(score, timestep, x_1, condition_mask_batch)
                loss_epoch += loss.item()

                loss.backward()
                optimizer.step()

            self.train_loss.append(loss_epoch)

            # -----------------------------------------------------------------------------------------------
            # --- Validation ---
            if val_data is not None:
                self.model.eval()
                optimizer.eval()

                val_loss = 0
                for batch in val_data:
                    if isinstance(batch, (list, tuple)):
                        x_0, condition_mask_val_batch = batch
                    else:
                        x_0 = batch
                        condition_mask_val_batch = torch.distributions.bernoulli.Bernoulli(0.33).sample(x_0.shape)

                    x_0 = x_0.to(device)
                    condition_mask_val_batch = condition_mask_val_batch.to(device)

                    timestep = torch.rand(x_0.shape[0],1, device=device)*(1.-eps) + eps

                    x_1 = torch.randn_like(x_0)*(1-condition_mask_val_batch) + (condition_mask_val_batch)*x_0
                    x_t = self.forward_diffusion_sample(x_0, timestep, x_1, condition_mask_val_batch)
                    out = self.model(x=x_t, t=timestep, c=condition_mask_val_batch)
                    score = self.output_scale_function(timestep, out)
                    val_loss += self.loss_fn(score, timestep, x_1, condition_mask_val_batch).item()
                
                self.val_loss.append(val_loss)

                if val_loss <= self.best_loss:
                    self.best_loss = val_loss
                    patience_counter = 0
                    if checkpoint_path is not None:
                        self.save(f"{checkpoint_path}Model_checkpoint.pickle")
                else:
                    patience_counter += 1
                
                if verbose:
                    print(f'--- Training Loss: {loss_epoch:11.3f} --- Validation Loss: {val_loss:11.3f} ---')
                    print()
            else:
                if val_loss <= self.best_loss:
                    self.best_loss = val_loss
                    patience_counter = 0
                    if checkpoint_path is not None:
                        self.save(f"{checkpoint_path}Model_checkpoint.pickle")
                else:
                    patience_counter += 1

            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                break

        end_time = time.time()
        time_elapsed = (end_time - start_time) / 60

        if verbose:
            print(f"Training finished after {time_elapsed:.1f} minutes")

    # ------------------------------------
    # /////////// Sample ///////////
        
    def sample(self, data, condition_mask=None, temperature=1, timesteps=50, num_samples=1_000, device="cpu", cfg_alpha=None, verbose=True):
        self.model.eval()
        self.model.to(device)

        if isinstance(data, torch.utils.data.DataLoader):
            all_samples = []
            for batch in data:
                if isinstance(batch, (list, tuple)):
                    data_batch, condition_mask_batch = batch
                else:
                    data_batch = batch
                    condition_mask_batch = condition_mask.unsqueeze(0).repeat(data_batch.shape[0], 1)
                samples = self._sample_batch(data_batch, condition_mask_batch, temperature, timesteps, num_samples, device, cfg_alpha, verbose)
                all_samples.append(samples)
            return torch.cat(all_samples, dim=0)
        else:
            if len(data.shape) == 1:
                data = data.unsqueeze(0)
            if len(condition_mask.shape) == 1:
                condition_mask = condition_mask.unsqueeze(0).repeat(data.shape[0], 1)
            return self._sample_batch(data, condition_mask, temperature, timesteps, num_samples, device, cfg_alpha, verbose)

    def _sample_batch(self, data, condition_mask, temperature, timesteps, num_samples, device, cfg_alpha, verbose=True):
        data = data.to(device)
        condition_mask = condition_mask.to(device)

        # Shaping
        if len(data.shape) == 1:
            data = data.unsqueeze(0)
        if len(condition_mask.shape) == 1:
            condition_mask = condition_mask.unsqueeze(0).repeat(data.shape[0], 1)

        self.timesteps = timesteps

        joint_data = torch.zeros_like(condition_mask)
        if torch.sum(condition_mask==1).item()!=0:
            joint_data[condition_mask==1] = data.flatten()

        condition_mask_samples = condition_mask.unsqueeze(1).repeat(1,num_samples,1)

        data = joint_data
        
        x = data.unsqueeze(1).repeat(1,num_samples,1)
        random_t1_samples = self.sde.marginal_prob_std(torch.ones_like(x)) * torch.randn_like(x) * (1-condition_mask_samples)
        x += random_t1_samples

        dt = (1/timesteps)
        eps = 1e-3
        time_steps = torch.linspace(1., eps, timesteps, device=device)

        self.x_t = torch.zeros(x.shape[0], timesteps+1, x.shape[1], x.shape[2])
        self.score_t = torch.zeros(x.shape[0], timesteps+1, x.shape[1], x.shape[2])
        self.dx_t = torch.zeros(x.shape[0], timesteps+1, x.shape[1], x.shape[2])
        self.x_t[:,0,:,:] = x
        
        for n in tqdm.tqdm(range(len(data)), disable=not verbose):
            for i,t in enumerate(time_steps):
                timestep = t.reshape(-1, 1) * (1. - eps) + eps
                
                out_cond = self.model(x=x[n,:], t=timestep, c=condition_mask_samples[n]).squeeze(-1).detach()
                out_cond = out_cond / temperature
                score_cond = self.output_scale_function(timestep, out_cond)

                # Classifier-free guidance
                if cfg_alpha is not None:
                    out_uncond = self.model(x=x[n,:], t=timestep, c=torch.zeros_like(condition_mask_samples[n])).squeeze(-1).detach()
                    out_uncond = out_uncond / temperature
                    score_uncond = self.output_scale_function(timestep, out_uncond)

                    score = score_uncond + cfg_alpha * (score_cond - score_uncond)
                else:
                    score = score_cond

                dx = self.sigma**(2*timestep)* score * dt

                x[n,:] = x[n,:] + dx * (1-condition_mask_samples[n,:])

                self.x_t[n,i+1] = x[n,:]
                self.dx_t[n,i] = dx
                self.score_t[n,i] = score

        return x.detach()

    # ------------------------------------
    # /////////// Advanced Sampling ///////////

    def _get_score(self, x, t, condition_mask, cfg_alpha=None):
        """Get score estimate with optional classifier-free guidance"""
        # Get conditional score
        score_cond = self.model(x=x, t=t, c=condition_mask)
        score_cond = self.output_scale_function(t, score_cond)
        
        # Apply classifier-free guidance if requested
        if cfg_alpha is not None:
            score_uncond = self.model(x=x, t=t, c=torch.zeros_like(condition_mask))
            score_uncond = self.output_scale_function(t, score_uncond)
            score = score_uncond + cfg_alpha * (score_cond - score_uncond)
        else:
            score = score_cond
            
        return score

    def _corrector_step(self, x, t, condition_mask, steps, snr, cfg_alpha=None):
        """Corrector steps using Langevin dynamics"""
        for _ in range(steps):
            # Get score estimate
            with torch.no_grad():
                score = self._get_score(x, t, condition_mask, cfg_alpha)
            
            # Langevin dynamics update
            noise_scale = torch.sqrt(snr * 2 * self.sde.marginal_prob_std(t)**2)
            noise = torch.randn_like(x) * noise_scale
            
            # Update x with the score and noise, respecting the condition mask
            grad_step = snr * self.sde.marginal_prob_std(t)**2 * score
            x = x + grad_step * (1-condition_mask) + noise * (1-condition_mask)

        return x

    def _hybrid_sampler(self, data, condition_mask=None, timesteps=30, num_samples=1_000, corrector_steps=5, 
                    order=2, snr=0.1, final_corrector_steps=3, device="cpu",
                    cfg_alpha=None, verbose=True):
        """
        Hybrid sampling approach combining DPM-Solver with Predictor-Corrector refinement.
        """
        self.model.eval()
        self.model.to(device)
        
        # Process data and condition_mask
        data = data.to(device)
        condition_mask = condition_mask.to(device)
        
        # Ensure proper shapes
        if len(data.shape) == 1:
            data = data.unsqueeze(0)
        if len(condition_mask.shape) == 1:
            condition_mask = condition_mask.unsqueeze(0).repeat(data.shape[0], 1)
        
        # Set up masked data for conditioning
        joint_data = torch.zeros_like(condition_mask)
        if torch.sum(condition_mask==1).item() > 0:
            joint_data[condition_mask==1] = data.flatten()

        condition_mask_samples = condition_mask.unsqueeze(1).repeat(1,num_samples,1)
        
        # Initialize from noise
        x = joint_data.clone().unsqueeze(1).repeat(1,num_samples,1)
        sigma_t1 = self.sde.marginal_prob_std(torch.ones(1, device=device)).item()
        x[condition_mask_samples==0] = sigma_t1 * torch.randn_like(x[condition_mask_samples==0])
        
        # Set up time steps for DPM-Solver
        eps = 1e-3
        time_steps = torch.linspace(1.0, eps, timesteps+1, device=device)
        
        # Main sampling loop
        for n in tqdm.tqdm(range(len(x)), disable=not verbose):
            for i in range(timesteps):
                t_now = time_steps[i]
                t_next = time_steps[i+1]
                
                # Get score at current timestep
                with torch.no_grad():
                    score_t = self._get_score(x[n,:], t_now.reshape(-1, 1), condition_mask_samples[n,:], cfg_alpha)
                
                # ------- PREDICTOR: DPM-Solver update -------
                # First-order update
                alpha_now = self.sde.marginal_prob_std(t_now).to(device)
                alpha_next = self.sde.marginal_prob_std(t_next).to(device)
                
                # First-order update
                x_pred = x[n,:] - (t_now - t_next) * (alpha_now**2) * score_t
                
                if order >= 2 and i < timesteps - 1:
                    # Get score at the predicted point
                    with torch.no_grad():
                        score_next = self._get_score(x_pred, t_next.reshape(-1, 1), condition_mask_samples[n,:], cfg_alpha)
                    
                    # Second-order correction
                    x_pred = x[n,:] - 0.5 * (t_now - t_next) * (
                        (alpha_now**2) * score_t + 
                        (alpha_next**2) * score_next
                    )
                
                # Apply condition mask to preserve conditional values
                x[n,:] = x_pred * (1-condition_mask_samples[n,:]) + x[n,:] * condition_mask_samples[n,:]
                
                # ------- CORRECTOR: Langevin MCMC steps -------
                # Only apply corrector steps occasionally to save computation
                if corrector_steps > 0 and (i % 5 == 0 or i >= timesteps - 5):
                    steps = corrector_steps
                    if i >= timesteps - final_corrector_steps:
                        steps = corrector_steps * 2  # More steps at the end
                        
                    x[n,:] = self._corrector_step(x[n,:], t_next.reshape(-1, 1), condition_mask_samples[n,:], 
                                        steps, snr, cfg_alpha)
            
        return x.detach()
    
    def sample_hybrid(self, data, condition_mask=None, timesteps=30, 
                    corrector_steps=5, order=2, snr=0.1, final_corrector_steps=3,
                    num_samples=1_000, device="cpu", cfg_alpha=None, verbose=True):
        """
        Sample using the hybrid DPM-Solver + Predictor-Corrector approach
        
        Args:
            data: Input data
            condition_mask: Binary mask indicating observed values (1) and latent values (0)
            timesteps: Number of diffusion steps
            corrector_steps: Number of Langevin MCMC steps per iteration
            order: Order of DPM-Solver (1 or 2)
            snr: Signal-to-noise ratio for Langevin steps
            final_corrector_steps: Extra correction steps at the end
            num_samples: Number of samples to generate
            device: Device to run sampling on
            cfg_alpha: Classifier-free guidance strength
            verbose: Whether to show progress bar
        
        Returns:
            Sampled data
        """
        if isinstance(data, torch.utils.data.DataLoader):
            all_samples = []
            for batch in data:
                if isinstance(batch, (list, tuple)):
                    data_batch, condition_mask_batch = batch
                else:
                    data_batch = batch
                    condition_mask_batch = condition_mask.unsqueeze(0).repeat(data_batch.shape[0], 1)
                
                samples = self._hybrid_sampler(data_batch, condition_mask_batch, timesteps, num_samples,
                                        corrector_steps, order, snr, final_corrector_steps,
                                        device, cfg_alpha, verbose)
            return samples
        else:
            if len(data.shape) == 1:
                data = data.unsqueeze(0)
            if len(condition_mask.shape) == 1:
                condition_mask = condition_mask.unsqueeze(0).repeat(data.shape[0], 1)
        
            samples = self._hybrid_sampler(data, condition_mask, timesteps, num_samples,
                                        corrector_steps, order, snr, final_corrector_steps,
                                        device, cfg_alpha, verbose)
            
            return samples

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