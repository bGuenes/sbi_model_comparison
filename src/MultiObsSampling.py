import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import tqdm

class TensorTupleDataset(Dataset):
    def __init__(self, tensor1, tensor2):
        self.tensor1 = tensor1
        self.tensor2 = tensor2
        assert len(tensor1) == len(tensor2), "Tensors must have the same length"
        
    def __len__(self):
        return len(self.tensor1)
    
    def __getitem__(self, idx):
        return self.tensor1[idx], self.tensor2[idx]

#################################################################################################
# ////////////////////////////////////////// Sampling //////////////////////////////////////////
#################################################################################################
class MultiObsSampling():
    def __init__(self, MTf):
        self.MTf = MTf
        # Get SDE from model for calculations
        self.sde = self.MTf.sde

    #############################################
    # ----- Standard Functions -----
    #############################################

    def _get_score(self, x, t, condition_mask, cfg_alpha=None):
        """Get score estimate with optional classifier-free guidance"""
        # Get conditional score
        with torch.no_grad():
            if len(x.shape) == 3:
                score_table = torch.zeros_like(x)
                for i in range(x.shape[0]):
                    score_table[i,:,:] = self.MTf.model(x=x[i], t=t, c=condition_mask[i])
                    score_table[i,:,:] = self.MTf.output_scale_function(t, score_table[i,:,:])

                    # Apply classifier-free guidance if requested
                    if cfg_alpha is not None:
                        score_uncond = self.MTf.model(x=x[i], t=t, c=torch.zeros_like(condition_mask[i]))
                        score_uncond = self.MTf.output_scale_function(t, score_uncond)
                        score_table[i,:,:] = score_uncond + cfg_alpha * (score_table[i,:,:] - score_uncond)

                if self.multi_observation:
                    score = self._compositional_score(score_table, t, self.hierarchy)
            
            else:
                score_cond = self.MTf.model(x=x, t=t, c=condition_mask)
                score_cond = self.MTf.output_scale_function(t, score_cond)
                
                # Apply classifier-free guidance if requested
                if cfg_alpha is not None:
                    score_uncond = self.MTf.model(x=x, t=t, c=torch.zeros_like(condition_mask))
                    score_uncond = self.MTf.output_scale_function(t, score_uncond)
                    score = score_uncond + cfg_alpha * (score_cond - score_uncond)
                else:
                    score = score_cond

        return score

    def _check_data_shape(self, data, condition_mask):
        # Check data shape
        # Required shape: (num_samples, num_features)
        if len(data.shape) == 1:
            data = data.unsqueeze(0)

        # Check condition mask shape
        # Required shape: (num_samples, num_features)
        if len(condition_mask.shape) == 1:
            condition_mask = condition_mask.unsqueeze(0).repeat(data.shape[0], 1)

        return data, condition_mask
        
    def _check_data_structure(self, data, condition_mask=None, batch_size=1e3):
        # Check if data is a DataLoader
        # Convert if necessary
        if not isinstance(data, torch.utils.data.DataLoader):
            if condition_mask is None:
                raise ValueError("Condition mask must be provided if data is not a DataLoader.")
            
            data, condition_mask = self._check_data_shape(data, condition_mask)
            dataset_cond = TensorTupleDataset(data, condition_mask)
            data_loader = DataLoader(dataset_cond, batch_size=int(batch_size), shuffle=False)
            return data_loader
        
        else:
            data_iter = iter(data)
            batch = next(data_iter)

            # Check if data is a tuple
            if isinstance(batch, tuple):
                return data
            else:
                data, condition_mask = self._check_data_shape(batch)
                dataset_cond = TensorTupleDataset(data, condition_mask)
                data_loader = DataLoader(dataset_cond, batch_size=int(batch_size), shuffle=False)
                return data_loader

    def _prepare_data(self, batch, num_samples, device):
        # Expand data and condition mask to match num_samples
        data, condition_mask = batch
        data = data.to(device)
        condition_mask = condition_mask.to(device)

        data = data.unsqueeze(1).repeat(1,num_samples,1)
        condition_mask = condition_mask.unsqueeze(1).repeat(1,num_samples,1)

        joint_data = torch.zeros_like(condition_mask)
        if torch.sum(condition_mask==1).item()!=0:
            joint_data[condition_mask==1] = data.flatten()

        return joint_data, condition_mask
    
    def _initial_sample(self, data, condition_mask):
        # Initialize with noise
        # Draw samples from initial noise distribution for latent variables
        # Keep observed variables fixed

        if self.multi_observation:
            # In case for compositional score modeling the random samples need to be the same across all samples
            random_noise_samples = self.sde.marginal_prob_std(torch.ones_like(data)) * torch.randn(data.shape[1], data.shape[2], device=self.device).unsqueeze(0).repeat(data.shape[0],1,1) * (1-condition_mask)
        else:
            random_noise_samples = self.sde.marginal_prob_std(torch.ones_like(data)) * torch.randn_like(data) * (1-condition_mask)

        data += random_noise_samples
        return data
    
    def _compositional_score(self, scores, t, hierarchy):
        prefactor = 0#(1 - len(scores))*(1 - t) # MISSING: score of prior
        compositional_scores = prefactor + torch.sum(scores, dim=0)
        compositional_scores = compositional_scores.repeat(len(scores), 1, 1)
        scores[:,:,hierarchy] = compositional_scores[:,:,hierarchy]/len(scores)

        return scores

    #############################################
    # ----- Main Sampling Loop -----
    #############################################
    
    def sample(self, data, condition_mask=None, timesteps=50, eps=1e-3, num_samples=1000, cfg_alpha=None, multi_obs_inference=False,
               order=2, snr=0.1, corrector_steps_interval=5, corrector_steps=5, final_corrector_steps=3,
               device="cpu", verbose=True, method="dpm", save_trajectory=False):
        """
        Sample from the model using the specified method

        Args:
            data: Input data
                    - Should be a DataLoader or a tuple of (data, condition_mask)
                        - Shape data: (num_samples, num_observed_features)
                        - Shape condition_mask: (num_samples, num_total_features)
                    - Can also be a single tensor of data, in which case condition_mask must be provided
            condition_mask: Binary mask indicating observed values (1) and latent values (0)
                    Shape: (num_samples, num_total_features)
            timesteps: Number of diffusion steps
            eps: End time for diffusion process
            num_samples: Number of samples to generate
            cfg_alpha: Classifier-free guidance strength
            multi_obs_inference: Whether to do bayesian inference with compositional data
                    False: Standard sampling
                    True: Multi-observation sampling on all latent variables
                    List of indices: Compositional indices

            - DPM-Solver parameters -
            order: Order of DPM-Solver (1, 2 or 3)
            snr: Signal-to-noise ratio for Langevin steps
            corrector_steps_interval: Interval for applying corrector steps
            corrector_steps: Number of Langevin MCMC steps per iteration
            final_corrector_steps: Extra correction steps at the end

            - Other parameters -
            device: Device to run sampling on
            verbose: Whether to show progress bar
            method: Sampling method to use (euler, dpm)
            save_trajectory: Whether to save the intermediate denoising trajectory
        """
        
        # Set parameters
        self.timesteps = timesteps
        self.timesteps_list = torch.linspace(1., eps, timesteps, device=device)
        self.dt = self.timesteps_list[0] - self.timesteps_list[1]
        self.eps = eps
        self.num_samples = num_samples
        self.cfg_alpha = cfg_alpha
        self.verbose = verbose
        self.device = device
        self.method = method
        self.save_trajectory = save_trajectory

        if method == "dpm":
            self.corrector_steps_interval = corrector_steps_interval
            self.corrector_steps = corrector_steps
            self.final_corrector_steps = final_corrector_steps
            self.snr = snr
            self.order = order

        if multi_obs_inference != False:
            if multi_obs_inference == True:
                self.multi_observation = True
                self.hierarchy = torch.where(condition_mask[0] == 0)[0].tolist()
            else:
                self.multi_observation = True
                self.hierarchy = multi_obs_inference

        # Check data structure
        data_loader = self._check_data_structure(data, condition_mask)
        
        # Move model to device and set to eval mode
        self.MTf.model.eval()
        self.MTf.to(self.device)
        
        # Loop over data samples
        all_samples = []
        for batch in data_loader:
            
            # Prepare data for sampling
            data_batch, condition_mask_batch = self._prepare_data(batch, num_samples, device)

            # Draw samples from initial noise distribution
            data_batch = self._initial_sample(data_batch, condition_mask_batch)
            
            # Get samples for this batch
            if method == "euler":
                samples = self._basic_sampler(data_batch, condition_mask_batch)
            elif method == "dpm" and not self.multi_observation:
                samples = self._dpm_sampler(data_batch, condition_mask_batch,
                                            order=order, snr=snr, corrector_steps_interval=corrector_steps_interval,
                                            corrector_steps=corrector_steps, final_corrector_steps=final_corrector_steps)
            elif method == "dpm" and self.multi_observation:
                samples = self._dpm_sampler_all(data_batch, condition_mask_batch,
                                            order=order, snr=snr, corrector_steps_interval=corrector_steps_interval,
                                            corrector_steps=corrector_steps, final_corrector_steps=final_corrector_steps)
            else:
                raise ValueError(f"Sampling method {method} not recognized.")

            all_samples.append(samples)
            
        # Return concatenated samples
        return torch.cat(all_samples, dim=0) if len(all_samples) > 1 else all_samples[0]
    
    #############################################
    # ----- Basic Sampling -----
    #############################################
    
    # Euler-Maruyama sampling
    def _basic_sampler(self, data, condition_mask):
        """
        Basic Euler-Maruyama sampling method
        
        Args:
            data: Input data 
                    Shape: (batch_size, num_samples, num_features)
            condition_mask: Binary mask indicating observed values (1) and latent values (0)
                    Shape: (batch_size, num_samples, num_features)  
        """

        if self.save_trajectory:
            # Storage for trajectory (optional)
            self.data_t = torch.zeros(data.shape[0], self.timesteps+1, data.shape[1], data.shape[2])
            self.score_t = torch.zeros(data.shape[0], self.timesteps+1, data.shape[1], data.shape[2])
            self.dx_t = torch.zeros(data.shape[0], self.timesteps+1, data.shape[1], data.shape[2])
            self.data_t[:,0,:,:] = data
            
        # Main sampling loop
        for n in tqdm.tqdm(range(len(data)), disable=not self.verbose):
            for i, t in enumerate(self.timesteps_list):

                t = t.reshape(-1, 1)
                
                # Get score estimate
                score = self._get_score(data[n,:], t, condition_mask[n,:], self.cfg_alpha)
                
                # Update step
                dx = self.sde.sigma**(2*t) * score * self.dt
                
                # Apply update respecting condition mask
                data[n,:] = data[n,:] + dx * (1-condition_mask[n,:])
                
                if self.save_trajectory:
                    # Store trajectory data
                    self.data_t[n,i+1] = data[n,:]
                    self.dx_t[n,i] = dx
                    self.score_t[n,i] = score

        return data.detach()
    
    #############################################
    # ----- Advanced Sampling -----
    #############################################
    
    # DPM sampling with Langevin corrector steps

    def _corrector_step(self, x, t, condition_mask, steps, snr, cfg_alpha=None):
        """
        Corrector steps using Langevin dynamics
        
        Args:
            x: Input data
            t: Time step
            """
        for _ in range(steps):
            # Get score estimate
            score = self._get_score(x, t, condition_mask, cfg_alpha)
            
            # Langevin dynamics update
            noise_scale = torch.sqrt(snr * 2 * self.sde.marginal_prob_std(t)**2)
            noise = torch.randn_like(x) * noise_scale
            
            # Update x with the score and noise, respecting the condition mask
            grad_step = snr * self.sde.marginal_prob_std(t)**2 * score
            x = x + grad_step * (1-condition_mask) + noise * (1-condition_mask)

        return x

    def _dpm_solver_1_step(self, data_t, t, t_next, condition_mask):
        """First-order solver"""
        sigma_now = self.sde.sigma_t(t)

        # First-order step
        score_now = self._get_score(data_t, t, condition_mask, self.cfg_alpha)
        data_next = data_t - (t-t_next) * sigma_now * score_now * (1-condition_mask)

        return data_next
    
    def _dpm_solver_2_step(self, data_t, t, t_next, condition_mask):
        """Second-order solver"""
        sigma_now = self.sde.sigma_t(t)
        sigma_next = self.sde.sigma_t(t_next)

        # First-order step
        score_half = self._get_score(data_t, t, condition_mask, self.cfg_alpha)
        data_half = data_t - (t-t_next) * sigma_now * score_half * (1-condition_mask)

        # Second-order step
        score_next = self._get_score(data_half, t_next, condition_mask, self.cfg_alpha)
        data_next = data_t - 0.5 * (t-t_next) * (sigma_now**2 * score_half + sigma_next**2 * score_next) * (1-condition_mask)

        return data_next
    
    def _dpm_solver_3_step(self, data_t, t, t_next, condition_mask):
        """Third-order solver"""
        # Get sigma values at different time points
        sigma_t = self.sde.sigma_t(t)
        t_mid = (t + t_next) / 2
        sigma_mid = self.sde.sigma_t(t_mid)
        sigma_next = self.sde.sigma_t(t_next)

        # First calculate the intermediate score at time t
        score_t = self._get_score(data_t, t, condition_mask, self.cfg_alpha)
        
        # First intermediate point (Euler step)
        data_mid1 = data_t - (t - t_mid) * sigma_t * score_t * (1-condition_mask)
        
        # Get score at the first intermediate point
        score_mid1 = self._get_score(data_mid1, t_mid, condition_mask, self.cfg_alpha)
        
        # Second intermediate point (using first intermediate)
        data_mid2 = data_t - (t - t_mid) * ((1/3) * sigma_t * score_t + (2/3) * sigma_mid * score_mid1) * (1-condition_mask)
        
        # Get score at the second intermediate point
        score_mid2 = self._get_score(data_mid2, t_mid, condition_mask, self.cfg_alpha)
        
        # Final step using all information
        data_next = data_t - (t - t_next) * ((1/4) * sigma_t * score_t + 
                                            (3/4) * sigma_next * score_mid2) * (1-condition_mask)
        
        return data_next

    def _dpm_sampler(self, data, condition_mask, 
                     order=2, 
                     snr=0.1, corrector_steps_interval=5, corrector_steps=5, final_corrector_steps=3):
        """
        Hybrid sampling approach combining DPM-Solver with Predictor-Corrector refinement.

        Args:
            data: Input data
            condition_mask: Binary mask indicating observed values (1) and latent values (0)
            order: Order of DPM-Solver (1, 2 or 3)
            snr: Signal-to-noise ratio for Langevin steps
            corrector_steps_interval: Interval for applying corrector steps
            corrector_steps: Number of Langevin MCMC steps per iteration
            final_corrector_steps: Extra correction steps at the end
        """

        if self.save_trajectory:
            # Storage for trajectory (optional)
            self.data_t = torch.zeros(data.shape[0], self.timesteps+1, data.shape[1], data.shape[2])
            self.data_t[:,0,:,:] = data

        # Main sampling loop
        for n in tqdm.tqdm(range(len(data)), disable=not self.verbose):
            for i in range(self.timesteps-1):
                
                # ------- PREDICTOR: DPM-Solver -------
                t_now = self.timesteps_list[i].reshape(-1, 1)
                t_next = self.timesteps_list[i+1].reshape(-1, 1)

                if order == 1:
                    data[n,:] = self._dpm_solver_1_step(data[n,:], t_now, t_next, condition_mask[n,:])
                elif order == 2:
                    data[n,:] = self._dpm_solver_2_step(data[n,:], t_now, t_next, condition_mask[n,:])
                elif order == 3:
                    data[n,:] = self._dpm_solver_3_step(data[n,:], t_now, t_next, condition_mask[n,:])
                else:
                    raise ValueError(f"Only orders 1, 2 or 3 are supported in the DPM-Solver.")
                
                # ------- CORRECTOR: Langevin MCMC steps -------
                # Only apply corrector steps occasionally to save computation
                if corrector_steps > 0 and (i % corrector_steps_interval == 0 or i >= self.timesteps - corrector_steps_interval):
                    steps = corrector_steps
                    if i >= self.timesteps - final_corrector_steps:
                        steps = corrector_steps * 2  # More steps at the end
                        
                    data[n,:] = self._corrector_step(data[n,:], t_next, condition_mask[n,:], 
                                               steps, snr, self.cfg_alpha)

                if self.save_trajectory:
                    # Store trajectory data
                    self.data_t[n,i+1] = data[n,:]

        return data.detach()
    
    def _dpm_sampler_all(self, data, condition_mask, 
                     order=2, 
                     snr=0.1, corrector_steps_interval=5, corrector_steps=5, final_corrector_steps=3):
        """
        Hybrid sampling approach combining DPM-Solver with Predictor-Corrector refinement.

        Args:
            data: Input data
            condition_mask: Binary mask indicating observed values (1) and latent values (0)
            order: Order of DPM-Solver (1, 2 or 3)
            snr: Signal-to-noise ratio for Langevin steps
            corrector_steps_interval: Interval for applying corrector steps
            corrector_steps: Number of Langevin MCMC steps per iteration
            final_corrector_steps: Extra correction steps at the end
        """

        if self.save_trajectory:
            # Storage for trajectory (optional)
            self.data_t = torch.zeros(data.shape[0], self.timesteps+1, data.shape[1], data.shape[2])
            self.data_t[:,0,:,:] = data

        # Main sampling loop
        for i in tqdm.tqdm(range(self.timesteps-1), disable=not self.verbose):
            
            # ------- PREDICTOR: DPM-Solver -------
            t_now = self.timesteps_list[i].reshape(-1, 1)
            t_next = self.timesteps_list[i+1].reshape(-1, 1)

            if order == 1:
                data = self._dpm_solver_1_step(data, t_now, t_next, condition_mask)
            elif order == 2:
                data = self._dpm_solver_2_step(data, t_now, t_next, condition_mask)
            elif order == 3:
                data = self._dpm_solver_3_step(data, t_now, t_next, condition_mask)
            else:
                raise ValueError(f"Only orders 1, 2 or 3 are supported in the DPM-Solver.")
            
            # ------- CORRECTOR: Langevin MCMC steps -------
            # Only apply corrector steps occasionally to save computation
            if corrector_steps > 0 and (i % corrector_steps_interval == 0 or i >= self.timesteps - corrector_steps_interval):
                steps = corrector_steps
                if i >= self.timesteps - final_corrector_steps:
                    steps = corrector_steps * 2  # More steps at the end
                    
                data = self._corrector_step(data, t_next, condition_mask, 
                                            steps, snr, self.cfg_alpha)

            if self.save_trajectory:
                # Store trajectory data
                self.data_t[:,i+1] = data

        return data.detach()
    