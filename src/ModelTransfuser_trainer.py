import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

import schedulefree

import numpy as np

import pickle
import sys
import os
import tqdm
import time

from src.ConditionTransformer import DiT
from src.Simformer import Transformer
from src.sde import VESDE, VPSDE
from src.Sampler import Sampler
from src.Trainer import Trainer
from src.MultiObsSampling import MultiObsSampling

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
        
        # Define Trainer
        self.trainer = Trainer(self)
        # Define Sampler
        self.sampler = Sampler(self)
        self.multi_obs_sampler = MultiObsSampling(self)

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
    
    def output_scale_function(self, t, x):
        scale = self.sde.marginal_prob_std(t).to(x.device)
        return x / scale
    
    
    # ------------------------------------
    # /////////// Training ///////////
    
    def train(self, data, condition_mask_data=None, 
                batch_size=128, max_epochs=500, lr=1e-3, device="cpu", 
                val_data=None, condition_mask_val=None, 
                verbose=True, checkpoint_path=None, early_stopping_patience=20,
                cfg_prob=0.2):

        if device == "cuda":
            world_size = torch.cuda.device_count()
        else :
            world_size = 1

        self.trainer.train(world_size=world_size, train_data=data, condition_mask_data=condition_mask_data, val_data=val_data, condition_mask_val=condition_mask_val,
                            max_epochs=max_epochs, early_stopping_patience=early_stopping_patience, batch_size=batch_size, lr=lr, cfg_prob=cfg_prob,
                            checkpoint_path=checkpoint_path, device=device, verbose=verbose)

    # ------------------------------------
    # /////////// Sample ///////////
        
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

        if multi_obs_inference == False:
            if device == "cuda":
                # Run sampling on all available GPUs
                world_size = torch.cuda.device_count()
                if world_size > 1:
                    os.environ['MASTER_ADDR'] = 'localhost'
                    os.environ["MASTER_PORT"] = "29500"

                    manager = mp.Manager()
                    result_dict = manager.dict()
                    #mp.set_start_method('spawn', force=True)
                    mp.spawn(self.sampler.sample, 
                            args=(world_size, data, condition_mask, timesteps, eps, num_samples, cfg_alpha,
                                    order, snr, corrector_steps_interval, corrector_steps, final_corrector_steps,
                                    device, verbose, method, save_trajectory, result_dict), 
                            nprocs=world_size, join=True)
                    samples = result_dict.get('samples', None)
                    manager.shutdown()

            else:
                # Run sampling on specified device
                samples = self.sampler.sample(rank=0, world_size=1, data=data, condition_mask=condition_mask, timesteps=timesteps, num_samples=num_samples, device=device, cfg_alpha=cfg_alpha,
                                        order=order, snr=snr, corrector_steps_interval=corrector_steps_interval, corrector_steps=corrector_steps, final_corrector_steps=final_corrector_steps,
                                        verbose=verbose, method=method, save_trajectory=save_trajectory)
        
        if multi_obs_inference == True:
            # Hierarchical Compositional Score Modeling
            samples = self.multi_obs_sampler.sample(data=data, condition_mask=condition_mask, timesteps=timesteps, num_samples=num_samples, device=device, cfg_alpha=cfg_alpha, multi_obs_inference=multi_obs_inference,
                                      order=order, snr=snr, corrector_steps_interval=corrector_steps_interval, corrector_steps=corrector_steps, final_corrector_steps=final_corrector_steps,
                                      verbose=verbose, method=method, save_trajectory=save_trajectory)

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