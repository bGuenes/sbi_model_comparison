import torch
import numpy as np
import tqdm

def _hybrid_sampler(self, data, condition_mask=None, timesteps=30, corrector_steps=5, 
                   order=2, snr=0.1, final_corrector_steps=3, temperature=1.0, device="cpu",
                   cfg_alpha=None, verbose=True):
    """
    Hybrid sampling approach combining DPM-Solver with Predictor-Corrector refinement.
    
    Args:
        data: Input data or conditioning information
        condition_mask: Binary mask indicating which parts to condition on
        timesteps: Number of DPM-Solver steps
        corrector_steps: Number of corrector steps per iteration
        order: Order of DPM-Solver (1, 2, or 3)
        snr: Signal-to-noise ratio for corrector steps
        final_corrector_steps: Additional corrector steps at the end
        temperature: Sampling temperature (lower for less diversity)
        cfg_alpha: Classifier-free guidance strength
        verbose: Whether to show progress bars
    
    Returns:
        Sampled data
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
        condition_mask = condition_mask.unsqueeze(0)
    
    # Set up masked data for conditioning
    joint_data = torch.zeros_like(condition_mask)
    if torch.sum(condition_mask==1).item() > 0:
        joint_data[condition_mask==1] = data.flatten()
    
    # Initialize from noise
    x = joint_data.clone()
    sigma_t1 = self.sde.marginal_prob_std(torch.ones(1, device=device)).item()
    x[condition_mask==0] = x[condition_mask==0] + sigma_t1 * torch.randn_like(x)[condition_mask==0]
    
    # Set up time steps for DPM-Solver
    eps = 1e-3
    time_steps = torch.linspace(1.0, eps, timesteps+1, device=device)
    
    # Main sampling loop
    for i in tqdm.tqdm(range(timesteps), desc="DPM-Solver sampling", disable=not verbose):
        t_now = time_steps[i]
        t_next = time_steps[i+1]
        
        # Get score at current timestep
        with torch.no_grad():
            score_t = self.get_score(x, t_now.reshape(-1, 1), condition_mask, cfg_alpha, temperature)
        
        # ------- PREDICTOR: DPM-Solver update -------
        # First-order update
        alpha_now = self.sde.marginal_prob_std(t_now).to(device)
        alpha_next = self.sde.marginal_prob_std(t_next).to(device)
        
        # First-order update
        x_pred = x - (t_now - t_next) * (alpha_now**2) * score_t
        
        if order >= 2 and i < timesteps - 1:
            # Get score at the predicted point
            with torch.no_grad():
                score_next = self.get_score(x_pred, t_next.reshape(-1, 1), condition_mask, cfg_alpha, temperature)
            
            # Second-order correction
            x_pred = x - 0.5 * (t_now - t_next) * (
                (alpha_now**2) * score_t + 
                (alpha_next**2) * score_next
            )
        
        # Apply condition mask to preserve conditional values
        x = x_pred * (1-condition_mask) + joint_data * condition_mask
        
        # ------- CORRECTOR: Langevin MCMC steps -------
        # Only apply corrector steps occasionally to save computation
        if corrector_steps > 0 and (i % 5 == 0 or i >= timesteps - 5):
            steps = corrector_steps
            if i >= timesteps - final_corrector_steps:
                steps = corrector_steps * 2  # More steps at the end
                
            x = self.corrector_step(x, t_next.reshape(-1, 1), condition_mask, 
                                    steps, snr, cfg_alpha, temperature)
    
    return x.detach()

    
def _get_score(self, x, t, condition_mask, cfg_alpha=None, temperature=1.0):
    """Get score estimate with optional classifier-free guidance"""
    # Get conditional score
    score_cond = self.model(x=x, t=t, c=condition_mask) / temperature
    score_cond = self.output_scale_function(t, score_cond)
    
    # Apply classifier-free guidance if requested
    if cfg_alpha is not None:
        score_uncond = self.model(x=x, t=t, c=torch.zeros_like(condition_mask)) / temperature
        score_uncond = self.output_scale_function(t, score_uncond)
        score = score_uncond + cfg_alpha * (score_cond - score_uncond)
    else:
        score = score_cond
        
    return score

def _corrector_step(self, x, t, condition_mask, steps, snr, cfg_alpha=None, temperature=1.0):
    """Corrector steps using Langevin dynamics"""
    for _ in range(steps):
        # Get score estimate
        with torch.no_grad():
            score = self.get_score(x, t, condition_mask, cfg_alpha, temperature)
        
        # Langevin dynamics update
        noise_scale = torch.sqrt(snr * 2 * self.sde.marginal_prob_std(t)**2)
        noise = torch.randn_like(x) * noise_scale
        
        # Update x with the score and noise, respecting the condition mask
        grad_step = snr * self.sde.marginal_prob_std(t)**2 * score
        x = x + grad_step * (1-condition_mask) + noise * (1-condition_mask)

    return x