# Baysian model comparison with score-based diffusion SBI

Use a score-based diffusion model with transformer architecture (simformer) to do SBI for model comparison. <br>
For the training of the simformer we have to first turn our data into noise and then train the simformer to denoise. <br>
Turning data step-by-step into noise: <br>

<p align="center">
  <img src="plots/theta_to_noise.gif" />
</p>