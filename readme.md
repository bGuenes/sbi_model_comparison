# Baysian model comparison with score-based diffusion SBI

We use a [score-based diffusion model](https://arxiv.org/abs/2011.13456) with [transformer](https://arxiv.org/abs/1706.03762) architecture ([simformer](https://arxiv.org/abs/2404.09636)) to do SBI for [model comparison](https://academic.oup.com/rasti/article/2/1/710/7382245). <br>
For the training of the simformer we have to first turn our data into noise and then train the simformer to denoise it for every timestep. <br>
Turning data step-by-step into noise: <br>

<p align="center">
  <img src="plots/theta_to_noise.gif" />
</p>