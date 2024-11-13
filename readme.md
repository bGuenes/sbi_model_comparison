# Baysian model comparison with score-based diffusion SBI

We use a [score-based diffusion model](https://arxiv.org/abs/2011.13456) with [transformer](https://arxiv.org/abs/1706.03762) architecture ([simformer](https://arxiv.org/abs/2404.09636)) to do SBI for [model comparison](https://academic.oup.com/rasti/article/2/1/710/7382245). <br>

### Perturbing Data with a Diffusion Process
For the training of the simformer we have to first turn our data into noise and then train the simformer to denoise it for every timestep. <br>
The diffusion process is defined by the following equation:

$$
d\mathbf{x} = \mathbf{f}(\mathbf{x},t) dt + g(t) d\mathbf{w}
$$

where $\mathbf{f}(\cdot, t): \mathbb{R}^d \to \mathbb{R}^d$ is called the *drift coefficient* of the SDE, $g(t) \in \mathbb{R}$ is called the *diffusion coefficient*, and $\mathbf{w}$ represents the standard Brownian motion.

Step-by-step diffusion process with Variance Exploding SDE (VESDE): <br>

Galactic Parameters           |  Chemical Abundances
:-------------------------:|:-------------------------:
![](plots/theta_to_noise.gif)  |  ![](plots/x_to_noise.gif)

### Reversing the Diffusion Process
Reversing the diffusion process is running the diffusion process backwards in time. 

$$
d\mathbf{x} = [\mathbf{f}(\mathbf{x}, t) - g^2(t)\nabla_{\mathbf{x}}\log p_t(\mathbf{x})] dt + g(t) d\bar{\mathbf{w}}
$$

where $\bar{\mathbf{w}}$ is a Brownian motion in the reverse time direction, and $dt$ represents an infinitesimal negative time step. This reverse SDE can be computed once we know the drift and diffusion coefficients of the forward SDE, as well as the score of $p_t(\mathbf{x})$ for each $t\in[0, T]$. <br>

In the case of the VESDE, the drift and diffusion coefficients are given by:

$$
\mathbf{f}(\mathbf{x}, t) = \mathbf{0} \quad \text{and} \quad g(t) = \sigma^t
$$

where $\sigma$ is a hyperparameter that controls the scale of the diffusion process. <br> <br>
The remaining unkown would then only be the score-function $\nabla_\mathbf{x} \log p_t(\mathbf{x})$ to numerically obtain samples $p_0$ from the prior distribution $p_T$. <br>
That can be done by approximating the score function

$$
s_{\theta}(\mathbf{x},t) \approx \nabla_\mathbf{x} \log p_t(\mathbf{x})
$$

with a neural network. In this case, we use a transformer architecture to approximate the score function. <br>
