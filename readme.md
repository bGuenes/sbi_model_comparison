# Baysian model comparison with score-based diffusion SBI

We use a [score-based diffusion model](https://arxiv.org/abs/2011.13456) with [transformer](https://arxiv.org/abs/1706.03762) architecture ([simformer](https://arxiv.org/abs/2404.09636)) to do SBI for [model comparison](https://academic.oup.com/rasti/article/2/1/710/7382245). <br>

## Diffusion Model

### Perturbing Data with a Diffusion Process
For the training of the simformer we have to first turn our data into noise and then train the diffusion model to denoise it for every timestep. <br>
The diffusion process is defined by the following equation:

$$
d\mathbf{x} = \mathbf{f}(\mathbf{x},t) dt + g(t) d\mathbf{w}
$$

where $\mathbf{f}(\cdot, t): \mathbb{R}^d \to \mathbb{R}^d$ is called the *drift coefficient* of the SDE, $g(t) \in \mathbb{R}$ is called the *diffusion coefficient*, and $\mathbf{w}$ represents the standard Brownian motion. <br>
We use a Variance Exploding SDE (VESDE) for the diffusion process, where the *drift* and *diffusion coefficients* are defined as followed:

$$
\mathbf{f}(\mathbf{x}, t) = \mathbf{0} \quad \text{and} \quad g(t) = \sigma^t
$$

where $\sigma$ is a hyperparameter that controls the scale of the diffusion process. <br>
Following the theory of SDEs, we can formulate the pertubated data distribution at timestep $t$ as:

$$
    p_{0t}(\mathbf{x}_t|\mathbf{x}_0) = \mathcal{N} \bigg( \mathbf{x}_t | \mathbf{x}_0, \frac{1}{2 \ln \sigma}(\sigma^{2t}-1)\mathbf{I} \bigg)
$$

So the variance function over time of the pertubated data distribution is given by:

$$
    \sigma_t^2 = \frac{1}{2 \ln \sigma}(\sigma^{2t}-1)
$$

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
That can be done by approximating the score function with a neural network.

$$
s_{\theta}(\mathbf{x},t) \approx \nabla_\mathbf{x} \log p_t(\mathbf{x})
$$

In this case, we use a transformer architecture to approximate the score function. <br>

### Time-Dependent Score-Based Model
In theory there are no limitations on the model used to approximate the score function. However, as proposed in the [All-in-one Simualtion Based Inference](https://arxiv.org/abs/2404.09636) paper, we use a transformer architecture as they  overcome limitations of feed-forward networks in effectively dealing with sequential inputs. <br>

The transformer takes tokens as inputs and processes them in parallel. The transformer consists of an encoder and a decoder. The encoder processes the input tokens and the decoder processes the output tokens. The transformer uses self-attention to weigh the importance of each token in the input sequence. <br>

Our tokens are the data values $\mathbf{x}$, the embedded nodes, condtion mask and the time $t$. 

|| Node ID | Values | Condition Mask | Time |
|-------------------------|:-------------------------:| :-------------------------:| :-------------------------:|:-------------------------:|
||Unique ID for every Value | Joint data $\hat{x}$ | Binary Conndition indicating observed or latent | Time in diffusion process|
| **Shape** | `(batch_size,sequence_length)` | `(batch_size,sequence_length,1)` | `(batch_size,sequence_length,1)` | `(batch_size,1)` |
| **Example** | `[0, 1, 2]`<br>`[0, 1, 2]`<br>`[0, 1, 2]` | `[[0.1], [0.2], [0.3]]`<br>`[[1.1], [1.2], [1.3]]`<br>`[[2.1], [2.2], [2.3]]` | `[[0], [0], [1]]`<br>`[[0], [1], [1]]`<br>`[[1], [0], [1]]` | `[10]`<br>`[25]`<br>`[99]` |
| **Embedding**        | Embedded over `dim_id=20` using [`nn.Embedding()`](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html) | Repeated across `dim_values=20` | Embedded over `dim_condition=10` learnable parameters | Embedded over `dim_time=20` using [`GaussianFourierEmbedding`](https://arxiv.org/abs/2006.10739) |

The tokens are passed through the [`nn.Transformer()`](https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html) and then decoded in an output layer to estimate the score for each value. <br>

### Training

We can train a time-dependent score-based model $s_{\theta}(\mathbf{x},t)$ to approximate the score function $\nabla_{\mathbf{x}}\log p_t(\mathbf{x})$ to obtain samples from $p_0$ using samples from a prior distribution $p_T$. <br>
During the training process our goal is to minimize the following weighted sum of denoising score matching objectives: 

$$
    \min_\theta \mathbb{E}_{t} \bigg[\lambda(t) \mathbb{E}_{\mathbf{x}_0}\mathbb{E}_{\mathbf{x}_t}\big[ \| s_\theta(\mathbf{x}_t,t) - \nabla_{\mathbf{x}}\log p_{t}(\mathbf{x})\|_2^2 \big]\bigg]
$$

The expectation over $\mathbf{x}_0$ can be estimated by samples from our original data distribution $p_0$. <br>
The expectation over $\mathbf{x}_t$ can be estimated by samples from the pertubated data distribution $p_{0t}$. <br>
And $\lambda(t)$ is the weighting function that can be used to assign different importance to different timesteps. In the case of our VESDE we set it to $\lambda(t) = \sigma_t^2$. <br>
<br>
The training process follows these steps:
1. Pick a datapoint $\mathbf{x}_0$
2. Sample $\mathbf{x}_1 \sim \mathcal{N}(\mathbf{x}|0,\mathbf{I})$
3. Sample $t \sim \text{Uniform}(0,1)$
4. Calculate $\mathbf{x}_t = \mathbf{x}_0 + \sqrt{\frac{1}{2 \ln \sigma}(\sigma^{2t}-1)} \cdot \mathbf{x}_1$. This is a sample from $p_{0t}(\mathbf{x}_t|\mathbf{x}_0)$
5. Evaluate the score model at $\mathbf{x}_t$ and $t$, $s_{\theta}(\mathbf{x}_t,t)$
6. Calculate the score matching loss for a single sample: $\mathcal{L}_t(\theta) = \sigma_t^2 ||\mathbf{x}_1-\sigma_ts_{\theta}(\mathbf{x}_t,t)||^2$
7. Update $\theta$ using gradient-based method with $\nabla_{\theta}\mathcal{L}(\theta)$

### Value Denoising
