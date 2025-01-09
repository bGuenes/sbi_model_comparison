# Bayesian model comparison with score-based diffusion SBI

We use a [score-based diffusion model](https://arxiv.org/abs/2011.13456) with [transformer](https://arxiv.org/abs/1706.03762) architecture ([simformer](https://arxiv.org/abs/2404.09636)) to do SBI for [model comparison](https://academic.oup.com/rasti/article/2/1/710/7382245). <br>

``` mermaid
flowchart LR
    A@{label: "Simulation Data", shape: cyl} --> |train| B(Diffusion Model)
    

    B --> C((NLE))
    B --> D((NPE))
    E@{label: "Observation Data", shape: cyl} --> |posterior estimate| D

    D --> F([MAP])

    F --> |evaluate likelihood| C
    C & D --> G([Marginal Likelihood])
    C & G --> H([Updated Model Prior])
    
```

---
---
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
Following the theory of SDEs, we can formulate the perturbed data distribution at timestep $t$ as:

$$
    p_{0t}(\mathbf{x}_t|\mathbf{x}_0) = \mathcal{N} \bigg( \mathbf{x}_t | \mathbf{x}_0, \frac{1}{2 \ln \sigma}(\sigma^{2t}-1)\mathbf{I} \bigg)
$$

So the variance function over time of the perturbed data distribution is given by:

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
d\mathbf{x} = \big[\mathbf{f}(\mathbf{x}, t) - g^2(t)\nabla_{\mathbf{x}}\log p_t(\mathbf{x}) \big] dt + g(t) d\bar{\mathbf{w}}
$$

where $\bar{\mathbf{w}}$ is a Brownian motion in the reverse time direction, and $dt$ represents an infinitesimal negative time step. This reverse SDE can be computed once we know the drift and diffusion coefficients of the forward SDE, as well as the score of $p_t(\mathbf{x})$ for each $t\in[0, T]$. <br>

In the case of the VESDE, the drift and diffusion coefficients are given by:

$$
\mathbf{f}(\mathbf{x}, t) = \mathbf{0} \quad \text{and} \quad g(t) = \sigma^t
$$

where $\sigma$ is a hyperparameter that controls the scale of the diffusion process. <br> <br>
The remaining unknown would then only be the score-function $\nabla_\mathbf{x} \log p_t(\mathbf{x})$ to numerically obtain samples $p_0$ from the prior distribution $p_T$. <br>
That can be done by approximating the score function with a neural network.

$$
s_{\theta}(\mathbf{x},t) \approx \nabla_\mathbf{x} \log p_t(\mathbf{x})
$$

In this case, we use a transformer architecture to approximate the score function. <br>

### Time-Dependent Score-Based Model
In theory there are no limitations on the model used to approximate the score function. However, as proposed in the [All-in-one Simualtion Based Inference](https://arxiv.org/abs/2404.09636) paper, we use a transformer architecture as they  overcome limitations of feed-forward networks in effectively dealing with sequential inputs. <br>

The transformer takes tokens as inputs and processes them in parallel. The transformer consists of an encoder and a decoder. The encoder processes the input tokens and the decoder processes the output tokens. The transformer uses self-attention to weigh the importance of each token in the input sequence. <br>

Our tokens are the data values $\mathbf{x}$, the embedded nodes, condition mask and the time $t$. 

|| Node ID | Values | Condition Mask | Time |
|-------------------------|:-------------------------:| :-------------------------:| :-------------------------:|:-------------------------:|
||Unique ID for every Value | Joint data $\hat{x}$ | Binary Condition indicating observed or latent | Time in diffusion process|
| **Shape** | `(batch_size,sequence_length)` | `(batch_size,sequence_length,1)` | `(batch_size,sequence_length,1)` | `(batch_size,1)` |
| **Example** | `[0, 1, 2]`<br>`[0, 1, 2]`<br>`[0, 1, 2]` | `[[0.1], [0.2], [0.3]]`<br>`[[1.1], [1.2], [1.3]]`<br>`[[2.1], [2.2], [2.3]]` | `[[0], [0], [1]]`<br>`[[0], [1], [1]]`<br>`[[1], [0], [1]]` | `[10]`<br>`[25]`<br>`[99]` |
| **Embedding**        | Embedded over `dim_id=20` using [`nn.Embedding()`](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html) | Repeated across `dim_values=20` | Embedded over `dim_condition=10` learnable parameters | Embedded over `dim_time=64` using [`GaussianFourierEmbedding`](https://arxiv.org/abs/2006.10739) |

The tokens are passed through the [`nn.Transformer()`](https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html) and then decoded in an output layer to estimate the score for each value. <br>

### Training

We can train a time-dependent score-based model $s_{\theta}(\mathbf{x},t)$ to approximate the score function $\nabla_{\mathbf{x}}\log p_t(\mathbf{x})$ to obtain samples from $p_0$ using samples from a prior distribution $p_T$. <br>
During the training process our goal is to minimize the following weighted sum of denoising score matching objectives: 

$$
    \min_ \theta \mathbb{E}_ {t} \bigg[\lambda(t) \mathbb{E}_ {\mathbf{x}_ 0} \mathbb{E}_ {\mathbf{x}_ t}\big[ \Vert s_ \theta(\mathbf{x}_ t,t) - \nabla_ {\mathbf{x}}\log p_ {t}(\mathbf{x}) \Vert _ 2^2 \big]\bigg]
$$

The expectation over $\mathbf{x}_ 0$ can be estimated by samples from our original data distribution $p_ 0$. <br>
The expectation over $\mathbf{x}_ t$ can be estimated by samples from the pertubated data distribution $p_ {0t}$. <br>
And $\lambda(t)$ is the weighting function that can be used to assign different importance to different timesteps. In the case of our VESDE we set it to $\lambda(t) = \sigma_t^2$. <br>
<br>
The training process follows these steps:

1. Pick a datapoint $\mathbf{x}_0$

2. Sample $\mathbf{x}_1 \sim \mathcal{N}(\mathbf{x}|0,\mathbf{I})$

3. Sample $t \sim \text{Uniform}(0,1)$

4. Calculate $\mathbf{x}_ t = \mathbf{x}_ 0 + \sqrt{\frac{1}{2 \ln \sigma}(\sigma^{2t}-1)} \cdot \mathbf{x}_ 1$. This is a sample from $p_ {0t}(\mathbf{x}_ t|\mathbf{x}_ 0)$

5. Evaluate the score model at $\mathbf{x}_ t$ and $t$, $s_ {\theta}(\mathbf{x}_ t,t)$

6. Calculate the score matching loss for a single sample: $\mathcal{L}_ t(\theta) = \sigma_ t^2 \Vert \mathbf{x}_ 1-\sigma_ t s_ {\theta}(\mathbf{x}_ t,t) \Vert ^2$

7. Update $\theta$ using gradient-based method with $\nabla_ {\theta}\mathcal{L}(\theta)$

### Sampling

We use the Euler-Maruyama method to solve the SDE. 
The Euler-Maruyama method is a simple and widely used method to solve SDEs. 
It is a first-order method that approximates the solution of an SDE by discretizing the time interval into small steps. <br>

$$
\mathbf{x}_ {t+1} = \mathbf{x}_ t + \mathbf{f}(\mathbf{x}_ t,t) \Delta t + g(t) \Delta \mathbf{w}
$$

where 

$$
\Delta \mathbf{w} = \mathbf{w}_ {t+1} - \mathbf{w}_ t = s_ {\theta}(\mathbf{x}_ t,t)
$$

We can then rewrite the denoising step at time $t$ as:

$$
\mathbf{x}_ {t-1} = \mathbf{x}_ t - \frac12 \sigma^{2t} s_ {\theta}(\mathbf{x}_ t,t)dt
$$


Basically we take a sample $x_T$ from the prior distribution $p_ T$ and give it to the transformer to get the score $s_ {\theta}(\mathbf{x}_ T,T)$. <br>
With that we can calculate the noise that needs to be subtracted from the sample $x_T$, which returns a slightly denoised sample. <br>
These steps can be repeated to get a fully denoised sample $x_0$ at $t=0$. <br>

Distribution Denoising | Single Sample Denoising
:-------------------------:|:-------------------------:
![](plots/test_big.gif) | ![](plots/test_quiver.gif)

### Conditining 
Tell the model which values are observed and which are latent. <br>
> **_Note:_** Not sure how to do this yet

---
---
## Bayesian Model Comparison

We have different models $\mathcal{M}$, that can describe our system (different yield sets). Our goal is to infere which model $\mathcal{M}$ is best suited to describe the observations $x$. <br>
We use Bayes' theorem to compare the models.

$$ \begin{align*}
P(\theta|x;\mathcal{M}) &= \frac{P(x|\theta;\mathcal{M})P(\theta|\mathcal{M})}{P(x|\mathcal{M})} = \frac{\mathcal{L}(\theta) \cdot \pi(\theta)}{z} \\ \\
\text{Posterior} &= \frac{\text{Likelihood} \times \text{Prior}}{\text{Evidence}}
\end{align*} $$

A crucial part of the model comparison is the computation of the evidence or marginal likelihood $z$. The evidence is the probability of  observing the data over all parameters $\theta$ and can be computed by marginalizing the likelihood over the prior. <br>

$$
P(\text{data}) = \int P(\text{data}|\theta)P(\theta) d\theta
$$

### Learned Harmonic Mean Estimator
Unfortunately in a high-dimensional parameter space, the evidence is computationally intractable. <br>
The learned harmonic mean estimator gives us a method to estimate the evidence for that case with the use of importance sampling. <br> 

We utilize the fact that the harmonic mean of the likelihoods is the reciprocal of the evidence. <br>
The harmonic mean estimator is given by [[Newton & Raftery 1994](https://www.jstor.org/stable/2346025)]:

$$ \begin{align*}
\rho &= \mathbb{E}_ {P(\theta|x)} [\frac1{\mathcal{L}(\theta)}] \\
 &= \int d\theta \frac1{\mathcal{L}(\theta)}P(\theta|x) \\
 &= \int d\theta \frac1{\mathcal{L}(\theta)}\frac{\mathcal{L}(\theta) \pi(\theta)}{z} \\
 &= \frac1z \\
=> \hat\rho &=\frac1N \sum_{i=1}^{N} \frac1{\mathcal{L}(\theta)}
\end{align*} $$

If we treat this as an importance sampling problem, we can elimate the problem of an exploding variance. <br>
Therefore we introduce a new target distribution $\phi(\theta)$. <br>
The harmonic mean estimator can then be re-written as [[Mancini et al. 2023](https://academic.oup.com/rasti/article/2/1/710/7382245#supplementary-data)]:

$$
\rho = z^{-1} = \frac1N \sum_ {i=1}^{N} \frac{\phi(\theta_ i)}{\mathcal{L}(\theta_ i)\pi(\theta_ i)}, \quad \text{where } \theta_ i \sim p(\theta|\text{data})
$$

Where we sample from the posterior distribution and calculate the likelihood of the sample.<br>
The ideal target distribution $\phi(\theta)$ is the posterior distribution $p(\theta|\text{data})$ which we can learn using SBI, like described in the previous section. 
<br>

> **_NOTE:_** The paper states, that the target distribution $\phi(\theta)$ should not be the learned NPE, but instead be a different model, that however approximates the posterior aswell. <br>
 Honestly I'm not really sure yet what this means! <br>

Now the benefit of using a diffusion model for SBI comes into play. 
The diffusion model from the previous section can be used to estimate the likelihood $\mathcal{L}(\theta)$
 and the posterior $\phi(\theta)$ of our system, 
 meaning it is just needed to train one model to estimate the evidence $z$. <br>

### Evidence Calculation

1. Take observation sample $x_i$

2. Sample posterior $\theta_{i;j} \sim P_j(\theta|x_i)$ using the diffusion model for each model $\mathcal{M_j}$

3. Evaluate likelihood at sample position $\mathcal{L}_j(\theta_ {i;j})$ also using the diffusion model

$=>$  Compute evidence $z$ by repeating $1.-3.$ $N$-times and then using the harmonic mean estimator $\hat \rho = \frac1N \sum_{i=1}^{N} \frac{\phi(\theta_i)}{\mathcal{L}(\theta_i)\pi(\theta_i)}$

### Model Comparison
To predict the best fitting model, we use Bayes update rule to calculate the posterior probability of each model. <br>

$$ \begin{align*}
P(\mathcal{M}_ j|x) &= \frac{P(x|\mathcal{M}_ j) P(\mathcal{M}_ j)}{P(x)} \\\\
&= \frac{\mathcal{L}_ j(\theta_ {i;j}) \cdot \pi(\mathcal{M}_ j)}{z}
\end{align*} $$

We start with a uniform prior over the models. By evaluating multiple observations, the posterior probability of each model can be updated. <br>
The model with the highest posterior probability is the best fitting model. <br>
