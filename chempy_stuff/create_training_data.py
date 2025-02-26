import numpy as np
from scipy.stats import norm


import sbi.utils as utils
from sbi.utils.user_input_checks import check_sbi_inputs, process_prior, process_simulator
from sbi.inference import NPE, simulate_for_sbi

import torch
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform

import time as t
import pickle

# ----- Load the model -------------------------------------------------------------------------------------------------------------------------------------------
# --- Define the prior ---
# Elements to track
labels_out_H = ['C', 'Fe', 'H', 'He', 'Mg', 'N', 'Ne', 'O', 'Si']
labels_out = ['C', 'Fe', 'He', 'Mg', 'N', 'Ne', 'O', 'Si']

# Input parameters
labels_in = ['high_mass_slope', 'log10_N_0', 'log10_starformation_efficiency', 'log10_sfr_scale', 'outflow_feedback_fraction', 'time']
priors = torch.tensor([[-2.3000,  0.3000],
                       [-2.8900,  0.3000],
                       [-0.3000,  0.3000],
                       [ 0.5500,  0.1000],
                       [ 0.5000,  0.1000]])

combined_priors = utils.MultipleIndependent(
    [Normal(p[0]*torch.ones(1), p[1]*torch.ones(1)) for p in priors] +
    [Uniform(torch.tensor([2.0]), torch.tensor([12.8]))],
    validate_args=False)


# --- Set up the model ---
class Model_Torch(torch.nn.Module):
    def __init__(self):
        super(Model_Torch, self).__init__()
        self.l1 = torch.nn.Linear(len(labels_in), 100)
        self.l2 = torch.nn.Linear(100, 40)
        self.l3 = torch.nn.Linear(40, 9)

    def forward(self, x):
        x = torch.tanh(self.l1(x))
        x = torch.tanh(self.l2(x))
        x = self.l3(x)
        return x

model = Model_Torch()

# --- Load the weights ---
model.load_state_dict(torch.load('../data/pytorch_state_dict.pt'))
#model.load_state_dict(torch.load('sbi_chemical_abundances/data/pytorch_state_dict.pt'))
model.eval()


# ----- Create data -------------------------------------------------------------------------------------------------------------------------------------------
# --- Sample from the prior ---
def sample_from_prior(prior, num_samples):
    theta = prior.sample((num_samples,))
    return theta

theta = sample_from_prior(combined_priors, 100000)

# --- Simulate the data ---
def simulator(params):
    y = model(params)
    y = y.detach().numpy()

    # Remove H from data, because it is just used for normalization (output with index 2)
    y = np.delete(y, 2,1)

    return y

x = simulator(theta)

# --- Add noise ---
pc_ab = 5 # percentage error in abundance

x_err = np.ones_like(x)*float(pc_ab)/100.
x = norm.rvs(loc=x,scale=x_err)
x = torch.tensor(x).float()


# ----- Save the data -------------------------------------------------------------------------------------------------------------------------------------------
# --- Save the data ---
theta = theta.numpy()
x = x.numpy()

np.save('data/theta.npy', theta)
np.save('data/x.npy', x)

print("Data saved!")