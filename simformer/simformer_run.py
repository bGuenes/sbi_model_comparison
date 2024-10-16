from simformer import *
import matplotlib.pyplot as plt
import torch

# Load data
x = np.load("data/x.npy")
theta = np.load("data/theta.npy")


# Define beta schedule
T = 300

simformer = Simformer(T)

