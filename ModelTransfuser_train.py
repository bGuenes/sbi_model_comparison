from ModelTransfuser.ModelTransfuser import *
import matplotlib.pyplot as plt

import numpy as np
import torch
import os

# -------------------------------------
# Load data

# --- Load in training data ---
path_training = os.getcwd() + '/ModelTransfuser/data/chempy_TNG_train_data.npz'
training_data = np.load(path_training, mmap_mode='r')

elements = training_data['elements']
train_x = training_data['params']
train_y = training_data['abundances']


# ---  Load in the validation data ---
path_test = os.getcwd() + '/ModelTransfuser/data/chempy_TNG_val_data.npz'
val_data = np.load(path_test, mmap_mode='r')

val_x = val_data['params']
val_y = val_data['abundances']


# --- Clean the data ---
# Chempy sometimes returns zeros or infinite values, which need to removed
def clean_data(x, y):
    # Remove all zeros from the training data
    index = np.where((y == 0).all(axis=1))[0]
    x = np.delete(x, index, axis=0)
    y = np.delete(y, index, axis=0)

    # Remove all infinite values from the training data
    index = np.where(np.isfinite(y).all(axis=1))[0]
    x = x[index]
    y = y[index]

    # Remove H from Elements
    y = np.delete(y, 2, 1)

    return x, y


train_x, train_y = clean_data(train_x, train_y)
val_x, val_y     = clean_data(val_x, val_y)

# convert to torch tensors
train_x = torch.tensor(train_x, dtype=torch.float32)
train_y = torch.tensor(train_y, dtype=torch.float32)
val_x = torch.tensor(val_x, dtype=torch.float32)
val_y = torch.tensor(val_y, dtype=torch.float32)

train_data = torch.cat((train_x, train_y), 1)
val_data = torch.cat((val_x, val_y), 1)


# -------------------------------------
# Set up the model

# Define the ModelTransfuser

# Time steps for the diffusion process
T = 50
t = torch.linspace(0, 1, T)

ModelTransfuser = ModelTransfuser(T, train_data.shape)

ModelTransfuser.set_normalization(train_data)

# -------------------------------------
# Train

ModelTransfuser.train(train_data, val_data=val_data, epochs=10)

ModelTransfuser.save("ModelTransfuser/models/ModelTransfuser_t50_normed.pickle")

epoch = np.arange(0, len(ModelTransfuser.train_loss))

plt.plot(epoch, np.array(ModelTransfuser.train_loss)/train_data.shape[0], label='Train Loss')
plt.plot(epoch, np.array(ModelTransfuser.val_loss)/val_data.shape[0], label='Val Loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.savefig('ModelTransfuser_train_loss_normed.png')