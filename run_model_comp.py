from src.ModelTransfuser import ModelTransfuser

import numpy as np
import torch
from scipy.stats import norm
import os
import time

import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------------------
# Load data
# -----------------------------------------

def load_data(name):
    # --- Load in training data ---
    path_training = os.getcwd() + f'/data/model_comp_data(AGB,SN2,SN1a)/{name}_train.npz'
    training_data = np.load(path_training, mmap_mode='r')

    train_theta = training_data['params']
    train_x = training_data['abundances']

    # ---  Load in the validation data ---
    path_test = os.getcwd() + f'/data/model_comp_data(AGB,SN2,SN1a)/{name}_val.npz'
    val_data = np.load(path_test, mmap_mode='r')

    val_theta = val_data['params']
    val_x = val_data['abundances']

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

    train_theta, train_x = clean_data(train_theta, train_x)
    val_theta, val_x     = clean_data(val_theta, val_x)

    # convert to torch tensors
    train_theta = torch.tensor(train_theta, dtype=torch.float32)
    train_x = torch.tensor(train_x, dtype=torch.float32)
    val_theta = torch.tensor(val_theta, dtype=torch.float32)
    val_x = torch.tensor(val_x, dtype=torch.float32)

    # --- add noise ---
    pc_ab = 5 # percentage error in abundance

    train_x_err = torch.ones_like(train_x)*float(pc_ab)/100.
    train_x = norm.rvs(loc=train_x,scale=train_x_err)
    train_x = torch.tensor(train_x).float()

    val_x_err = torch.ones_like(val_x)*float(pc_ab)/100.
    val_x = norm.rvs(loc=val_x,scale=val_x_err)
    val_x = torch.tensor(val_x).float()

    # --- Concatenate the data ---
    # train_data = torch.cat((train_theta, train_x), 1)
    # val_data = torch.cat((val_theta, val_x), 1)

    return train_theta, train_x, val_theta, val_x


if __name__ == "__main__":
    start = time.time()
    # -----------------------------------------
    # Get the names of the data files
    names = [name.replace("_train.npz","") for name in os.listdir("data/model_comp_data(AGB,SN2,SN1a)/") if "train" in name]
    
    # -----------------------------------------
    # Init MTf
    MTf = ModelTransfuser()

    # -----------------------------------------
    # Load in data
    for name in names:
        # --- Load in the data ---
        train_theta, train_x, val_theta, val_x = load_data(name)

        MTf.add_data(name, train_theta, train_x, val_theta, val_x)

    # -----------------------------------------
    # Init model
    MTf.init_models(sde_type="vesde", sigma=2.5, hidden_size=36, depth=5, num_heads=1, mlp_ratio=3)

    # -------------------------------------------
    # Train model
    MTf.train_models(path="data/big_MTf_model_comp", batch_size=512, device="cuda")

    end = time.time()
    print(f"Time taken: {(end - start)/60:.2f} minutes")
    