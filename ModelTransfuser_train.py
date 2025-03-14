from src.ModelTransfuser_cfg import *
import matplotlib.pyplot as plt

from scipy.stats import norm
import numpy as np
import torch
from torch.utils.data import DataLoader
import os

def load_data(batch_size=64):
    # -------------------------------------
    # Load data

    # --- Load in training data ---
    path_training = os.getcwd() + '/data/Chempy_data/chempy_TNG_train_data.npz'
    training_data = np.load(path_training, mmap_mode='r')

    elements = training_data['elements']
    train_x = training_data['params']
    train_y = training_data['abundances']


    # ---  Load in the validation data ---
    path_test = os.getcwd() + '/data/Chempy_data/chempy_TNG_val_data.npz'
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

    # --- add noise ---
    pc_ab = 5 # percentage error in abundance

    train_y_err = torch.ones_like(train_y)*float(pc_ab)/100.
    train_y = norm.rvs(loc=train_y,scale=train_y_err)
    train_y = torch.tensor(train_y).float()

    val_y_err = torch.ones_like(val_y)*float(pc_ab)/100.
    val_y = norm.rvs(loc=val_y,scale=val_y_err)
    val_y = torch.tensor(val_y).float()

    # --- Concatenate the data ---
    train_data = torch.cat((train_x, train_y), 1)
    val_data = torch.cat((val_x, val_y), 1)

    train_data = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    val_data = DataLoader(val_data, shuffle=True, batch_size=batch_size)

    return train_data, val_data

# -------------------------------------

if __name__ == "__main__":
    # Load data
    batch_size = 1028
    train_data, val_data = load_data(batch_size=batch_size)

    # Setup model
    model = ModelTransfuser(14, depth=8, hidden_size=96, mlp_ratio=8, sigma=10)

    device = torch.device("cuda:0")

    # Train model
    model.train(train_data, val_data=val_data, batch_size=batch_size, cfg_prob=0.24, device=device, checkpoint_path="data/models/optuna_3_29/")
    model.save("data/models/cfg2/ModelTransfuser_cfg.pickle")

    epoch = np.arange(0, len(model.train_loss))

    plt.plot(epoch, np.array(model.train_loss)/train_data.shape[0], label='Train Loss')
    plt.plot(epoch, np.array(model.val_loss)/val_data.shape[0], label='Val Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.savefig('plots/loss/ModelTransfuser_train_loss_cfg2.png')