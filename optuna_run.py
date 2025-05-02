import os
import numpy as np
from scipy.stats import norm
from scipy.stats import gaussian_kde
import argparse

import torch

from compass import ScoreBasedInferenceModel as SBIm

import tarp
import optuna
from optuna.study import MaxTrialsCallback

# -------------------------------------
# Load data
def load_data():
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

    return train_data, val_data

# -------------------------------------
# Optuna
def objective(trial):

    try:
        # Variables
        batch_size = trial.suggest_int('batch_size', 16,1024)
        sigma = trial.suggest_float('sigma', 1.1, 30.0)
        depth = trial.suggest_int('depth', 1, 12)
        num_heads = trial.suggest_int('num_heads', 1, 32)
        hidden_size_factor = trial.suggest_int('hidden_size_factor', 1,256)
        hidden_size = num_heads*hidden_size_factor
        mlp_ratio = trial.suggest_int('mlp_ratio', 1, 10)

        # Load data
        train_data, val_data = load_data()

        # Setup model
        nodes_size = train_data.shape[1]
        sbim = SBIm(nodes_size=nodes_size, sigma=sigma, depth=depth, hidden_size=hidden_size, num_heads=num_heads, mlp_ratio=mlp_ratio)

        # Train model
        sbim.train(train_data, val_data=val_data, batch_size=batch_size, max_epochs=500, device="cuda", verbose=True, early_stopping_patience=20)

        # Evaluate model
        mask = torch.zeros_like(val_data[0])
        mask[6:] = 1
        val_theta, val_x = val_data[:1000, :6], val_data[:1000, 6:]

        samples = sbim.sample(val_x, condition_mask=mask, verbose=True, num_samples=1000, timesteps=100, device="cuda")
        
        theta_hat = samples[:,:,:6].contiguous().cpu().numpy()
        val_theta = val_theta.cpu().numpy()

        # Log Prob
        def calc_log_prob(samples, theta):
            try:
                kde = gaussian_kde(samples.T)
                return kde.logpdf(theta).item()
            except:
                return -1e20
            
        log_probs = np.array([calc_log_prob(theta_hat[i], val_theta[i]) for i in range(len(theta_hat))])
        mean_log_prob = -np.mean(log_probs)

        # measure tarp
        ecp, alpha = tarp.get_tarp_coverage(
            theta_hat.transpose(1,0,2), val_theta,
            norm=True, bootstrap=True,
            num_bootstrap=100
        )
        tarp_diff = np.abs(ecp-np.linspace(0,1,ecp.shape[1])).max()

        return mean_log_prob, tarp_diff.item()

    except Exception as e:
        print(f"Trial failed with error: {e}")
        # Return very bad scores for failed trials
        return float('inf'), float('inf')

# -------------------------------------
# Main
if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()

    available_gpus = ','.join(map(str, range(torch.cuda.device_count())))
    parser.add_argument('--gpus', type=str, default=available_gpus, help=f'comma-separated list of GPU ids to use (default: {available_gpus})')
    
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    world_size = len(args.gpus.split(','))

    # Optuna
    study_name = 'ModelTransfuser_Chempy'  # Unique identifier of the study.
    storage_name = 'sqlite:///ModelTransfuser_Chempy.db'
    study = optuna.create_study(study_name=study_name, storage=storage_name,directions=['minimize', 'minimize'], load_if_exists=True)
    study = optuna.load_study(study_name=study_name, storage=storage_name)
    study.optimize(objective, callbacks=[MaxTrialsCallback(1000)])
    