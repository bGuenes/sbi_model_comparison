import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import argparse

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from src.ModelTransfuser_cfg import ModelTransfuser

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
# DDP 
def ddp_main(gpu, world_size, batch_size, max_epochs, sigma, depth, hidden_size, num_heads, mlp_ratio, cfg_prob, temp, result_dict):
    rank = gpu
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    
    device = torch.device(f'cuda:{gpu}')

    # Load dataset and setup DistributedSampler
    train_dataset, val_dataset = load_data()
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler)

    # Setup model and wrap with DDP
    model = ModelTransfuser(14, sigma=sigma, depth=depth, hidden_size=hidden_size, num_heads=num_heads, mlp_ratio=mlp_ratio)
    model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    
    # Train model
    model.module.train(train_loader, val_data=val_loader, batch_size=batch_size, max_epochs=max_epochs, device=device, cfg_prob=cfg_prob,
            checkpoint_path=None, verbose=(rank==0))

    if rank == 0:
        print(f"Model trained {batch_size}, {sigma}, {depth}, {hidden_size}, {num_heads}, {mlp_ratio}, {cfg_prob}, {temp}")
    
    torch.cuda.empty_cache()

    # Mask to evaluate Posterior
    mask = torch.zeros_like(val_dataset[0])
    mask[6:] = 1
    val_theta, val_x = val_dataset[:1000, :6], val_dataset[:1000, 6:]

    val_x_sampler = DistributedSampler(val_x, num_replicas=world_size, rank=rank, shuffle=False)
    val_x_DL = DataLoader(val_x, batch_size=1000, shuffle=False, sampler=val_x_sampler)

    theta_hat = model.module.sample(val_x_DL, condition_mask=mask, device=device, temperature=temp, verbose=(rank==0))
    theta_hat = theta_hat.mean(dim=1)[:,:6].contiguous()

    # Gather all theta_hat tensors from all GPUs
    gathered_theta_hat = [torch.zeros_like(theta_hat) for _ in range(world_size)]
    dist.all_gather(gathered_theta_hat, theta_hat)

    if rank == 0:
        gathered_theta_hat = [tensor.to("cpu") for tensor in gathered_theta_hat]
        theta_hat = torch.cat(gathered_theta_hat, dim=0)
        mse_posterior = torch.mean((val_theta - theta_hat)**2, dim=0)

        # measure tarp
        # Convert to numpy and add required dimension for bootstrap
        theta_hat_np = theta_hat.cpu().numpy()
        val_theta_np = val_theta.cpu().numpy()
        theta_hat_3d = np.expand_dims(theta_hat_np, 0)  # Add sample dimension
        
        ecp, alpha = tarp.get_tarp_coverage(
            theta_hat_3d, val_theta_np,
            norm=True, bootstrap=True,
            num_bootstrap=100
        )
        tarp_val = torch.mean(torch.from_numpy(ecp[:,ecp.shape[1]//2]))
        tarp_diff = abs(tarp_val-0.5)
        
        # Store results in shared dictionary
        result_dict['mse_posterior'] = mse_posterior
        result_dict['tarp_diff'] = tarp_diff

    dist.destroy_process_group()

# -------------------------------------
# Optuna
def objective(trial):

    try:
        # Variables
        batch_size = trial.suggest_categorical('batch_size', [64,128,512,1028])
        sigma = trial.suggest_float('sigma', 1.1, 30.0)
        depth = trial.suggest_int('depth', 1, 12)
        num_heads = trial.suggest_int('num_heads', 1, 32)
        hidden_size_factor = trial.suggest_int('hidden_size_factor', 1,20)
        hidden_size = num_heads*hidden_size_factor
        mlp_ratio = trial.suggest_int('mlp_ratio', 1, 10)
        cfg_prob = trial.suggest_float('cfg_prob', 0.0, 1.0)
        cfg_prob = None if cfg_prob == 0.0 else cfg_prob
        temp = trial.suggest_float('Temperature', 0.1, 10.0)

        result_dict = mp.Manager().dict()

        # Train model
        max_epochs = 500
        mp.spawn(ddp_main, args=(world_size, batch_size, max_epochs, sigma, depth, hidden_size, num_heads, mlp_ratio, cfg_prob, temp, result_dict), nprocs=world_size)

        # Get results from the shared dict
        mse_posterior = result_dict.get('mse_posterior')
        tarp_diff = result_dict.get('tarp_diff')

        return mse_posterior.mean().item(), tarp_diff.item()

    except Exception as e:
        print(f"Trial failed with error: {e}")
        # Return very bad scores for failed trials
        return float('inf'), float('inf')

# -------------------------------------
# Main
if __name__ == "__main__":
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

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
    study.optimize(objective, callbacks=[MaxTrialsCallback(100)])
    