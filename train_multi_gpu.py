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

def load_data():
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

    return train_data, val_data

def ddp_main(gpu, world_size, batch_size, sigma, depth, hidden_size, num_heads, mlp_ratio, cfg_prob, path):
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
    model.module.train(train_loader, val_data=val_loader, batch_size=batch_size, device=device, cfg_prob=cfg_prob,
              checkpoint_path=path, verbose=(rank==0))

    if rank == 0:
        model.module.save(path+"Model.pickle")
        # Plot loss (assumes loss lists collected in model.module.train_loss)
        epoch = np.arange(0, len(model.module.train_loss))
        plt.plot(epoch, np.array(model.module.train_loss)/len(train_dataset), label='Train Loss')
        plt.plot(epoch, np.array(model.module.val_loss)/len(val_dataset), label='Val Loss')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(path+'Loss.png')
    
    dist.destroy_process_group()

if __name__ == "__main__":
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--sigma', type=float, default=25.0)
    parser.add_argument('--depth', type=int, default=6)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--num_heads', type=int, default=16)
    parser.add_argument('--mlp_ratio', type=int, default=4)
    parser.add_argument('--cfg_prob', type=float, default=0.2)
    parser.add_argument('--path', type=str, default='data/models/')
    # Detect available GPUs as default
    available_gpus = ','.join(map(str, range(torch.cuda.device_count())))
    parser.add_argument('--gpus', type=str, default=available_gpus, help=f'comma-separated list of GPU ids to use (default: {available_gpus})')
    
    args = parser.parse_args()

    batch_size = args.batch_size
    sigma = args.sigma
    depth = args.depth
    hidden_size = args.hidden_size
    num_heads = args.num_heads
    mlp_ratio = args.mlp_ratio
    cfg_prob = None if args.cfg_prob == 0 else float(args.cfg_prob)
    path = args.path

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    world_size = len(args.gpus.split(','))

    mp.spawn(ddp_main, args=(world_size, batch_size, sigma, depth, hidden_size, num_heads, mlp_ratio, cfg_prob, path), nprocs=world_size)
