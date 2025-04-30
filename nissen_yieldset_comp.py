from compass import ModelTransfuser
from compass import ScoreBasedInferenceModel as SBIm

import torch
import numpy as np
import os
import time

if __name__ == "__main__":
    start = time.time()

    MTf = ModelTransfuser(path="data/big_MTf_model_comp/")

    # -----------------------------------------
    # Load in the models
    names = [name for name in os.listdir("data/big_MTf_model_comp/") if "checkpoint" in name]

    for name in names:
        model = SBIm.load("data/big_MTf_model_comp/" + name, device="cuda")
        model_name = name.replace("_checkpoint.pt", "")
        MTf.add_model(model_name, model)

    # -----------------------------------------
    # Load in the data
    data = np.load("data/nissen_gce_data/nissen_solar.npy").T
    data_std = np.load("data/nissen_gce_data/nissen_solar_std.npy").T

    # Remove star 4, because time is negative
    data = np.delete(data, 4, axis=0)
    data_std = np.delete(data_std, 4, axis=0)

    # Convert to torch tensors
    data = torch.tensor(data, dtype=torch.float32)
    data_std = torch.tensor(data_std, dtype=torch.float32)

    # Nissen GCE provides Age, C, Fe, Mg, O and Si
    # The models are trained on more elements, so we need to provide a mask to tell the model which data is available
    condition_mask = torch.tensor([0,0,0,0,0,1,1,1,0,1,0,0,1,1], dtype=torch.float32)

    # ------------------------------------------
    # Run the comparison
    print()
    print("Running comparison...")
    MTf.compare(x=data, err=data_std, condition_mask=condition_mask, device="cuda")
    MTf.plots()

    # ------------------------------------------
    end = time.time()
    print("Time taken: ", end-start)
    