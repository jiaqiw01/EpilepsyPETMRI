from src.samplers import BaseSampler, Euler_Maruyama_sde_predictor, Langevin_sde_corrector, soft_diffusion_momentum_sde_predictor
from src import (get_standard_sde, get_standard_score, PETMR, 
			score_model_simple_trainer)
from configs.default_config import get_default_configs
import torch 
import torchvision
import numpy as np 
import functools 
import matplotlib.pyplot as plt
from tqdm import tqdm 
from torch.optim import Adam
from torch.utils.data import DataLoader
import os

def visualize(sde, dataloader, save_root, eps=1e-5):

    """
    The loss function for training score-based generative models.
    Args:
        model: A PyTorch model instance that represents a 
        time-dependent score-based model.
        x: A mini-batch of training data.
        sde: the forward sde
        eps: A tolerance value for numerical stability.
    """
    for i, batch in enumerate(dataloader):
        if (i+1) % 10 == 0:
            mris = batch[:, 1:, ...]
            t1 = mris[:, [0], ...].squeeze().numpy()
            t2f = mris[:, [1], ...].squeeze().numpy()
            fname = f"t1_batch_{i}.png"
            plt.imsave(os.path.join(save_root, fname), t1, cmap='gray')
            fname = f"t2f_batch_{i}.png"
            plt.imsave(os.path.join(save_root, fname), t2f, cmap='gray')

            for t in [0.0001, 0.3, 0.5, 0.7, 0.9]:
                random_t = torch.tensor([t])
                x = batch[:, [0], ...]
                z = torch.randn_like(x)

                mean, std = sde.marginal_prob(x, random_t)  # for VESDE the mean is just x
                perturbed_x = mean + z * std[:, None, None, None]
                print(perturbed_x.shape)
                x_input = perturbed_x.squeeze().numpy()
                fname = f"noise_{t}_batch_{i}.png"
                print(x_input.shape)
                plt.imsave(os.path.join(save_root, fname), x_input, cmap='gray')

if __name__ == "__main__":
    config = get_default_configs()

    device = torch.device("cuda:2")

    sde = get_standard_sde(config)

    brain_dataset = PETMR(data_h5="/data/jiaqiw01/PET_MRI/data/all_cases_rawspace_complete_max_sdc_only.h5", 
                            slice_file="/data/jiaqiw01/PET_MRI/data/slices/finalized_val_slice.txt",
                            contrast_list=['T1', 'T2_FLAIR'],
                            slice_offset=0,
                            save_root=None,
                            aug=False)

    train_dl = DataLoader(brain_dataset,batch_size=1, num_workers=1)
    save = f"./noise_visualization_{config.sde.beta_min}_{config.sde.beta_max}"
    os.makedirs(save, exist_ok=True)
    visualize(sde, train_dl, save)


