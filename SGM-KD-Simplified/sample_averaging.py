#!/usr/bin/env python3

import argparse
import glob
import random
import json
import time
from pathlib import Path
import nibabel as nib
import os
from collections import defaultdict
import numpy as np
import accelerate
import safetensors.torch as safetorch
import torch
from tqdm import trange, tqdm
from torch.utils.data.dataloader import default_collate
from torch.utils import data
import k_diffusion as K
import matplotlib.pyplot as plt


LOWDOSE_SUBJECT = [
    'case_0142',
    'Fdg_Stanford_006',
    'case-0290',
    'Fdg_Stanford_002',
    'case-0285',
    'case_0254',
    'case_0119',
    'case_0233',
    'case_0234',
    '24521',
    '24102',
    '26097',
    '25501',
    '24921',
    '26233',
    '26053',
    '25654',
    '24849',
]

# LOWDOSE_SUBJECT = [
#     '26281', "26330", "26329"
# ]

seed = 42
torch.manual_seed(seed) 

@torch.no_grad()
def batch2avgvolume(batch_img, device, pad=True, pad_val=0):
    batch_size, ch_size, H, W = batch_img.shape
    radius = ch_size // 2
    
    # batch shape to averaged volume
    padded_size = batch_size + (2 * radius)
    averaged_volume = torch.zeros((ch_size, padded_size, H, W), device=device)
    dup_slices = torch.ones(padded_size, dtype=torch.int32, device=device) * ch_size 

    for ch in range(ch_size):
        averaged_volume[ch, ch:ch+batch_size] = batch_img[:,ch]
        dup_slices[ch] = ch + 1
        dup_slices[-ch-1] = ch + 1

    # B+2*radious, H, W
    print(dup_slices[0], dup_slices[1], dup_slices[2])
    averaged_volume = torch.sum(averaged_volume, dim=0, keepdim=False) / dup_slices[:, None, None]

    if not pad:
        averaged_volume = averaged_volume[radius:-radius]
    return averaged_volume

@torch.no_grad()
def volume2batch(volume_img, batch_img_shape, device):
    batch_size, ch_size, H, W = batch_img_shape
    radius = ch_size // 2
    padded_size = batch_size + (2 * radius)

    # averaged volume to batch shape
    double_padded_size = batch_size + (4 * radius)
    batch_img = torch.zeros((ch_size, double_padded_size, H, W), device=device)
    # batch_img = batch_img.to(device, non_blocking=True)
    for ch in range(ch_size-1, -1, -1):
        batch_img[ch_size-1-ch, ch:ch+padded_size] = volume_img
    
    batch_img = batch_img[:, ch_size-1:ch_size-1+batch_size].permute(1, 0, 2, 3) # B, C, H, W
    return batch_img

def correction(preds):
    pass


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--batch-size', type=int, default=64,
                   help='the batch size')
    p.add_argument('--checkpoint', type=Path, required=True,
                   help='the checkpoint to use')
    p.add_argument('--contrast', nargs='+', default=['T1'],
                   help='contrast')
    p.add_argument('--block_size', type=int, default=1,
                   help='block size')
    p.add_argument('--config', type=Path,
                   help='the model config')
    p.add_argument('-n', type=int, default=64,
                   help='the number of images to sample')
    p.add_argument('--prefix', type=str, default='out',
                   help='the output prefix')
    p.add_argument('--steps', type=int, default=50,
                   help='the number of denoising steps')
    p.add_argument('--nii_save_dest', type=str)
    p.add_argument('--num_correction', type=int, default=1)
    p.add_argument('--correction_coeff', type=float, default=1.0)
    p.add_argument('--output_slice', required=True)
    p.add_argument('--test_brats', action='store_true')
    p.add_argument('--random_avg', action='store_true')
    p.add_argument('--num_sample', type=int, default=None)
    p.add_argument('--sampler', type=str, default='heun')

    args = p.parse_args()

    config = K.config.load_config(args.config if args.config else args.checkpoint)
    model_config = config['model']
    # TODO: allow non-square input sizes
    assert len(model_config['input_size']) == 2 and model_config['input_size'][0] == model_config['input_size'][1]
    size = model_config['input_size']

    accelerator = accelerate.Accelerator()
    device = accelerator.device
    print('Using device:', device, flush=True)

    inner_model = K.config.make_model(config).eval().requires_grad_(False).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    inner_model.load_state_dict(ckpt['model'])
    print(f"Inner model loaded from {args.checkpoint}")

    accelerator.print('Parameters:', K.utils.n_params(inner_model))
    model = K.Denoiser(inner_model, sigma_data=model_config['sigma_data'])

    sigma_min = model_config['sigma_min']
    sigma_max = model_config['sigma_max']

    # test_dataset = K.utils.PETMRTestBraTS(test_files=['BraTS20_Training_001', "BraTS20_Training_005", "BraTS20_Training_010"],
    #                                       contrast_list=args.contrast,
    #                                       normalization_method="-1-1",
    #                                       slice_offset=args.block_size)

    test_dataset = K.utils.PETMRTestClassEmb("/data/jiaqiw01/PET_MRI/data/all_cases_rawspace_complete_max.h5",
        subjects=LOWDOSE_SUBJECT,
        img_size=256,
        normalization_method="-1-1", 
        slice_offset=args.block_size, 
        contrast_list=args.contrast, 
        output_slice=args.output_slice,
        distance=2
    )
    test_dl = None

    # test_dl = data.DataLoader(test_dataset, 1, shuffle=False, drop_last=False,
    #                         num_workers=0, persistent_workers=False, pin_memory=True)
    
    starting_noise = torch.randn([1, model_config['input_channels'], size[0], size[1]], device=device) * sigma_max

    batch_dict = defaultdict(list)
    for idx in range(len(test_dataset)):
        subj_id = test_dataset[idx]['case_name']
        # print(pid)
        slc_idx = test_dataset[idx]['slice_idx']
        batch_dict[subj_id].append(test_dataset[idx])

    @torch.no_grad()
    @K.utils.eval_mode(model)
    def run(test_dl, batch_size=1):
        if accelerator.is_local_main_process:
            tqdm.write('Sampling...')
        sigmas = K.sampling.get_sigmas_karras(args.steps, sigma_min, sigma_max, rho=7., device=device)

        def sample_fn_no_correction(n, unet_cond, sub_batch, num_iter):
            x_t = starting_noise
            # x_t = torch.randn([n, model_config['input_channels'], size[0], size[1]], device=device) * sigma_max
            print(x_t.shape) # 64, 3, 256, 256
            x0_recons = []
            # shape of x_t: n, 3, 256, 256
            for s in range(num_iter):
                start_idx = s * sub_batch
                end_idx = min((s + 1) * sub_batch, full_batch)
                x_t_sub = x_t[start_idx:end_idx]
                unet_cond_sub = unet_cond[start_idx:end_idx]
                extra_args = {'unet_cond': unet_cond_sub}
                for i in trange(len(sigmas) - 1):
                    x_t_sub, score = K.sampling.sample_heun_onestep(model, x_t_sub, i, sigmas, extra_args=extra_args)            # x_0 = K.sampling.sample_lms(model, x, sigmas, extra_args, disable=not accelerator.is_local_main_process)
                x0_recons.append(x_t_sub) # x_t_sub: 12, 3, 256, 256

            x0_recon = torch.cat(x0_recons, dim=0) # 64, 3, 256, 256
            return x0_recon
        
        def sample_fn_moving_window(n, unet_cond, sub_batch, num_iter, num_correction=1, correction_coeff=1.0, ws=3):
            x_t = torch.randn([n, model_config['input_channels'], size[0], size[1]], device=device) * sigma_max
            # x = starting_noise.clone()
            print(x_t.shape) # 64, 3, 256, 256
            avg_noise = None
            for i in trange(len(sigmas) - 1):
                x0_recons = []
                scores = []
                # shape of x_t: n, 3, 256, 256
                for s in range(n - ws + 1):
                    start_idx = s
                    end_idx = s + ws
                    x_t_sub = x_t[start_idx:end_idx] # 3, 1, 256, 256
                    unet_cond_sub = unet_cond[start_idx:end_idx]
                    extra_args = {'unet_cond': unet_cond_sub}
                    # x_t_sub, score = K.sampling.sample_heun_onestep(model, x_t_sub, i, sigmas, extra_args=extra_args)            # x_0 = K.sampling.sample_lms(model, x, sigmas, extra_args, disable=not accelerator.is_local_main_process)
                    x_t_sub, score = K.sampling.sample_dpmpp_2m_sde_onestep(model, x_t_sub, sigmas, i, extra_args=extra_args, eta=0.0, solver_type='heun')
                    x0_recons.append(x_t_sub.squeeze()) # x_t_sub: 3, 1, 256, 256 -> 3, 256, 256
                    scores.append(score.squeeze())

                s_tmin=0
                s_tmax=float('inf')
                gamma = min(0 / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
                sigma_hat = sigmas[i] * (gamma + 1)
                dt = sigmas[i + 1] - sigma_hat

                x0_recon = torch.stack(x0_recons, dim=0) # 86, 3, 256, 256
                scores = torch.stack(scores, dim=0)
                print(x0_recon.shape)
                avg_noise = batch2avgvolume(x0_recon, device) # 91, 3, 256, 256
                print("===== Avg Noise, should be 91, 256, 256 ======")
                print(avg_noise.shape)
                x_t = volume2batch(avg_noise, x_t.shape, device)
                print("Back to batch")
                print(x_t.shape)
                batch_size, ch_size, H, W = x_t.shape
                dim = torch.sqrt(torch.tensor(H * W, dtype=torch.float32, device=device))
                sig_t = sigmas[i]
                print(f"Applying volume-wise correction for {num_correction} steps...")
                for m in range(num_correction):
                    # average scores
                    score_volume = batch2avgvolume(scores, device, pad=True, pad_val=0)
                    score_l2_norm_squared = torch.sum(torch.pow(score_volume, 2), dim=(1, 2), keepdim=True)
                    gamma = correction_coeff * (dim / score_l2_norm_squared) * dt
                    # print(gamma)
                    gamma_score_batch = volume2batch(gamma * score_volume, x_t.shape, device)
                    x_t += gamma_score_batch

            return x_t

        def sample_fn_score_norm(n, unet_cond, sub_batch, num_iter, num_correction=1, correction_coeff=1.0, batch_size=16):
            x_t = torch.randn([n, model_config['input_channels'], size[0], size[1]], device=device) * sigma_max
            # x = starting_noise.clone()
            print(x_t.shape) # 64, 3, 256, 256
            avg_noise = None
            for i in trange(len(sigmas) - 1):
                x0_recons = []
                scores = []
                # shape of x_t: n, 3, 256, 256
                for s in range(n - batch_size + 1):
                    start_idx = s
                    end_idx = s + batch_size
                    x_t_sub = x_t[start_idx:end_idx] # 3, 1, 256, 256
                    unet_cond_sub = unet_cond[start_idx:end_idx]
                    extra_args = {'unet_cond': unet_cond_sub}
                    # x_t_sub, score = K.sampling.sample_heun_onestep(model, x_t_sub, i, sigmas, extra_args=extra_args)            # x_0 = K.sampling.sample_lms(model, x, sigmas, extra_args, disable=not accelerator.is_local_main_process)
                    x_t_sub, score = K.sampling.sample_dpmpp_2m_sde_onestep(model, x_t_sub, sigmas, i, extra_args=extra_args, eta=0.0, solver_type='heun')
                    x0_recons.append(x_t_sub) # x_t_sub: 16, 1, 256, 256
                    scores.append(score) # score: 16, 1, 256, 256

                s_tmin=0
                s_tmax=float('inf')
                gamma = min(0 / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
                sigma_hat = sigmas[i] * (gamma + 1)
                dt = sigmas[i + 1] - sigma_hat
            
                x0_recon = torch.stack(x0_recons, dim=0) # 89, 1, 256, 256 -> average
                x0_recon = (x0_recon - x0_recon.min()) / (x0_recon.max() - x0_recon.min()) * 2 - 1 
                scores = torch.stack(scores, dim=0) 
                scores = (scores - scores.min()) / (scores.max() - scores.min()) * 2 - 1 # so score is between -1 to 1..?
                
                assert x0_recon.shape == x_t.shape

                batch_size, ch_size, H, W = x_t.shape
                dim = torch.sqrt(torch.tensor(H * W, dtype=torch.float32, device=device))
                sig_t = sigmas[i]
                print(f"Applying volume-wise correction for {num_correction} steps...")
                for m in range(num_correction):
                    # average scores
                    score_l2_norm_squared = torch.sum(torch.pow(scores, 2), dim=(2, 3), keepdim=True) # 89, 1, 1, 1
                    gamma = correction_coeff * (dim / score_l2_norm_squared) * dt
                    # print(gamma)
                    gamma_score_batch = gamma * scores
                    x_t += gamma_score_batch

            return x_t

        @torch.no_grad()
        def sample_fn(n, unet_cond, class_cond, sub_batch, num_iter, num_correction=1, correction_coeff=1.0, random_avg=False, num_sample=None, sampler='heun'):
            x_t = torch.randn([n, model_config['input_channels'], size[0], size[1]], device=device) * sigma_max
            # x_t = starting_noise.repeat(89, 1, 1, 1)
            print(x_t.shape) # 89, 3, 256, 256
            avg_noise = None
            if random_avg:
                random_idxes = random.sample(range(len(sigmas)), num_sample)

            for i in trange(len(sigmas) - 1):
                x0_recons = []
                scores = []
                # shape of x_t: n, 3, 256, 256
                for s in range(num_iter):
                    start_idx = s * sub_batch
                    end_idx = min((s + 1) * sub_batch, full_batch)
                    x_t_sub = x_t[start_idx:end_idx]
                    unet_cond_sub = unet_cond[start_idx:end_idx]
                    class_cond_sub = class_cond[start_idx:end_idx]
                    extra_args = {'unet_cond': unet_cond_sub,
                                  'class_cond': class_cond_sub}
                    if sampler == 'heun':
                        x_t_sub, score = K.sampling.sample_heun_onestep(model, x_t_sub, i, sigmas, extra_args=extra_args, s_noise=1.)            # x_0 = K.sampling.sample_lms(model, x, sigmas, extra_args, disable=not accelerator.is_local_main_process)
                    elif sampler == 'dpm':
                        x_t_sub, score = K.sampling.sample_dpmpp_2m_sde_onestep(model, x_t_sub, sigmas, i, extra_args=extra_args, eta=0.0, solver_type='heun')
                    else:
                        raise NotImplementedError
                    x0_recons.append(x_t_sub) # x_t_sub: 12, 3, 256, 256
                    scores.append(score)

                x0_recon = torch.cat(x0_recons, dim=0) # 89, 3, 256, 256
                scores = torch.cat(scores, dim=0) # 89, 3, 256, 256
                # print(x0_recon.shape)

                if (random_avg and i in random_idxes) or (not random_avg):
                    s_tmin=0
                    s_tmax=float('inf')
                    gamma = min(0 / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
                    sigma_hat = sigmas[i] * (gamma + 1)
                    dt = sigmas[i + 1] - sigma_hat
                    avg_noise = batch2avgvolume(x0_recon, device) # 91, 3, 256, 256
                    # print("===== Avg Noise, should be 91, 256, 256 ======")
                    # print(avg_noise.shape)
                    x_t = volume2batch(avg_noise, x_t.shape, device)
                    # print("Back to batch")
                    # print(x_t.shape)
                    batch_size, ch_size, H, W = x_t.shape
                    dim = torch.sqrt(torch.tensor(H * W, dtype=torch.float32, device=device))
                    print(f"Applying volume-wise correction for {num_correction} steps...")
                    for m in range(num_correction):
                        # average scores
                        score_volume = batch2avgvolume(scores, device, pad=True, pad_val=0)
                        score_l2_norm_squared = torch.sum(torch.pow(score_volume, 2), dim=(1, 2), keepdim=True)
                        gamma = correction_coeff * (dim / score_l2_norm_squared) * dt
                        # print(gamma)
                        gamma_score_batch = volume2batch(gamma * score_volume, x_t.shape, device)
                        x_t += gamma_score_batch
                else:
                    x_t = x0_recon

            return x_t
        
        # test_subj = {}
        # if args.test_brats:
        #     target_length = 155
        # else:
        #     target_length = 89

        # separate test dl by subject id
        # reformat this just in case we forgot to change the destination name!
        cont = '_'.join(args.contrast)
        if args.random_avg:
            args.nii_save_dest = f"/data/jiaqiw01/debug_tests/kdiff/{cont}_3class_v1_2neighbor_step{args.steps}_random_{args.num_sample}_{args.sampler}"
        else:
            args.nii_save_dest = f"/data/jiaqiw01/debug_tests/kdiff/{cont}_3class_v1_2neighbor_step{args.steps}_full_avg_{args.sampler}"
            
        total_time = 0
        for subj in batch_dict.keys():
            start_time = time.time()
            test_batch = default_collate(batch_dict[subj])
            inputs = test_batch
            dest = os.path.join(args.nii_save_dest, subj)
            if os.path.exists(dest + '/pred.nii'):
                print(f"Skipping subject {subj}")
                continue
            unet_cond = inputs['lr'].to(device)  
            class_cond = inputs['label'].to(device)
            # print(unet_cond.shape)
            sub_batch = 8
            full_batch = unet_cond.shape[0]
            num_iter = int(np.ceil(full_batch / sub_batch))
            # final_recon_vol = sample_fn_moving_window(full_batch, unet_cond, sub_batch, num_iter, num_correction=args.num_correction, correction_coeff=args.correction_coeff, batch_size=16)
            # final_recon_vol = sample_fn_moving_window(full_batch, unet_cond, sub_batch, num_iter, num_correction=args.num_correction, correction_coeff=args.correction_coeff, ws=3)
            final_recon_vol = sample_fn(full_batch, unet_cond, class_cond, sub_batch, num_iter, num_correction=args.num_correction, correction_coeff=args.correction_coeff, random_avg=args.random_avg, num_sample=args.num_sample)
            # final_recon_vol = sample_fn_no_correction(full_batch, unet_cond, sub_batch, num_iter)
            end_time = time.time()
            duration = (int)(end_time - start_time)
            total_time += duration
            print(f"Finished sampling one subject {subj}")
            mid_slice = model_config['input_channels'] // 2
            # assert mid_slice == 1
            pred = final_recon_vol.cpu().numpy()[:, mid_slice, :, :] # 89/86?, 256, 256
            pred = np.transpose(pred, (1, 2, 0))
            print("====== Final volume shape =======")
            print(pred.shape)
            if len(subj) == 5: # new cases
                target_path = os.path.join("/data/jiaqiw01/preprocessed_cases", subj, "PET_full.nii")
            else: # old cases
                target_path = glob.glob(os.path.join("/data/jiahong/data/FDG_PET_preprocessed", subj, "reslice_PET*.nii"))[0]
            
            target = nib.load(target_path)
            aff = target.affine
            pred_nib = nib.Nifti1Image(pred, affine=aff)
            os.makedirs(dest, exist_ok=True)
            nib.save(pred_nib, os.path.join(dest, 'pred.nii'))
        print("Average time per subject: ", total_time / len(LOWDOSE_SUBJECT))
        return total_time


    try:
        total_time = run(test_dl)
        sampling_config = {
            'step': args.steps,
            'channel': args.output_slice,
            'averaging': True,
            'random_average': True if args.random_avg else False,
            'num_average': args.num_sample if args.num_sample else args.steps,
            'contrasts': args.contrast,
            'num_correction': args.num_correction,
            'correction_coeff': args.correction_coeff,
            'sigma_min': sigma_min,
            'sigma_max': sigma_max,
            'ckpt_used': str(args.checkpoint),
            'model_config': str(args.config),
            'sampler': args.sampler,
            'average sampling time': total_time / len(LOWDOSE_SUBJECT)
        }
        with open(os.path.join(args.nii_save_dest, 'sampling_info.json'), 'w') as fw:
            json.dump(sampling_config, fw, indent=4)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
