#!/usr/bin/env python3

"""Samples from k-diffusion models."""

import argparse
import glob
from pathlib import Path
import nibabel as nib
import json
import os, time
import numpy as np
import accelerate
import safetensors.torch as safetorch
from collections import defaultdict
from torch.utils.data.dataloader import default_collate
import torch
from tqdm import trange, tqdm
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
    p.add_argument('--output_slice', type=int, required=True,
                   help='channel slices')
    p.add_argument('--config', type=Path,
                   help='the model config')
    p.add_argument('-n', type=int, default=64,
                   help='the number of images to sample')
    p.add_argument('--prefix', type=str, default='out',
                   help='the output prefix')
    p.add_argument('--steps', type=int, default=50,
                   help='the number of denoising steps')
    p.add_argument('--nii_save_dest', type=str)
    p.add_argument('--sampler', type=str, default='heun')

    args = p.parse_args()

    seed = 42
    torch.manual_seed(seed) 

    config = K.config.load_config(args.config if args.config else args.checkpoint)
    model_config = config['model']
    # TODO: allow non-square input sizes
    assert len(model_config['input_size']) == 2 and model_config['input_size'][0] == model_config['input_size'][1]
    size = model_config['input_size']

    accelerator = accelerate.Accelerator()
    device = accelerator.device
    print('Using device:', device, flush=True)

    inner_model = K.config.make_model(config).eval().requires_grad_(False).to(device)
    ckpt = torch.load(args.checkpoint, map_location='cpu')
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

    test_dataset = K.utils.PETMRTestClassEmb(
        "/data/jiaqiw01/PET_MRI/data/all_cases_rawspace_complete_max.h5",
        subjects=LOWDOSE_SUBJECT,
        img_size=256,
        normalization_method="-1-1", 
        slice_offset=args.block_size, 
        contrast_list=args.contrast, 
        output_slice=args.output_slice,
        distance=1
    )

    test_dl = None

    #starting_noise = torch.randn([1, model_config['input_channels'], size[0], size[1]], device=device) * sigma_max

    batch_dict = defaultdict(list)
    for idx in range(len(test_dataset)):
        subj_id = test_dataset[idx]['case_name']
        # print(pid)
        slc_idx = test_dataset[idx]['slice_idx']
        batch_dict[subj_id].append(test_dataset[idx])

    starting_noise = torch.randn([1, model_config['input_channels'], size[0], size[1]], device=device) * sigma_max

    @torch.no_grad()
    @K.utils.eval_mode(model)
    def run(test_dl, batch_size=1, num_slices=1):
        if accelerator.is_local_main_process:
            tqdm.write('Sampling...')
        sigmas = K.sampling.get_sigmas_karras(args.steps, sigma_min, sigma_max, rho=7., device=device)
        
        def sample_fn(n, unet_cond, class_cond, num_slices=1, sampler='heun'):
            # x = torch.randn([n, num_slices, size[0], size[1]], device=device) * sigma_max
            x = starting_noise.repeat(n, 1, 1, 1)
            extra_args = {'unet_cond': unet_cond,
                          'class_cond': class_cond}
            if sampler == 'heun':
                x_0 = K.sampling.sample_heun(model, x, sigmas, extra_args=extra_args, s_noise=1.)
            elif sampler == 'dpm':
                x_0 = K.sampling.sample_dpmpp_2m_sde(model, x, sigmas, extra_args=extra_args, eta=0.0, solver_type='heun')            # x_0 = K.sampling.sample_lms(model, x, sigmas, extra_args, disable=not accelerator.is_local_main_process)
            else:
                raise NotImplementedError
            return x_0
        total_time = 0
        for subj in batch_dict.keys():
            test_batch = default_collate(batch_dict[subj])
            inputs = test_batch
            dest = os.path.join(args.nii_save_dest, subj)
            unet_cond = inputs['lr'].to(device)  
            class_cond = inputs['label'].to(device)
            # print(unet_cond.shape)
            sub_batch = batch_size
            full_batch = unet_cond.shape[0]
            num_iter = int(np.ceil(full_batch / sub_batch))
            volume = []
            start_time = time.time()
            for i in range(num_iter + 1):
                start_idx = i * sub_batch
                end_idx = min((i + 1) * sub_batch, full_batch)
                bs = end_idx - start_idx
                if bs <= 0:
                    break
                unet_cond_sub = unet_cond[start_idx:end_idx]
                class_cond_sub = class_cond[start_idx:end_idx]
                sub_vol = sample_fn(bs, unet_cond_sub, class_cond_sub, num_slices=num_slices, sampler=args.sampler)
                if sub_vol.shape[1] != 1:
                    # multichannel sampling, choose middle one
                    sub_vol = sub_vol[:, num_slices//2, ...] # batch, 256, 256
                volume.append(sub_vol) # 16, 256, 256 -> 16, 256, 256
            volume = torch.concat(volume, dim=0) # 89
            total_time += (int)(time.time() - start_time)
            assert volume.shape == (89, 256, 256)
            pred = volume.cpu().numpy() # 89/86?, 256, 256
            pred = np.transpose(pred, (1, 2, 0))
            if len(subj) == 5: # new cases
                target_path = os.path.join("/data/jiaqiw01/preprocessed_cases", subj, "reslice_PET_full.nii")
            else: # old cases
                target_path = glob.glob(os.path.join("/data/jiahong/data/FDG_PET_preprocessed", subj, "reslice_PET*.nii"))[0]
            target = nib.load(target_path)
            aff = target.affine
            pred_nib = nib.Nifti1Image(pred, affine=aff)
            os.makedirs(dest, exist_ok=True)
            nib.save(pred_nib, os.path.join(dest, 'pred.nii'))
            print(f"Nii saved for subject {subj}")
        print("Average time per subject: ", total_time / len(LOWDOSE_SUBJECT))
        return total_time


    try:
        total_time = run(test_dl, batch_size=args.batch_size, num_slices=args.output_slice)
        sampling_config = {
            'step': args.steps,
            'channel': args.output_slice,
            'averaging': False,
            'contrast': args.contrast,
            # 'num_correction': args.num_correction,
            # 'correction_coeff': args.correction_coeff,
            'sigma_min': sigma_min,
            'sigma_max': sigma_max,
            'sampler': args.sampler,
            'average sampling time': total_time / len(LOWDOSE_SUBJECT)
        }
        with open(os.path.join(args.nii_save_dest, 'sampling_info.json'), 'w') as fw:
            json.dump(sampling_config, fw)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
