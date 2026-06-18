from contextlib import contextmanager
import hashlib
import math
from pathlib import Path
import shutil
import threading
import time
import urllib
import warnings
import os
import nibabel as nib
import glob
from PIL import Image
import safetensors
import torch
from torch import nn, optim
from torch.utils import data
from torchvision.transforms import functional as TF
from torchvision.transforms import v2
# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import torchvision.datasets as datasets
from torchvision import transforms
from torch.utils.data import Dataset

import random
import numpy as np
import torch
import h5py
from torch.utils.data import Dataset
from torchvision import transforms
# from torchvision.transforms import v2
import matplotlib.pyplot as plt

def random_rot(img1,img2):
    k = np.random.randint(0, 3)
    img1 = np.rot90(img1, k+1)
    img2 = np.rot90(img2, k+1)
    return img1,img2

def random_flip(img1,img2):
    axis = np.random.randint(0, 2)
    img1 = np.flip(img1, axis=axis).copy()
    img2 = np.flip(img2, axis=axis).copy()
    return img1,img2

class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        lr, hr = sample['lr'], sample['hr']

        if random.random() > 0.5:
            lr, hr = random_rot(lr, hr)
        if random.random() > 0.5:
            lr, hr = random_flip(lr, hr)
        sample = {'lr': lr,'hr': hr}
        return sample
    
class PETMRTestBraTS(Dataset):
    def __init__(self, test_files, src='/data/jiahong/data/BraTS/MICCAI_BraTS2020_TrainingData', img_size=256, slice_offset=1, num_ds=10, contrast_list=['T1_GRE', 'T2_FLAIR'], normalization_method="-1-1"):
        super().__init__()
        self.img_size = img_size
        self.test_files = test_files
        self.slice_offset = slice_offset
        self.contrast_list = contrast_list
        self.pad_val = -1 if normalization_method == "-1-1" else 0
        self.t1, self.t2f, self.seg = self.get_brats(src)
        print("======= BraTS Datasets =======")
        print(self.test_files)
        self.slices = []
        self.normalization_method = normalization_method
    
        for j in range(len(self.test_files)):
            for i in range(self.seg[0].shape[-1]):
                self.slices.append([j, i])

    def norm(self, img, by='max'):
        if by == 'max':
            img = (img - img.min()) / (img.max() - img.min())
            return img
        
    def get_brats(self, src):
        output_t1 = []
        output_t2f = []
        output_seg = []
        for ds in self.test_files:
            print(os.path.join(src, ds, "*t1.nii.gz"))
            t1_path = glob.glob(os.path.join(src, ds, "*t1.nii.gz"))[0]
            t2f_path = glob.glob(os.path.join(src, ds, "*t2.nii.gz"))[0]
            seg_path = glob.glob(os.path.join(src, ds, "*seg.nii.gz"))[0]
            t1 = nib.load(t1_path).get_fdata()
            t2f = nib.load(t2f_path).get_fdata()
            seg = nib.load(seg_path).get_fdata()
            # need to pad into 256, 256
            h, w, d = t1.shape
            h_pad = (256 - h)//2
            v_pad = (256 - w)//2
            pad_val = 0
            padded_t1 = self.norm(np.pad(t1, ((h_pad, h_pad), (v_pad, v_pad), (0, 0)), mode='constant', constant_values=self.pad_val))
            output_t1.append(padded_t1)
            padded_seg = np.pad(seg, ((h_pad, h_pad), (v_pad, v_pad), (0, 0)), mode='constant', constant_values=0)
            output_seg.append(padded_seg)
            padded_t2f = self.norm(np.pad(t2f, ((h_pad, h_pad), (v_pad, v_pad), (0, 0)), mode='constant', constant_values=self.pad_val))
            output_t2f.append(padded_t2f)
        return output_t1, output_t2f, output_seg
        
        
    def __len__(self):
        print(f"Total training slices: {len(self.slices)}")
        return len(self.slices)

    def __getitem__(self, idx):
        ds_idx, slice_idx = self.slices[idx]
        slc_idx_sel = np.clip(np.arange(slice_idx-self.slice_offset, slice_idx+self.slice_offset+1), 0, self.t1[0].shape[-1]-1)
        output  = {}
        imgs = []
        
        for contrast in self.contrast_list:
            if 'T1' in contrast:
                img = self.t1[ds_idx][:, :, slc_idx_sel]
            elif contrast == 'T2_FLAIR':
                img = self.t2f[ds_idx][:, :, slc_idx_sel]
            else:
                print(f"I don't know what contrast {contrast} this is")
                exit(0)
            imgs.append(img)

        # Concat inputs for multi-modal multi-slice models
        img = np.concatenate(imgs, axis=-1)
        img = np.transpose(img, (2, 0, 1))
        if self.normalization_method == "-1-1":
            inputs = torch.Tensor(img * 2 - 1).float()
        else:
            inputs = torch.Tensor(img).float()
            
        seg = self.seg[ds_idx][:, :, slice_idx:slice_idx+1]
        seg = np.transpose(seg, (2, 0, 1))

        output['lr'] = inputs
        output['seg'] = seg
        output['case_name'] = self.test_files[ds_idx]
        output['slice_idx'] = slice_idx
        return output


class PETMRTestClassEmb(Dataset):
    def __init__(self, data_h5, subjects, img_size=256, slice_offset=1, contrast_list=['T1_GRE', 'T2_FLAIR'], normalization_method="-1-1", output_slice=1, prior_volume_path=None, distance=1):
        super().__init__()
        self.data = h5py.File(data_h5, 'r')
        self.contrast_list = contrast_list
        self.slices = []
        self.subjects = subjects
        self.output_slice = output_slice
        self.distance = distance
        if len(contrast_list) > 1:
            self.multimodal = True
        else:
            self.multimodal = False
        with h5py.File(data_h5, "r") as f:
            self.subject_list = list(f.keys())
        for subj in subjects:
            assert subj in self.subject_list, f"This subject {subj} is not found in h5py... exiting.."
            for i in range(89):
                self.slices.append([subj, i])
        self.image_size = img_size
        self.slice_offset = slice_offset
        self.normalization_method = normalization_method
        self.prior = prior_volume_path
        self.prior_volumes = {}
        if self.prior:
            for subj in subjects:
                prior_path = os.path.join(self.prior, subj, 'pred.nii')
                assert os.path.exists(prior_path), f"Ops, prior volume not found for {subj}"
                prior_volume = nib.load(prior_path).get_fdata() # 256, 256, 89
                prior_volume = ((prior_volume - prior_volume.min()) / (prior_volume.max() - prior_volume.min())) * 2 - 1
                self.prior_volumes[subj]= prior_volume
        
    def __len__(self):
        print(f"Total training slices: {len(self.slices)}")
        return len(self.slices)

    def __getitem__(self, idx):
        subj = self.slices[idx][0]
        slice_idx = int(self.slices[idx][1]) # 3
        target_idx_sel = np.clip(np.arange(slice_idx - self.slice_offset, slice_idx + self.slice_offset + 1), 0, 88)        #     target_selection = slice_selection - 1
        slc_idx_sel = np.clip(np.arange(slice_idx - self.slice_offset - self.distance, slice_idx + self.slice_offset + 1 + self.distance), 0, 88)  # with neighbors..
        # print(slc_idx_sel, target_idx_sel)

        output  = {}
        imgs = []
        for contrast in self.contrast_list:
            if 'T1' in contrast:
                if subj+'/'+'T1_GRE' in self.data.keys():
                    img = np.array(self.data[subj+'/T1_GRE'])[:,:,slc_idx_sel]
                elif subj+'/T1_SE' in self.data.keys():
                    img = np.array(self.data[subj+'/T1_SE'])[:,:,slc_idx_sel]
                else:
                    print(f"T1 not found for sub {subj}..?????")
                    exit(0)
                imgs.append(img)
            elif contrast == 'T2_FLAIR':
                if subj+'/'+'T2_FLAIR' in self.data.keys():
                    img = np.array(self.data[subj+'/T2_FLAIR'])[:,:,slc_idx_sel]
                elif subj+'/'+'T2_FLAIR_2D' in self.data.keys():
                    img = np.array(self.data[subj+'/T2_FLAIR_2D'])[:,:,slc_idx_sel]
                else:
                    print(f"T2_FLAIR not found for sub {subj}..?????")
                    exit(0)
                imgs.append(img)
            elif contrast == 'PET_1p':
                if subj+'/'+'PET_1p' in self.data.keys():
                    img = np.array(self.data[subj+'/PET_1p'])[:,:,slc_idx_sel]
                else:
                    print(f"PET_1p not found for sub {subj}? Exiting..")
                    exit(0)
                imgs.append(img)
                
            elif contrast == 'PET_0.3p':
                if subj+'/'+'PET_0.3p' in self.data.keys():
                    img = np.array(self.data[subj+'/PET_0.3p'])[:,:,slc_idx_sel]
                else:
                    print(f"PET_0.3p not found for sub {subj}? Exiting..")
                    exit(0)
                imgs.append(img)

            elif contrast == 'PET_0.1p':
                if subj+'/'+'PET_0.1p' in self.data.keys():
                    img = np.array(self.data[subj+'/PET_0.1p'])[:,:,slc_idx_sel]
                else:
                    print(f"PET_0.1p not found for sub {subj}? Exiting..")
                    exit(0)
                imgs.append(img)
            elif contrast == 'PET_0.5p':
                if subj+'/'+'PET_0.5p' in self.data.keys():
                    img = np.array(self.data[subj+'/PET_0.5p'])[:,:,slc_idx_sel]
                else:
                    print(f"PET_10p not found for sub {subj}? Exiting..")
                    exit(0)
                imgs.append(img)
        # Interleaved images so that slice_1_t1, slice_1_t2, slice_2_t1, slice_2_t2
        if len(imgs) == 2:
            img = np.empty((self.image_size, self.image_size, 2 * len(slc_idx_sel)))
            img[:,:, ::2] = imgs[0]
            img[:, :, 1::2] = imgs[1]
        elif len(imgs) == 3:
            img = np.empty((self.image_size, self.image_size, 3 * len(slc_idx_sel)))
            img[:,:, ::3] = imgs[0]
            img[:, :, 1::3] = imgs[1]
            img[:, :, 2::3] = imgs[2]
            # print("interleaved multi-contrast..")
        # else:
        img = np.concatenate(imgs, axis=-1)
        img = np.transpose(img, (2, 0, 1))
        if self.normalization_method == "-1-1":
            inputs = torch.Tensor(img * 2 - 1)
        else:
            inputs = torch.Tensor(img)

        # target PET
        if subj+'/PET' in self.data.keys():
            targets = np.array(self.data[subj+'/PET'])[:,:,target_idx_sel]
        elif subj+'/PET_MAC' in self.data.keys():
            targets = np.array(self.data[subj+'/PET_MAC'])[:,:,target_idx_sel]
        elif subj+'/PET_QCLEAR' in self.data.keys():
            targets = np.array(self.data[subj+'/PET_QCLEAR'])[:,:,target_idx_sel]
        elif subj+'/PET_TOF' in self.data.keys():
            targets = np.array(self.data[subj+'/PET_TOF'])[:,:,target_idx_sel]
        else:
            print("Ops, PET not found??? Exiting..")
            exit(0)

        targets = np.transpose(targets, (2, 0, 1))
        if self.normalization_method == "-1-1":
            targets = torch.Tensor(targets * 2 - 1)
        else:
            targets = torch.Tensor(targets)

        output['mri'] = inputs
        output['pet'] = targets
        if 'label' in self.data[subj].attrs:
            if 'tumor' in self.data[subj].attrs['label']:
                output['label'] = 0
            elif 'amy' in self.data[subj].attrs['label']:
                output['label'] = 2
            else:
                output['label'] = 1
        else:
            # default to fdg label 1
            print(f"Label not found for subject {subj}...???")
            output['label'] = 1

        if self.prior is not None:
            prior_volume = self.prior_volumes[subj]
            prior = torch.Tensor(np.transpose(prior_volume[:, :, slc_idx_sel], (2, 0, 1))) # 3, 256, 256
        else:
            prior = torch.Tensor([0]) # just a dummy replacement
            
        output['lr'] = inputs
        output['hr'] = targets
        output['case_name'] = subj
        output['slice_idx'] = slice_idx
        output['prior'] = prior
        # print(subj, slice_idx, targets.shape, img.shape)
        return output
    

class PETMRClassEmb(Dataset):
    def __init__(self, data_h5, slice_file, img_size=(160, 192), slice_offset=1, contrast_list=['T1_GRE', 'T2_FLAIR'], normalization_method="-1-1", is_test=False, output_slice=1, target_lowdose=False, aug=False, distance=1):
        self.data = h5py.File(data_h5, 'r')
        self.contrast_list = contrast_list
        self.slices = []
        self.image_size = img_size
        self.output_slice = output_slice
        self.slice_offset = slice_offset
        self.aug = aug
        self.save_transform = 0
        self.distance = distance
        self.target_lowdose = target_lowdose
        with open(slice_file, 'r') as fr:
            for line in fr:
                res = line.strip('\n').split(" ")
                self.slices.append(res)
        self.normalization_method = normalization_method
        self.is_test = is_test
        with h5py.File(data_h5, "r") as f:
            self.subject_list = list(f.keys())
        self.fill = -1 if self.normalization_method == "-1-1" else 0
        # self.transform=transforms.Compose([RandomGenerator(output_size=[256, 256])])

        # self.transform = v2.Compose([
        #     v2.RandomHorizontalFlip(p=0.5),
        #     # v2.RandomResizedCrop(size=(256, 256), scale=(0.7,1)),
        #     v2.RandomRotation(degrees=(-10, 10), fill=self.fill)
        # ])
        self.tumors = ["case_0103", "case_0110", "case_0111", "case_0117", "case_0118", "case_0124", 
                        "case_0129", "case_0132", "case_0136", "case_0139", "case_0142", 
                        "case_0144", "case_0145", "case_0146", "case_0148", "case_0149", 
                        "case_0151", "case_0156", "case_0157", "case_0158", "case_0160",
                         "case_0161", "case_0164", "case_0168", "case_0169", "case_0174", 
                         "case_0175", "case_0188", "case_0196", "case_0208", "case_0219", 
                         "case_0226", "case_0238", "case_0240", "case_0241", "case_0242", 
                         "case_0248", "case_0251", "case_0252", "case_0257", "case_0258", 
                         "Fdg_Stanford_001", "Fdg_Stanford_002", "Fdg_Stanford_003", 
                         "Fdg_Stanford_004", "Fdg_Stanford_005", "Fdg_Stanford_006", 
                         "Fdg_Stanford_010", "Fdg_Stanford_011", "Fdg_Stanford_012", 
                         "Fdg_Stanford_014", "Fdg_Stanford_015", "Fdg_Stanford_016", 
                         "Fdg_Stanford_017", "Fdg_Stanford_023", "Fdg_Stanford_024", 
                         "Fdg_Stanford_025", "Fdg_Stanford_026", "Fdg_Stanford_027", 
                         "Fdg_Stanford_028", "Fdg_Stanford_029", "Fdg_Stanford_030", 
                         "Fdg_Stanford_031", "Fdg_Stanford_032", "852_06182015", "1496_05272016",
                        "1549_06232016", "1604_07142016", "1619_07192016", "2002_02142017", 
                        "2010_02212017", "2120_04122017", "2275_07062017", "2284_07112017", 
                        "2374_08242017", "case-0261", "case-0263", "case-0266", "case-0270", 
                        "case-0272", "case-0284", "case-0286", "case-0287", "case-0289", "case-0290", 
                        "case-0295", "case-0299"]
        
    def __len__(self):
        print(f"Total training slices: {len(self.slices)}")
        return len(self.slices)
    
    def augment(self, inputs, target, subj, slc): # input: (n, 160, 192); target: (1, 160, 192)
        # print(inputs.shape, target.shape)
        n, h, w = inputs.shape

        state = torch.get_rng_state()
        # make input n, 3, 160, 192
        repeated_inputs = np.repeat(np.expand_dims(inputs, 1), 3, axis=1) 

        # print(repeated_inputs.shape)
        assert repeated_inputs.shape == (n, 3, h, w)
        torch.set_rng_state(state)
        augmented_inputs = self.transform(torch.Tensor(repeated_inputs))
        # print("Augmented...")
        # print(augmented_inputs.shape)
        assert augmented_inputs.shape == (n, 3, h, w)

        # make target 1, 3, 160, 192
        repeated_target = np.repeat(target, 3, axis=0)
        torch.set_rng_state(state)
        augmented_target = self.transform(torch.Tensor(repeated_target))

        res_input = np.ones_like(inputs) * self.fill
        for i in range(n):
            # get the first channel
            res_input[i, :, :] = augmented_inputs[i, 0, :, :]

        res_target = augmented_target[0:1, :, :]
        
        assert res_target.shape == target.shape
        assert res_input.shape == inputs.shape
        return res_input, res_target

    def __getitem__(self, idx):
        subj = self.slices[idx][0]
        slice_idx = int(self.slices[idx][1]) # 3
        target_idx_sel = np.clip(np.arange(slice_idx - self.slice_offset, slice_idx + self.slice_offset + 1), 0, 88)        #     target_selection = slice_selection - 1
        slc_idx_sel = np.clip(np.arange(slice_idx - self.slice_offset - self.distance, slice_idx + self.slice_offset + 1 + self.distance), 0, 88)  # with neighbors..
        # print(slc_idx_sel, target_idx_sel)

        output  = {}
        imgs = []
        for contrast in self.contrast_list:
            if 'T1' in contrast:
                if subj+'/'+'T1_GRE' in self.data.keys():
                    img = np.array(self.data[subj+'/T1_GRE'])[:,:,slc_idx_sel]
                elif subj+'/T1_SE' in self.data.keys():
                    img = np.array(self.data[subj+'/T1_SE'])[:,:,slc_idx_sel]
                else:
                    print(f"T1 not found for sub {subj}..?????")
                    exit(0)
                imgs.append(img)
            elif contrast == 'T2_FLAIR':
                if subj+'/'+'T2_FLAIR' in self.data.keys():
                    img = np.array(self.data[subj+'/T2_FLAIR'])[:,:,slc_idx_sel]
                elif subj+'/'+'T2_FLAIR_2D' in self.data.keys():
                    img = np.array(self.data[subj+'/T2_FLAIR_2D'])[:,:,slc_idx_sel]
                else:
                    print(f"T2_FLAIR not found for sub {subj}..?????")
                    exit(0)
                imgs.append(img)
            elif contrast == 'PET_1p':
                if subj+'/'+'PET_1p' in self.data.keys():
                    img = np.array(self.data[subj+'/PET_1p'])[:,:,slc_idx_sel]
                else:
                    print(f"PET_1p not found for sub {subj}? Exiting..")
                    exit(0)
                imgs.append(img)
                
            elif contrast == 'PET_0.3p':
                if subj+'/'+'PET_0.3p' in self.data.keys():
                    img = np.array(self.data[subj+'/PET_0.3p'])[:,:,slc_idx_sel]
                else:
                    print(f"PET_0.3p not found for sub {subj}? Exiting..")
                    exit(0)
                imgs.append(img)

            elif contrast == 'PET_0.1p':
                if subj+'/'+'PET_0.1p' in self.data.keys():
                    img = np.array(self.data[subj+'/PET_0.1p'])[:,:,slc_idx_sel]
                else:
                    print(f"PET_0.1p not found for sub {subj}? Exiting..")
                    exit(0)
                imgs.append(img)
            elif contrast == 'PET_0.5p':
                if subj+'/'+'PET_0.5p' in self.data.keys():
                    img = np.array(self.data[subj+'/PET_0.5p'])[:,:,slc_idx_sel]
                else:
                    print(f"PET_10p not found for sub {subj}? Exiting..")
                    exit(0)
                imgs.append(img)
        # Interleaved images so that slice_1_t1, slice_1_t2, slice_2_t1, slice_2_t2
        if len(imgs) == 2:
            img = np.empty((self.image_size, self.image_size, 2 * len(slc_idx_sel)))
            img[:,:, ::2] = imgs[0]
            img[:, :, 1::2] = imgs[1]
        elif len(imgs) == 3:
            img = np.empty((self.image_size, self.image_size, 3 * len(slc_idx_sel)))
            img[:,:, ::3] = imgs[0]
            img[:, :, 1::3] = imgs[1]
            img[:, :, 2::3] = imgs[2]
            # print("interleaved multi-contrast..")
        # else:
        img = np.concatenate(imgs, axis=-1)
        img = np.transpose(img, (2, 0, 1))
        if self.normalization_method == "-1-1":
            inputs = torch.Tensor(img * 2 - 1)
        else:
            inputs = torch.Tensor(img)

        # target PET
        if subj+'/PET' in self.data.keys():
            targets = np.array(self.data[subj+'/PET'])[:,:,target_idx_sel]
        elif subj+'/PET_MAC' in self.data.keys():
            targets = np.array(self.data[subj+'/PET_MAC'])[:,:,target_idx_sel]
        elif subj+'/PET_QCLEAR' in self.data.keys():
            targets = np.array(self.data[subj+'/PET_QCLEAR'])[:,:,target_idx_sel]
        elif subj+'/PET_TOF' in self.data.keys():
            targets = np.array(self.data[subj+'/PET_TOF'])[:,:,target_idx_sel]
        else:
            print("Ops, PET not found??? Exiting..")
            exit(0)

        targets = np.transpose(targets, (2, 0, 1))
        if self.normalization_method == "-1-1":
            targets = torch.Tensor(targets * 2 - 1)
        else:
            targets = torch.Tensor(targets)

        output['mri'] = inputs
        output['pet'] = targets
        if 'label' in self.data[subj].attrs:
            if 'tumor' in self.data[subj].attrs['label']:
                output['label'] = 0
            elif 'amy' in self.data[subj].attrs['label']:
                output['label'] = 2
            else:
                output['label'] = 1
        else:
            # default to fdg label 1
            if 'FBB' in subj or 'PD' in subj:
                output['label'] = 2
            else:
                print(f"Label not found for subject {subj}...???")
                output['label'] = 1
        # if self.aug and self.transform:
        #     a, b = self.augment(inputs, targets, subj, slice_idx)
        #     output['mri'] = a
        #     output['pet'] = b
            
        return output



class PETMR(Dataset):
    def __init__(self, data_h5, slice_file, save_root=None, img_size=(160, 192), slice_offset=1, contrast_list=['T1_GRE', 'T2_FLAIR'], normalization_method="-1-1", is_test=False, output_slice=1, target_lowdose=False, aug=False):
        self.data = h5py.File(data_h5, 'r')
        self.contrast_list = contrast_list
        self.slices = []
        self.image_size = img_size
        self.output_slice = output_slice
        self.slice_offset = slice_offset
        self.aug = aug
        self.save_transform = 0
        self.save_root = save_root
        self.target_lowdose = target_lowdose
        with open(slice_file, 'r') as fr:
            for line in fr:
                res = line.strip('\n').split(" ")
                self.slices.append(res)
        self.normalization_method = normalization_method
        self.is_test = is_test
        with h5py.File(data_h5, "r") as f:
            self.subject_list = list(f.keys())
        self.fill = -1 if self.normalization_method == "-1-1" else 0
        # self.transform=transforms.Compose([RandomGenerator(output_size=[256, 256])])

        self.transform = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            # v2.RandomResizedCrop(size=(256, 256), scale=(0.7,1)),
            v2.RandomRotation(degrees=(-10, 10), fill=self.fill)
        ])
        
        
    def __len__(self):
        print(f"Total training slices: {len(self.slices)}")
        return len(self.slices)
    
    def augment(self, inputs, target, subj, slc): # input: (n, 160, 192); target: (1, 160, 192)
        # print(inputs.shape, target.shape)
        n, h, w = inputs.shape

        state = torch.get_rng_state()
        # make input n, 3, 160, 192
        repeated_inputs = np.repeat(np.expand_dims(inputs, 1), 3, axis=1) 

        # print(repeated_inputs.shape)
        assert repeated_inputs.shape == (n, 3, h, w)
        torch.set_rng_state(state)
        augmented_inputs = self.transform(torch.Tensor(repeated_inputs))
        # print("Augmented...")
        # print(augmented_inputs.shape)
        assert augmented_inputs.shape == (n, 3, h, w)

        # make target 1, 3, 160, 192
        repeated_target = np.repeat(target, 3, axis=0)
        torch.set_rng_state(state)
        augmented_target = self.transform(torch.Tensor(repeated_target))

        res_input = np.ones_like(inputs) * self.fill
        for i in range(n):
            # get the first channel
            res_input[i, :, :] = augmented_inputs[i, 0, :, :]

        res_target = augmented_target[0:1, :, :]
        
        assert res_target.shape == target.shape
        assert res_input.shape == inputs.shape
        return res_input, res_target

    def __getitem__(self, idx):
        subj = self.slices[idx][0]
        slice_idx = int(self.slices[idx][1])
        slc_idx_sel = np.clip(np.arange(slice_idx-self.slice_offset, slice_idx+self.slice_offset+1), 0, 88)
        output  = {}
        imgs = []
        for contrast in self.contrast_list:
            if 'T1' in contrast:
                if subj+'/'+'T1_GRE' in self.data.keys():
                    img = np.array(self.data[subj+'/T1_GRE'])[:,:,slc_idx_sel]
                elif subj+'/T1_SE' in self.data.keys():
                    img = np.array(self.data[subj+'/T1_SE'])[:,:,slc_idx_sel]
                else:
                    print(f"T1 not found for sub {subj}..?????")
                    exit(0)
                imgs.append(img)
            elif contrast == 'T2_FLAIR':
                if subj+'/'+'T2_FLAIR' in self.data.keys():
                    img = np.array(self.data[subj+'/T2_FLAIR'])[:,:,slc_idx_sel]
                elif subj+'/'+'T2_FLAIR_2D' in self.data.keys():
                    img = np.array(self.data[subj+'/T2_FLAIR_2D'])[:,:,slc_idx_sel]
                else:
                    print(f"T2_FLAIR not found for sub {subj}..?????")
                    exit(0)
                imgs.append(img)
            elif contrast == 'PET_1p':
                if subj+'/'+'PET_1p' in self.data.keys():
                    img = np.array(self.data[subj+'/PET_1p'])[:,:,slc_idx_sel]
                else:
                    print(f"PET_1p not found for sub {subj}? Exiting..")
                    exit(0)
                imgs.append(img)
                
            elif contrast == 'PET_0.3p':
                if subj+'/'+'PET_0.3p' in self.data.keys():
                    img = np.array(self.data[subj+'/PET_0.3p'])[:,:,slc_idx_sel]
                else:
                    print(f"PET_0.3p not found for sub {subj}? Exiting..")
                    exit(0)
                imgs.append(img)

            elif contrast == 'PET_0.1p':
                if subj+'/'+'PET_0.1p' in self.data.keys():
                    img = np.array(self.data[subj+'/PET_0.1p'])[:,:,slc_idx_sel]
                else:
                    print(f"PET_0.1p not found for sub {subj}? Exiting..")
                    exit(0)
                imgs.append(img)
            elif contrast == 'PET_0.5p':
                if subj+'/'+'PET_0.5p' in self.data.keys():
                    img = np.array(self.data[subj+'/PET_0.5p'])[:,:,slc_idx_sel]
                else:
                    print(f"PET_10p not found for sub {subj}? Exiting..")
                    exit(0)
                imgs.append(img)
        # Interleaved images so that slice_1_t1, slice_1_t2, slice_2_t1, slice_2_t2
        if len(imgs) == 2:
            img = np.empty((self.image_size, self.image_size, 2 * len(slc_idx_sel)))
            img[:,:, ::2] = imgs[0]
            img[:, :, 1::2] = imgs[1]
        elif len(imgs) == 3:
            img = np.empty((self.image_size, self.image_size, 3 * len(slc_idx_sel)))
            img[:,:, ::3] = imgs[0]
            img[:, :, 1::3] = imgs[1]
            img[:, :, 2::3] = imgs[2]
            # print("interleaved multi-contrast..")
        # else:
        img = np.concatenate(imgs, axis=-1)
        img = np.transpose(img, (2, 0, 1))
        if self.normalization_method == "-1-1":
            inputs = torch.Tensor(img * 2 - 1)
        else:
            inputs = torch.Tensor(img)

        # do not concat
        if self.target_lowdose:
            if subj+'/PET_10p' in self.data.keys():
                if self.output_slice > 1:
                    targets = np.array(self.data[subj+'/PET_10p'])[:,:,slc_idx_sel]
                else:
                    targets = self.data[subj+'/PET_10p'][:,:,slice_idx:slice_idx+1]
            else:
                print(f"PET 10p not found for subj {subj}.... Exiting")
                exit(0)
        else:
            if subj+'/PET' in self.data.keys():
                if self.output_slice > 1:
                    targets = np.array(self.data[subj+'/PET'])[:,:,slc_idx_sel]
                else:
                    targets = self.data[subj+'/PET'][:,:,slice_idx:slice_idx+1]
            elif subj+'/PET_MAC' in self.data.keys():
                if self.output_slice > 1:
                    targets = np.array(self.data[subj+'/PET_MAC'])[:,:,slc_idx_sel]
                else:
                    targets = self.data[subj+'/PET_MAC'][:,:,slice_idx:slice_idx+1]
            elif subj+'/PET_QCLEAR' in self.data.keys():
                if self.output_slice > 1:
                    targets = np.array(self.data[subj+'/PET_QCLEAR'])[:,:,slc_idx_sel]
                else:
                    targets = self.data[subj+'/PET_QCLEAR'][:,:,slice_idx:slice_idx+1]
            elif subj+'/PET_TOF' in self.data.keys():
                if self.output_slice > 1:
                    targets = np.array(self.data[subj+'/PET_TOF'])[:,:,slc_idx_sel]
                else:
                    targets = self.data[subj+'/PET_TOF'][:,:,slice_idx:slice_idx+1]
            else:
                print("Ops, PET not found??? Exiting..")
                exit(0)

        targets = np.transpose(targets, (2, 0, 1))
        if self.normalization_method == "-1-1":
            targets = torch.Tensor(targets * 2 - 1)
        else:
            targets = torch.Tensor(targets)

        output['mri'] = inputs
        output['pet'] = targets
        # if self.aug and self.transform:
        #     a, b = self.augment(inputs, targets, subj, slice_idx)
        #     output['mri'] = a
        #     output['pet'] = b
            
        return output


def from_pil_image(x):
    """Converts from a PIL image to a tensor."""
    x = TF.to_tensor(x)
    if x.ndim == 2:
        x = x[..., None]
    return x * 2 - 1


def to_pil_image(x):
    """Converts from a tensor to a PIL image."""
    if x.ndim == 4:
        assert x.shape[0] == 1
        x = x[0]
    if x.shape[0] == 1:
        x = x[0]
    if x.shape[0] >= 3:
        x = x[x.shape[0]//2]
    print("Converting to image")
    print(x.shape)
    return TF.to_pil_image((x.clamp(-1, 1) + 1) / 2)


def hf_datasets_augs_helper(examples, transform, image_key, mode='RGB'):
    """Apply passed in transforms for HuggingFace Datasets."""
    images = [transform(image.convert(mode)) for image in examples[image_key]]
    return {image_key: images}


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f'input has {x.ndim} dims but target_dims is {target_dims}, which is less')
    return x[(...,) + (None,) * dims_to_append]


def n_params(module):
    """Returns the number of trainable parameters in a module."""
    return sum(p.numel() for p in module.parameters())


def download_file(path, url, digest=None):
    """Downloads a file if it does not exist, optionally checking its SHA-256 hash."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with urllib.request.urlopen(url) as response, open(path, 'wb') as f:
            shutil.copyfileobj(response, f)
    if digest is not None:
        file_digest = hashlib.sha256(open(path, 'rb').read()).hexdigest()
        if digest != file_digest:
            raise OSError(f'hash of {path} (url: {url}) failed to validate')
    return path


@contextmanager
def train_mode(model, mode=True):
    """A context manager that places a model into training mode and restores
    the previous mode on exit."""
    modes = [module.training for module in model.modules()]
    try:
        yield model.train(mode)
    finally:
        for i, module in enumerate(model.modules()):
            module.training = modes[i]


def eval_mode(model):
    """A context manager that places a model into evaluation mode and restores
    the previous mode on exit."""
    return train_mode(model, False)


@torch.no_grad()
def ema_update(model, averaged_model, decay):
    """Incorporates updated model parameters into an exponential moving averaged
    version of a model. It should be called after each optimizer step."""
    model_params = dict(model.named_parameters())
    averaged_params = dict(averaged_model.named_parameters())
    assert model_params.keys() == averaged_params.keys()

    for name, param in model_params.items():
        averaged_params[name].lerp_(param, 1 - decay)

    model_buffers = dict(model.named_buffers())
    averaged_buffers = dict(averaged_model.named_buffers())
    assert model_buffers.keys() == averaged_buffers.keys()

    for name, buf in model_buffers.items():
        averaged_buffers[name].copy_(buf)


class EMAWarmup:
    """Implements an EMA warmup using an inverse decay schedule.
    If inv_gamma=1 and power=1, implements a simple average. inv_gamma=1, power=2/3 are
    good values for models you plan to train for a million or more steps (reaches decay
    factor 0.999 at 31.6K steps, 0.9999 at 1M steps), inv_gamma=1, power=3/4 for models
    you plan to train for less (reaches decay factor 0.999 at 10K steps, 0.9999 at
    215.4k steps).
    Args:
        inv_gamma (float): Inverse multiplicative factor of EMA warmup. Default: 1.
        power (float): Exponential factor of EMA warmup. Default: 1.
        min_value (float): The minimum EMA decay rate. Default: 0.
        max_value (float): The maximum EMA decay rate. Default: 1.
        start_at (int): The epoch to start averaging at. Default: 0.
        last_epoch (int): The index of last epoch. Default: 0.
    """

    def __init__(self, inv_gamma=1., power=1., min_value=0., max_value=1., start_at=0,
                 last_epoch=0):
        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value
        self.max_value = max_value
        self.start_at = start_at
        self.last_epoch = last_epoch

    def state_dict(self):
        """Returns the state of the class as a :class:`dict`."""
        return dict(self.__dict__.items())

    def load_state_dict(self, state_dict):
        """Loads the class's state.
        Args:
            state_dict (dict): scaler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_value(self):
        """Gets the current EMA decay rate."""
        epoch = max(0, self.last_epoch - self.start_at)
        value = 1 - (1 + epoch / self.inv_gamma) ** -self.power
        return 0. if epoch < 0 else min(self.max_value, max(self.min_value, value))

    def step(self):
        """Updates the step count."""
        self.last_epoch += 1


class InverseLR(optim.lr_scheduler._LRScheduler):
    """Implements an inverse decay learning rate schedule with an optional exponential
    warmup. When last_epoch=-1, sets initial lr as lr.
    inv_gamma is the number of steps/epochs required for the learning rate to decay to
    (1 / 2)**power of its original value.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        inv_gamma (float): Inverse multiplicative factor of learning rate decay. Default: 1.
        power (float): Exponential factor of learning rate decay. Default: 1.
        warmup (float): Exponential warmup factor (0 <= warmup < 1, 0 to disable)
            Default: 0.
        min_lr (float): The minimum learning rate. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
    """

    def __init__(self, optimizer, inv_gamma=1., power=1., warmup=0., min_lr=0.,
                 last_epoch=-1, verbose=False):
        self.inv_gamma = inv_gamma
        self.power = power
        if not 0. <= warmup < 1:
            raise ValueError('Invalid value for warmup')
        self.warmup = warmup
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.")

        return self._get_closed_form_lr()

    def _get_closed_form_lr(self):
        warmup = 1 - self.warmup ** (self.last_epoch + 1)
        lr_mult = (1 + self.last_epoch / self.inv_gamma) ** -self.power
        return [warmup * max(self.min_lr, base_lr * lr_mult)
                for base_lr in self.base_lrs]


class ExponentialLR(optim.lr_scheduler._LRScheduler):
    """Implements an exponential learning rate schedule with an optional exponential
    warmup. When last_epoch=-1, sets initial lr as lr. Decays the learning rate
    continuously by decay (default 0.5) every num_steps steps.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        num_steps (float): The number of steps to decay the learning rate by decay in.
        decay (float): The factor by which to decay the learning rate every num_steps
            steps. Default: 0.5.
        warmup (float): Exponential warmup factor (0 <= warmup < 1, 0 to disable)
            Default: 0.
        min_lr (float): The minimum learning rate. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
    """

    def __init__(self, optimizer, num_steps, decay=0.5, warmup=0., min_lr=0.,
                 last_epoch=-1, verbose=False):
        self.num_steps = num_steps
        self.decay = decay
        if not 0. <= warmup < 1:
            raise ValueError('Invalid value for warmup')
        self.warmup = warmup
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.")

        return self._get_closed_form_lr()

    def _get_closed_form_lr(self):
        warmup = 1 - self.warmup ** (self.last_epoch + 1)
        lr_mult = (self.decay ** (1 / self.num_steps)) ** self.last_epoch
        return [warmup * max(self.min_lr, base_lr * lr_mult)
                for base_lr in self.base_lrs]


class ConstantLRWithWarmup(optim.lr_scheduler._LRScheduler):
    """Implements a constant learning rate schedule with an optional exponential
    warmup. When last_epoch=-1, sets initial lr as lr.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        warmup (float): Exponential warmup factor (0 <= warmup < 1, 0 to disable)
            Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
    """

    def __init__(self, optimizer, warmup=0., last_epoch=-1, verbose=False):
        if not 0. <= warmup < 1:
            raise ValueError('Invalid value for warmup')
        self.warmup = warmup
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.")

        return self._get_closed_form_lr()

    def _get_closed_form_lr(self):
        warmup = 1 - self.warmup ** (self.last_epoch + 1)
        return [warmup * base_lr for base_lr in self.base_lrs]


def stratified_uniform(shape, group=0, groups=1, dtype=None, device=None):
    """Draws stratified samples from a uniform distribution."""
    if groups <= 0:
        raise ValueError(f"groups must be positive, got {groups}")
    if group < 0 or group >= groups:
        raise ValueError(f"group must be in [0, {groups})")
    n = shape[-1] * groups
    offsets = torch.arange(group, n, groups, dtype=dtype, device=device)
    u = torch.rand(shape, dtype=dtype, device=device)
    return (offsets + u) / n


stratified_settings = threading.local()


@contextmanager
def enable_stratified(group=0, groups=1, disable=False):
    """A context manager that enables stratified sampling."""
    try:
        stratified_settings.disable = disable
        stratified_settings.group = group
        stratified_settings.groups = groups
        yield
    finally:
        del stratified_settings.disable
        del stratified_settings.group
        del stratified_settings.groups


@contextmanager
def enable_stratified_accelerate(accelerator, disable=False):
    """A context manager that enables stratified sampling, distributing the strata across
    all processes and gradient accumulation steps using settings from Hugging Face Accelerate."""
    try:
        rank = accelerator.process_index
        world_size = accelerator.num_processes
        acc_steps = accelerator.gradient_state.num_steps
        acc_step = accelerator.step % acc_steps
        group = rank * acc_steps + acc_step
        groups = world_size * acc_steps
        with enable_stratified(group, groups, disable=disable):
            yield
    finally:
        pass


def stratified_with_settings(shape, dtype=None, device=None):
    """Draws stratified samples from a uniform distribution, using settings from a context
    manager."""
    if not hasattr(stratified_settings, 'disable') or stratified_settings.disable:
        return torch.rand(shape, dtype=dtype, device=device)
    return stratified_uniform(
        shape, stratified_settings.group, stratified_settings.groups, dtype=dtype, device=device
    )


def rand_log_normal(shape, loc=0., scale=1., device='cpu', dtype=torch.float32):
    """Draws samples from an lognormal distribution."""
    u = stratified_with_settings(shape, device=device, dtype=dtype) * (1 - 2e-7) + 1e-7
    return torch.distributions.Normal(loc, scale).icdf(u).exp()


def rand_log_logistic(shape, loc=0., scale=1., min_value=0., max_value=float('inf'), device='cpu', dtype=torch.float32):
    """Draws samples from an optionally truncated log-logistic distribution."""
    min_value = torch.as_tensor(min_value, device=device, dtype=torch.float64)
    max_value = torch.as_tensor(max_value, device=device, dtype=torch.float64)
    min_cdf = min_value.log().sub(loc).div(scale).sigmoid()
    max_cdf = max_value.log().sub(loc).div(scale).sigmoid()
    u = stratified_with_settings(shape, device=device, dtype=torch.float64) * (max_cdf - min_cdf) + min_cdf
    return u.logit().mul(scale).add(loc).exp().to(dtype)


def rand_log_uniform(shape, min_value, max_value, device='cpu', dtype=torch.float32):
    """Draws samples from an log-uniform distribution."""
    min_value = math.log(min_value)
    max_value = math.log(max_value)
    return (stratified_with_settings(shape, device=device, dtype=dtype) * (max_value - min_value) + min_value).exp()


def rand_v_diffusion(shape, sigma_data=1., min_value=0., max_value=float('inf'), device='cpu', dtype=torch.float32):
    """Draws samples from a truncated v-diffusion training timestep distribution."""
    min_cdf = math.atan(min_value / sigma_data) * 2 / math.pi
    max_cdf = math.atan(max_value / sigma_data) * 2 / math.pi
    u = stratified_with_settings(shape, device=device, dtype=dtype) * (max_cdf - min_cdf) + min_cdf
    return torch.tan(u * math.pi / 2) * sigma_data

# draw a random t and get the sigma at that t
def rand_cosine_interpolated(shape, image_d, noise_d_low, noise_d_high, sigma_data=1., min_value=1e-3, max_value=1e3, device='cpu', dtype=torch.float32):
    """Draws samples from an interpolated cosine timestep distribution (from simple diffusion)."""

    def logsnr_schedule_cosine(t, logsnr_min, logsnr_max):
        t_min = math.atan(math.exp(-0.5 * logsnr_max))
        t_max = math.atan(math.exp(-0.5 * logsnr_min))
        return -2 * torch.log(torch.tan(t_min + t * (t_max - t_min)))

    def logsnr_schedule_cosine_shifted(t, image_d, noise_d, logsnr_min, logsnr_max):
        shift = 2 * math.log(noise_d / image_d)
        return logsnr_schedule_cosine(t, logsnr_min - shift, logsnr_max - shift) + shift

    def logsnr_schedule_cosine_interpolated(t, image_d, noise_d_low, noise_d_high, logsnr_min, logsnr_max):
        logsnr_low = logsnr_schedule_cosine_shifted(t, image_d, noise_d_low, logsnr_min, logsnr_max)
        logsnr_high = logsnr_schedule_cosine_shifted(t, image_d, noise_d_high, logsnr_min, logsnr_max)
        return torch.lerp(logsnr_low, logsnr_high, t)

    logsnr_min = -2 * math.log(min_value / sigma_data)
    logsnr_max = -2 * math.log(max_value / sigma_data)
    u = stratified_with_settings(shape, device=device, dtype=dtype)
    logsnr = logsnr_schedule_cosine_interpolated(u, image_d, noise_d_low, noise_d_high, logsnr_min, logsnr_max)
    return torch.exp(-logsnr / 2) * sigma_data


def rand_split_log_normal(shape, loc, scale_1, scale_2, device='cpu', dtype=torch.float32):
    """Draws samples from a split lognormal distribution."""
    n = torch.randn(shape, device=device, dtype=dtype).abs()
    u = torch.rand(shape, device=device, dtype=dtype)
    n_left = n * -scale_1 + loc
    n_right = n * scale_2 + loc
    ratio = scale_1 / (scale_1 + scale_2)
    return torch.where(u < ratio, n_left, n_right).exp()


class FolderOfImages(data.Dataset):
    """Recursively finds all images in a directory. It does not support
    classes/targets."""

    IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp'}

    def __init__(self, root, transform=None):
        super().__init__()
        self.root = Path(root)
        self.transform = nn.Identity() if transform is None else transform
        self.paths = sorted(path for path in self.root.rglob('*') if path.suffix.lower() in self.IMG_EXTENSIONS)

    def __repr__(self):
        return f'FolderOfImages(root="{self.root}", len: {len(self)})'

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, key):
        path = self.paths[key]
        with open(path, 'rb') as f:
            image = Image.open(f).convert('RGB')
        image = self.transform(image)
        return image,


class CSVLogger:
    def __init__(self, filename, columns):
        self.filename = Path(filename)
        self.columns = columns
        if self.filename.exists():
            self.file = open(self.filename, 'a')
        else:
            self.file = open(self.filename, 'w')
            self.write(*self.columns)

    def write(self, *args):
        print(*args, sep=',', file=self.file, flush=True)


@contextmanager
def tf32_mode(cudnn=None, matmul=None):
    """A context manager that sets whether TF32 is allowed on cuDNN or matmul."""
    cudnn_old = torch.backends.cudnn.allow_tf32
    matmul_old = torch.backends.cuda.matmul.allow_tf32
    try:
        if cudnn is not None:
            torch.backends.cudnn.allow_tf32 = cudnn
        if matmul is not None:
            torch.backends.cuda.matmul.allow_tf32 = matmul
        yield
    finally:
        if cudnn is not None:
            torch.backends.cudnn.allow_tf32 = cudnn_old
        if matmul is not None:
            torch.backends.cuda.matmul.allow_tf32 = matmul_old


def get_safetensors_metadata(path):
    """Retrieves the metadata from a safetensors file."""
    return safetensors.safe_open(path, "pt").metadata()


def ema_update_dict(values, updates, decay):
    for k, v in updates.items():
        if k not in values:
            values[k] = v
        else:
            values[k] *= decay
            values[k] += (1 - decay) * v
    return values
