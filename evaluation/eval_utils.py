import nibabel as nib
import numpy as np
import os
import glob
import h5py
import matplotlib.pyplot as plt
import json
import pandas as pd
import seaborn as sns
import time
from scipy.stats import chisquare, wasserstein_distance
import torch
import torch.nn.functional as F
from math import exp
import numpy as np
import scipy.ndimage as ndimage
import skimage
import piq

sns.set_context("paper")
sns.set_style("darkgrid")

color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', 'lightblue', 'limegreen', 'black', 'maroon']
color_roi_mapping = {
    "cerebellum": '#1f77b4',
    "frontal_cortex": '#ff7f0e',
    "temporal_cortex": '#2ca02c',
    "parietal_cortex": '#d62728',
    "occipital_cortex": 'maroon',
    "insular_cortex": '#9467bd',
    "basal_ganglia": '#8c564b', 
    "deep_gray_matter": '#8c564b', 
    "brain_stem": '#e377c2',
    "left_cerebral_white_matter": '#7f7f7f',
    "right_cerebral_white_matter": 'black',
    "hippo+amyg": '#bcbd22',
    "CSF": '#17becf',
    "corpus_callosum": 'lightblue',
    "cortex": '#ff7f0e',
    "cerebral_white_matter": '#2ca02c',
    "thalamus": '#d62728',
    "hippocampus": 'maroon',
    "caudate": '#9467bd',
    "putamen": '#8c564b', 
    "globus_pallidus": '#7f7f7f',
    "amygdala": 'black',
}

ROIS_Aparc2009 = {
    'frontal_cortex': [11101, 11105, 11112, 11113, 11114, 11115, 11116, 11153, 11154, 11155, 
                      12101, 12105, 12112, 12113, 12114, 12115, 12116, 12153, 12154, 12155],
    'temporal_cortex': [11121, 11122, 11123, 11133, 11134, 11135, 11136, 11137, 11138, 11144, 11161, 11162, 11173, 11174, 11175,
                       12121, 12122, 12123, 12133, 12134, 12135, 12136, 12137, 12138, 12144, 12161, 12162, 12173, 12174, 12175],
    'parietal_cortex': [11125, 11126, 11127, 11157, 11172,
                       12125, 12126, 12127, 12157, 12172],
    'occipital_cortex': [11102, 11119, 11120, 11143, 11160,
                        12102, 12119, 12120, 12143, 12160],
    'insular_cortex': [11117, 11118, 11148, 11149, 11150,
               12117, 12118, 12148, 12149, 12150],
    'left_cerebral_white_matter': [2],
    'right_cerebral_white_matter': [41],
    'deep_gray_matter': [10, 11, 12, 13, 49, 50, 51, 52],
    'hippo+amyg': [17, 53, 18, 54],
    'brain_stem': [16],
    'CSF': [24],
    "corpus_callosum": [251, 252, 253, 254, 255],
    'cerebellum': [7, 8, 46, 47]
}

ROIS_Aparc2009_cortex_only = {
    'frontal_cortex': [11101, 11105, 11112, 11113, 11114, 11115, 11116, 11153, 11154, 11155, 
                      12101, 12105, 12112, 12113, 12114, 12115, 12116, 12153, 12154, 12155],
    'temporal_cortex': [11121, 11122, 11123, 11133, 11134, 11135, 11136, 11137, 11138, 11144, 11161, 11162, 11173, 11174, 11175,
                       12121, 12122, 12123, 12133, 12134, 12135, 12136, 12137, 12138, 12144, 12161, 12162, 12173, 12174, 12175],
    'parietal_cortex': [11125, 11126, 11127, 11157, 11172,
                       12125, 12126, 12127, 12157, 12172],
    'occipital_cortex': [11102, 11119, 11120, 11143, 11160,
                        12102, 12119, 12120, 12143, 12160],
    'insular_cortex': [11117, 11118, 11148, 11149, 11150,
               12117, 12118, 12148, 12149, 12150],
    'cerebellum': [7, 8, 46, 47]
}

LR_ROIS_Aparc2009 = {
    'frontal_cortex': {
        'left': [11101, 11105, 11112, 11113, 11114, 11115, 11116, 11153, 11154, 11155],
        'right': [12101, 12105, 12112, 12113, 12114, 12115, 12116, 12153, 12154, 12155]
    },
    'temporal_cortex': {
        'left': [11121, 11122, 11123, 11133, 11134, 11135, 11136, 11137, 11138, 11144, 11161, 11162, 11173, 11174, 11175],
        'right': [12121, 12122, 12123, 12133, 12134, 12135, 12136, 12137, 12138, 12144, 12161, 12162, 12173, 12174, 12175]
    },
    'parietal_cortex': {
        'left': [11125, 11126, 11127, 11157, 11172],
        'right': [12125, 12126, 12127, 12157, 12172]
    },
    'occipital_cortex': {
        'left': [11102, 11119, 11120, 11143, 11160],
        'right': [12102, 12119, 12120, 12143, 12160]
    },
    'insular_cortex': {
        'left': [11117, 11118, 11148, 11149, 11150,],
        'right': [12117, 12118, 12148, 12149, 12150]
    },
    "cerebral_white_matter": {
        'left': [2], 
        'right':[41]
    },
    'deep_gray_matter': { # caudate, putamen, gp, thalamus
        'left': [10, 11, 12, 13],
        'right': [49, 50, 51, 52]
    },
    'hippo+amyg': {
        'left': [17, 18],
        'right': [53, 54]
    },
}
    

ROIS_Aseg = {
        "cerebellum": [7, 8, 46, 47],
        "cortex": [3, 42],
        "cerebral_white_matter": [2, 41],
        "thalamus": [10, 49],
        "hippocampus": [17, 53],
        "caudate": [11, 50],
        "putamen": [12, 51], 
        "brain_stem": [16],
        "globus_pallidus": [13, 52],
        "amygdala": [18, 54],
        "CSF": [24],
        "corpus_callosum": [251, 252, 253, 254, 255]
    }

## ROI with lateralization
LR_ROIS_Aseg = {
    "cerebellum": {'left': [7, 8], 
                   'right': [46, 47]},
    "cortex": {'left': [3],
              'right': [42]},
    "cerebral_white_matter": {'left': [2], 
                              'right':[41]},
    "hippocampus": {'left': [17],
                   'right': [53]},
    "caudate": {'left': [11],
               'right': [50]},
    "thalamus": {'left': [10],
                'right': [49]},
    "putamen": {'left': [12],
                'right': [51]}, 
    "globus_pallidus": {'left': [13],
                        'right': [52]},
    "amygdala": {'left': [18], 
                 'right': [54]},
    # "wm_hypointensities": {'left': [78],
    #                        'right': [79]}
}
## Short name for visualization purpose
Short_names = {
    'frontal_cortex': 'FC',
    'temporal_cortex': 'TC',
    'parietal_cortex': 'PC',
    'occipital_cortex': 'OC',
    'insular_cortex': 'IC',
    'left_cerebral_white_matter': 'LCWM',
    'right_cerebral_white_matter': 'RCWM',
    'basal_ganglia': 'BG',
    'deep_gray_matter': 'DGM',
    'hippo+amyg': 'HipAmy',
    'hippo+thalamus+amyg': 'HipThAmy',
    "cerebral_white_matter": 'CWM',
    "cerebellum": "cb",
    "cortex": 'cortex',
    "hippocampus": "hippo",
    "caudate": "caudate",
    "thalamus": "tha",
    "putamen": "putamen",
    "globus_pallidus": "gp",
    "amygdala": "amygdala",
    "brain_stem": 'bs',
    "corpus_callosum": 'cc',
    'wm_hypointensities': 'wm_hypo'
}

TARGET_ROIS = ROIS_Aparc2009
TARGET_LR_ROIS = LR_ROIS_Aparc2009
TARGET_SEG_FILE = "aparc.a2009s+aseg.nii"
TARGET_CORTEX_ROIS = ROIS_Aparc2009_cortex_only

def json_to_pd_row(subj, res):
    # print(res)
    dfs = []
    for k, v in res.items():
        json_v = pd.json_normalize(v, sep='_')
        json_v['pet_type'] = k
        dfs.append(json_v)
    df = pd.concat(dfs, ignore_index=True)
    df['subject'] = subj
    col = df.pop("pet_type")
    df.insert(0, col.name, col)
    col = df.pop("subject")
    df.insert(0, col.name, col)
    return df

def plot_suvr_diff_one_subj(suvr_diff_dict, subj, ax, ignore_roi=['cerebellum'], save_name=None):
    for ignore in ignore_roi:
        if ignore in list(suvr_diff_dict.keys()):
            del suvr_diff_dict[ignore]
    preds = list(suvr_diff_dict.keys())
    # print(preds)
    all_rois = list(TARGET_ROIS.keys())
    all_names = [Short_names[x] if x in Short_names else x for x in all_rois]
    all_pred = [[] for _ in range(len(preds))]
    
    for i, pred in enumerate(preds):
        for roi in all_rois:
            pred_diff = suvr_diff_dict[pred][roi]
            all_pred[i].append(pred_diff)
            
    bar_width = 0.8 / (len(preds)+1)
    r1 = np.arange(len(all_rois))
    
    # Creating the bar plot
    colors = color_list[:len(preds)]
    labels = preds
    
    for i in range(len(all_pred)):
        r = [x + bar_width * i for x in r1]
        ax.bar(r, all_pred[i], color=colors[i], width=bar_width, edgecolor='grey', label=labels[i])
    
    # Adding the labels
    ax.set_xlabel('ROI Name', fontweight='bold')
    ax.set_xticks([r + bar_width / 2 for r in range(len(all_rois))], all_names, rotation=30, fontsize=10)

    # Adding the legend
    ax.legend()
    ax.set_title(f"SUVR Difference for Subject {subj}")

def plot_asymmetry(info, subj, ax, save_name=None):
    preds = list(info.keys())
    # print(preds)
    all_rois = list(TARGET_LR_ROIS.keys())
    all_names = [Short_names[x] for x in all_rois]
    all_pred = [[] for _ in range(len(preds))]

    for i, pred in enumerate(preds):
        for roi in all_rois:
            pred_diff = info[pred][roi]['diff']
            all_pred[i].append(pred_diff)
    
    # print(all_rois, all_pred)

    bar_width = 0.8 / (len(preds)+1)
    r1 = np.arange(len(all_rois))
    
    # Creating the bar plot
    colors = color_list[:len(preds)]
    labels = preds
    
    for i in range(len(all_pred)):
        r = [x + bar_width * i for x in r1]
        ax.bar(r, all_pred[i], color=colors[i], width=bar_width, edgecolor='grey', label=labels[i])
    
    # Adding the labels
    ax.set_xlabel('ROI Name', fontweight='bold')
    ax.set_xticks([r + bar_width / 2 for r in range(len(all_rois))], all_names, rotation=30, fontsize=10)
    # Adding the legend
    # ax.legend()
    ax.set_title(f"SUVR Asymmetry for Subject {subj}")
    # if save_name:
    #     plt.tight_layout()
    #     plt.savefig(save_name, bbox_inches='tight', dpi=400)
    # else:
    #     plt.show()
    # return fig

'''
    Plots
'''
def plot_slice_visulization_all_subj(subjects, all_subj_info, exp_name, views=['axial', 'coronal', 'sagittal'], save=None):
    nrow = len(subjects)
    ncol = 2 * len(views) # true pet & pred pet
    print(nrow, ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5*ncol, 5*nrow))
    flatten_axes = axes.flatten()
    axes_idx = 0
    used_axes_idx = []
    for i, subj in enumerate(subjects):
        if 'label' in all_subj_info[subj]:
            label = all_subj_info[subj]['label']
            print(label)
        else:
            label = ""
        for j, view in enumerate(views):
            if view == 'axial':
                idx = 89 // 2
                pred_slice = np.rot90(all_subj_info[subj]['pets'][exp_name][:, :, idx])
                target_slice = np.rot90(all_subj_info[subj]['pets']['truth'][:, :, idx])
                asp='equal'
            elif view == 'coronal':
                idx = 256 // 2
                pred_slice = np.rot90(all_subj_info[subj]['pets'][exp_name][:, idx, :])
                target_slice = np.rot90(all_subj_info[subj]['pets']['truth'][:, idx, :])
                asp='auto'
            else:
                idx = 256 // 2
                pred_slice = np.flip(np.rot90(all_subj_info[subj]['pets'][exp_name][idx, :, :]), axis=1)
                target_slice = np.flip(np.rot90(all_subj_info[subj]['pets']['truth'][idx, :, :]), axis=1)
                asp = 'auto'

            ax_idx = i*ncol+j
            flatten_axes[ax_idx].imshow(target_slice, cmap='gray', aspect=asp)
            flatten_axes[ax_idx].set_title(f"Subj {subj} {label} {view} Acquired")
            flatten_axes[ax_idx].axis("off")
            used_axes_idx.append(ax_idx)
    
            flatten_axes[ax_idx + len(views)].imshow(pred_slice, cmap='gray', aspect=asp)
            flatten_axes[ax_idx + len(views)].set_title(f"Subj {subj} {label} {view} Synth")
            flatten_axes[ax_idx + len(views)].axis("off")
            used_axes_idx.append(ax_idx + len(views))

    for j in range(len(axes)):
        if j not in used_axes_idx:
            fig.delaxes(axes[j])
    fig.suptitle(f"{exp_name} Slice Visualizations for All Subjects", y=1.03)
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight')
    else:
        plt.show()

def plot_slice_visualization_one_subject(subj, all_subj_info, exp_name, views=['axial', 'coronal', 'sagittal'], save=None, slice_idx_ratio=[0.3, 0.5, 0.7]):
    nrow = len(slice_idx_ratio)
    ncol = 2 * len(views) # true pet & pred pet
    print(nrow, ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5*ncol, 5*nrow))
    flatten_axes = axes.flatten()
    axes_idx = 0
    used_axes_idx = []
    if 'label' in all_subj_info[subj]:
        label = all_subj_info[subj]['label']
    else:
        label = ''
    for i, idx_ratio in enumerate(slice_idx_ratio):
        for j, view in enumerate(views):
            if view == 'axial':
                idx = int(89 * idx_ratio)
                pred_slice = np.rot90(all_subj_info[subj]['pets'][exp_name][:, :, idx])
                target_slice = np.rot90(all_subj_info[subj]['pets']['truth'][:, :, idx])
                asp='equal'
            elif view == 'coronal':
                idx = int(256 * idx_ratio)
                pred_slice = np.rot90(all_subj_info[subj]['pets'][exp_name][:, idx, :])
                target_slice = np.rot90(all_subj_info[subj]['pets']['truth'][:, idx, :])
                asp='auto'
            else:
                idx = int(256 * idx_ratio)
                pred_slice = np.flip(np.rot90(all_subj_info[subj]['pets'][exp_name][idx, :, :]), axis=1)
                target_slice = np.flip(np.rot90(all_subj_info[subj]['pets']['truth'][idx, :, :]), axis=1)
                asp = 'auto'

            ax_idx = i*ncol+2*j
            # print(ax_idx)
            flatten_axes[ax_idx].imshow(target_slice, cmap='gray', aspect=asp)
            flatten_axes[ax_idx].set_title(f"{label} {view} Acquired")
            flatten_axes[ax_idx].axis("off")
    
            flatten_axes[ax_idx + 1].imshow(pred_slice, cmap='gray', aspect=asp)
            flatten_axes[ax_idx + 1].set_title(f"{label} {view} Synth")
            flatten_axes[ax_idx + 1].axis("off")
    plt.subplots_adjust(wspace=0.2)
    fig.suptitle(f"{exp_name} Slice Visualizations for Subject {subj} label {label}", y=1.03, fontsize=18)
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight')
    else:
        plt.show()


# TODO: better replace with your own data structure
def gather_files(subj, test_dir, test_type, TARGET_ROIS, TARGET_LR_ROIS, target_seg_file='aseg.nii', pets_only=False, slice_start=22, slice_end=73):
    if not pets_only:
        print("Computing masks....")
        seg_file = os.path.join("xx", subj, target_seg_file)
        roi_masks_global, area_info_global = compute_masks(seg_file, TARGET_ROIS)
        
        roi_masks_left_right = compute_hemispheric_masks(seg_file, TARGET_LR_ROIS)
    
    if 'sub-control' in subj:
        pet_truth = [f"preprocessed_external/preprocessed_external_healthy/{subj}/pet/reslice_PET_full.nii"]
    elif 'sub-patient' in subj:
        pet_truth = [f"preprocessed_external/preprocessed_external_epilepsy/{subj}/pet/reslice_PET_full.nii"]
    elif len(subj) == 5:
        pet_truth = sorted(glob.glob(os.path.join("preprocessed_cases", subj, 'PET_full*.nii')))


    if len(pet_truth) == 0:
        raise FileNotFoundError(f"Could not find ground truth PET for {subj} in external dataset")
    
    pet_truth = pet_truth[0]
    print(pet_truth)
    pets = {'truth': nib.load(pet_truth).get_fdata()}
    # normalize true pet to 0-1
    temp = np.zeros((256, 256, 89))
    temp[:, :, slice_start:slice_end] = pets['truth'][:, :, slice_start:slice_end]
    pets['truth'] = norm(temp)
    
    # idx = 1
    all_data = {}
    pet_data = list(glob.glob(os.path.join(test_dir, subj) + "/*.nii"))
    if pet_data == []:
        print(f"Could not find predicted pet file for {subj} from {os.path.join(test_dir, subj) + "/*.nii"}")
        exit(0)
    curr_pred_pet = nib.load(pet_data[0]).get_fdata()
    assert curr_pred_pet.shape == (256, 256, 89)
    if curr_pred_pet.min() < 0:
        temp = np.ones((256, 256, 89)) * (-1)
    else:
        temp = np.zeros((256, 256, 89))
    temp[:, :, slice_start:slice_end] = curr_pred_pet[:, :, slice_start:slice_end]
    curr_pred_pet = temp
    # normalize synth pet to 0-1
    pets[test_type] = norm(curr_pred_pet) 
    if pets_only:
        return pets
    return pets, roi_masks_global, roi_masks_left_right, area_info_global



def norm(img, norm_type='max'):
    if img.min() < 0:
        img = (img + 1) / 2
    if norm_type == 'max':
        maxi = img.max()
        
        mini = img.min()
        # img = np.clip(img, a_min=mini, a_max=maxi)
        img = (img - mini) / (maxi - mini)
        img = np.clip(img, a_min=0, a_max=1.0)
    elif norm_type == 'mean':
        img = img / img.mean()
    return img

def compute_msssim(img1, img2):
    '''
        More accurate SSIM evaluation than skimage
    '''
    ms_ssim_index = piq.multi_scale_ssim(img1, img2, data_range=1.)
    return ms_ssim_index

def compute_roi_mask(roi_volume_labels, label_vol, shape=(256, 256, 89)):
    roi_mask = np.zeros(shape)
    for lab in roi_volume_labels:
        m = np.where(label_vol==lab, 1, 0)
        roi_mask = m + roi_mask
    # dup = np.zeros(shape)
    # dup[:, :, slice_start:slice_end] = roi_mask[:, :, slice_start:slice_end]
    return roi_mask, roi_mask.sum()


def h5_to_dict(h5_obj):
    """
    Recursively convert HDF5 group/file to nested dictionary.
    
    Parameters:
    -----------
    h5_obj : h5py.Group or h5py.File
        HDF5 group or file object to convert
    
    Returns:
    --------
    dict : Nested dictionary with all datasets and groups
    """
    result = {}
    for key in h5_obj.keys():
        item = h5_obj[key]
        
        if isinstance(item, h5py.Group):
            # Recursively convert group to dict
            result[key] = h5_to_dict(item)
        elif isinstance(item, h5py.Dataset):
            # Load dataset as numpy array
            result[key] = item[()]

    return result


def compute_masks(seg_file, target_rois=TARGET_ROIS, threshold=900):
    '''
        Mask combine left & right hemisphere
        Also compute area ratio roi/largest_roi
    '''
    label_vol = nib.load(seg_file).get_fdata()
    target_volume_maps = {}
    target_volume_area_maps = {}
    largest_area = 0
    # print(target_rois.keys())
    for roi in target_rois.keys():
        # print(roi)
        val, roi_area = compute_roi_mask(TARGET_ROIS[roi], label_vol)
        target_volume_maps[roi] = val
        target_volume_area_maps[roi] = roi_area
        if roi_area > largest_area:
            largest_area = roi_area
            
    for roi, a in target_volume_area_maps.items():
        target_volume_area_maps[roi] = a / largest_area
        
    return target_volume_maps, target_volume_area_maps

def compute_hemispheric_masks(seg_file, target_rois=TARGET_LR_ROIS):
    '''
        Separate left & right mask
        return format: 
        {
            'cerebellum': {'left': mask1, 'right': mask2},
            'thalamus': {'left', ... 'right': ...},
            .....
        }
    '''
    label_vol = nib.load(seg_file).get_fdata()
    roi_maps = {}
    for roi in target_rois.keys():
        roi_maps[roi] = {'left': compute_roi_mask(target_rois[roi]['left'], label_vol)[0],
                        'right': compute_roi_mask(target_rois[roi]['right'], label_vol)[0]}
    return roi_maps


def global_bland_altman(combined_df, suffix, target_pet_type, true_pet_type, shared_rois, save_name, ymin, ymax):
    '''
        Bland altman for ROIs, taking a dataframe with all subjects' stats
        Compute stats for 1 model at a time, model type = target_pet_type
        suffix: _suvr or _diff
        _diff is for SUVR asymmetry assessment
        _suvr is for SUVR assessment
    '''
    target_cols = [x+suffix for x in shared_rois]
    pet_pred_df = combined_df[combined_df['pet_type'] == target_pet_type].sort_values(by=['subject'])[target_cols]
    pet_true_df = combined_df[combined_df['pet_type'] == true_pet_type].sort_values(by=['subject'])[target_cols]
    suvr_dict_gt_list = [] # combine all rois
    suvr_dict_pred_list = [] # combine all rois
    suvr_dict_gt = {}
    suvr_dict_pred = {} # {'thalamus': [val1, val2, ... valn], 'hippocampus': [val1, val2, ....]}
    for roi in shared_rois:
        col_val = pet_pred_df[roi+suffix].values
        suvr_dict_pred[roi] = col_val
        suvr_dict_pred_list.extend(col_val)
        col_val = pet_true_df[roi+suffix].values
        suvr_dict_gt[roi] = col_val
        suvr_dict_gt_list.extend(col_val)
        
    # color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', 'lightblue', 'limegreen', 'black', 'maroon']
    plt.figure(figsize=(8,6))
    print(suvr_dict_gt.keys())
    print(suvr_dict_pred.keys())
    if ymin is not None:
        plt.ylim(ymin, ymax)
    for i, lab in enumerate(suvr_dict_gt.keys()):
        # print(suvr_dict_gt[lab], suvr_dict_pred[lab])
        plt.plot(0.5*(np.array(suvr_dict_gt[lab])+np.array(suvr_dict_pred[lab])), np.array(suvr_dict_gt[lab])-np.array(suvr_dict_pred[lab]), '.', alpha=0.6, color=color_roi_mapping[lab], label=lab, markersize=10)
        # plt.plot(0.5*(np.array(suvr_dict_gt[lab])+np.array(suvr_dict_pred[lab])), np.array(suvr_dict_gt[lab])-np.array(suvr_dict_pred[lab]), '.', alpha=0.6, color=color_roi_mapping[lab], markersize=10)

    
    sd196 = round(1.96*np.std(np.array(suvr_dict_gt_list)-np.array(suvr_dict_pred_list)), 2)
    mean = round(np.mean(np.array(suvr_dict_gt_list)-np.array(suvr_dict_pred_list)), 2)
    xmin = round(0.5*np.min(np.array(suvr_dict_gt_list)+np.array(suvr_dict_pred_list)), 2)
    xmax = round(0.5*np.max(np.array(suvr_dict_gt_list)+np.array(suvr_dict_pred_list)), 2)
    plt.plot([xmin, xmax], [mean+sd196, mean+sd196], color='tab:orange', linestyle='dotted')
    if sd196 >= 0.1:
        offset = 0.02
    else:
        offset = 0.01
    plt.text(xmax+0.05, mean+sd196+offset, "%.2f" % (mean+sd196), fontsize=12, weight='bold')
    plt.plot([xmin, xmax], [mean-sd196, mean-sd196], color='tab:orange', linestyle='dotted')
    plt.text(xmax+0.05, mean-sd196+offset, "%.2f" % (mean-sd196), fontsize=12, weight='bold')
    plt.plot([xmin, xmax], [mean, mean], color='tab:purple', linestyle='dotted')
    plt.legend(ncol=2)
    plt.text(xmax+0.05, mean+offset, "%.2f" % mean, fontsize=12, weight='bold')
    plt.xlabel('Mean of methods', fontsize=14)
    plt.ylabel('Acquired - Synthesized SUVR', fontsize=14)
    # plt.savefig('global_bland_altman.png')
    plt.legend(loc='upper left', bbox_to_anchor=(1.1, 1))
    plt.title(f"Global Bland Altman for Model {exp_type}", y=1.03)
    # Adjust the plot to make room for the legend
    plt.tight_layout()
    if save_name:
        plt.savefig(save_name, bbox_inches='tight', dpi=300)
        print(f"Saved at {save_name}")
    else:
        plt.show()


def compute_suvr(roi_masks, ref_name, pet, include_diff=True): # {pet1: {roi1: {'suvr': xx, ...}}, pet2: {roi1: {'suvr': xx}}}
    ref_mask = roi_masks[ref_name]
    if type(pet) == dict:
        res = {}
        res_diff = {}
        pet_data = pet['truth']
        res['truth'] = {}
        suv_ref = (pet_data * ref_mask).sum() / ref_mask.sum() # cerebellum total suv / cerebellum total volume
        for roi in roi_masks.keys():
            curr_roi_mask = roi_masks[roi]
            if curr_roi_mask.sum() == 0:
                print(f"{roi} has 0 sum?????")
            suv_roi = (pet_data * curr_roi_mask).sum() / curr_roi_mask.sum() # roi total suv / roi total volume
            suvr_roi = suv_roi / (suv_ref)
            res['truth'][roi] = {'suvr': suvr_roi}
        for p in pet.keys():
            if p == 'truth':
                continue
            res[p] = {}
            pet_data = pet[p]
            suv_ref = (pet_data * ref_mask).sum() / ref_mask.sum() # cerebellum total suv / cerebellum total volume
            for roi in roi_masks.keys():
                curr_roi_mask = roi_masks[roi]
                suv_roi = (pet_data * curr_roi_mask).sum() / curr_roi_mask.sum() # roi total suv / roi total volume
                suvr_roi = suv_roi / (suv_ref)
                res[p][roi] = {'suvr': suvr_roi}
                if include_diff:
                    if p not in res_diff:
                        res_diff[p] = {}
                    res_diff[p][roi] = res['truth'][roi]['suvr'] - suvr_roi # how suvr deviates from the true pet
        return res, res_diff
    else:
        raise NotImplementedError
    
def compute_metrics(all_pets):
    res = {}
    truth = all_pets['truth'][:, :, 20:71]
    for pet in all_pets.keys():
        if pet == 'truth':
            continue
        curr_pred_img = all_pets[pet][:, :, 20:71]
        psnr = skimage.metrics.peak_signal_noise_ratio(truth, curr_pred_img, data_range=1)
        ssim = skimage.metrics.structural_similarity(truth, curr_pred_img, data_range=1)
        rmse = skimage.metrics.normalized_root_mse(truth, curr_pred_img)

        slice_true_torch = torch.Tensor(truth).unsqueeze(0).permute(3, 0, 1, 2)
        slice_pred_torch = torch.Tensor(curr_pred_img).unsqueeze(0).permute(3, 0, 1, 2)
        msssim = piq.multi_scale_ssim(slice_true_torch, slice_pred_torch, data_range=1.)
        res[pet] = {'psnr': psnr, 'rmse': rmse, 'ssim': ssim, 'msssim': msssim}
        print(f"PET: {pet}, PSNR: {psnr}, MSSSIM: {msssim}, RMSE: {rmse}, SSIM: {ssim}")
    return res

def get_congruence_mae_scaled_per_roi(info_dict, area_info_dict):
    '''
        Asymmetry mse per subject
        return in the form {
            'pet_type1': val1,
            'pet_type2': val2,
            ...
        }
    '''
    true_asym = info_dict['truth']
    res = {}
    for pet_type in list(info_dict.keys()):
        if pet_type == 'truth':
            continue
        pred_asym = info_dict[pet_type]
        curr = {}
        for roi in true_asym.keys():
            # print(roi)
            true_asym_roi = true_asym[roi]['diff']
            pred_asym_roi = pred_asym[roi]['diff']
            se = abs(true_asym_roi - pred_asym_roi)
            if roi == 'cerebral_white_matter':
                scale = max(area_info_dict["left_cerebral_white_matter"], area_info_dict["right_cerebral_white_matter"]) # in area info this is separated so select the larger one
            else:
                scale = area_info_dict[roi]
            curr[roi] = se * scale
        res[pet_type] = curr
    return res

def get_congruence_mae_scaled(info_dict, area_info_dict):
    '''
        Asymmetry mse per subject
        return in the form {
            'pet_type1': val1,
            'pet_type2': val2,
            ...
        }
    '''
    true_asym = info_dict['truth']
    res = {}
    for pet_type in list(info_dict.keys()):
        if pet_type == 'truth':
            continue
        pred_asym = info_dict[pet_type]
        total_error = 0
        count = 0
        for roi in true_asym.keys():
            # print(roi)
            true_asym_roi = true_asym[roi]['diff']
            pred_asym_roi = pred_asym[roi]['diff']
            se = abs(true_asym_roi - pred_asym_roi)
            if roi == 'cerebral_white_matter':
                scale = max(area_info_dict["left_cerebral_white_matter"], area_info_dict["right_cerebral_white_matter"]) # in area info this is separated so select the larger one
            else:
                scale = area_info_dict[roi]
            total_error = se * scale
            # count += 1
        res[pet_type] = total_error
    return res



def get_congruence_mae(info_dict):
    '''
        Asymmetry mse per subject
        return in the form {
            'pet_type1': val1,
            'pet_type2': val2,
            ...
        }
    '''
    true_asym = info_dict['truth']
    res = {}
    for pet_type in list(info_dict.keys()):
        if pet_type == 'truth':
            continue
        pred_asym = info_dict[pet_type]
        total_error = 0
        count = 0
        for roi in true_asym.keys():
            true_asym_roi = true_asym[roi]['diff']
            pred_asym_roi = pred_asym[roi]['diff']
            se = abs(true_asym_roi - pred_asym_roi)
            total_error += se
            count += 1
        res[pet_type] = total_error / count
    return res

def get_congruence_index_per_roi(info_dict, area_info_dict):
    '''
        Asymmetry index per subject
        return in the form {
            'pet_type1': {'roi1': xxx, 'roi2': xxx}
            'pet_type2': {}
            ...
        }
    '''
    true_asym = info_dict['truth']
    res = {}
    for pet_type in list(info_dict.keys()):
        curr = {}
        if pet_type == 'truth':
            continue
        pred_asym = info_dict[pet_type]
        for roi in true_asym.keys():
            true_asym_roi = true_asym[roi]['diff']
            pred_asym_roi = pred_asym[roi]['diff']
            # check if they have the same sign
            if roi == 'cerebral_white_matter':
                scale = max(area_info_dict["left_cerebral_white_matter"], area_info_dict["right_cerebral_white_matter"]) # in area info this is separated so select the larger one
            else:
                scale = area_info_dict[roi]
            if true_asym_roi * pred_asym_roi > 0:
                curr[roi] = (1 * scale)
            else:
                curr[roi] = (-1 * scale)
            
        res[pet_type] = curr
    return res
    


def get_congruence_index(info_dict, area_info_dict):
    '''
        Asymmetry index per subject
        return in the form {
            'pet_type1': val1,
            'pet_type2': val2,
            ...
        }
    '''
    true_asym = info_dict['truth']
    res = {}
    total = 0
    for pet_type in list(info_dict.keys()):
        if pet_type == 'truth':
            continue
        pred_asym = info_dict[pet_type]
        correct = 0
        incorrect = 0
        for roi in true_asym.keys():
            true_asym_roi = true_asym[roi]['diff']
            pred_asym_roi = pred_asym[roi]['diff']
            # check if they have the same sign
            if roi == 'cerebral_white_matter':
                scale = max(area_info_dict["left_cerebral_white_matter"], area_info_dict["right_cerebral_white_matter"]) # in area info this is separated so select the larger one
            else:
                scale = area_info_dict[roi]
            if true_asym_roi * pred_asym_roi > 0:
                correct += 1
            total += 1
        res[pet_type] = correct / total
    return res
        


def compute_asymmetry(hemisphere_roi_masks, global_roi_masks, pet, area_info, include_diff=True, ref_name='cerebellum', per_roi=False):
    print("Inside compute asymmetry")
    ref_mask = global_roi_masks[ref_name]
    res = {}
    pet_data = pet['truth']
    res['truth'] = {}
    res_diff = {}
    suv_ref_global = (pet_data * ref_mask).sum() / ref_mask.sum() # global reference, i.e., both left & right cerebellum
    print(suv_ref_global)
    for roi in hemisphere_roi_masks.keys():
        curr_roi_mask_left = hemisphere_roi_masks[roi]['left']
        curr_roi_mask_right = hemisphere_roi_masks[roi]['right']
        suv_roi_left = (pet_data * curr_roi_mask_left).sum() / curr_roi_mask_left.sum() # roi total suv / roi total volume
        suvr_roi_left = suv_roi_left / (suv_ref_global)
        suv_roi_right = (pet_data * curr_roi_mask_right).sum() / curr_roi_mask_right.sum() # roi total suv / roi total volume
        suvr_roi_right = suv_roi_right / (suv_ref_global)
        res['truth'][roi] = {'left': suvr_roi_left,
                            'right': suvr_roi_right,
                            'diff': (suvr_roi_left - suvr_roi_right) / (suvr_roi_left + suvr_roi_right)}
    for p in pet.keys():
        if p == 'truth':
            continue
        res[p] = {}
        pet_data = pet[p]
        suv_ref_global = (pet_data * ref_mask).sum() / ref_mask.sum() # cerebellum total suv / cerebellum total volume
        for roi in hemisphere_roi_masks.keys():
            curr_roi_mask_left = hemisphere_roi_masks[roi]['left']
            curr_roi_mask_right = hemisphere_roi_masks[roi]['right']
            suv_roi_left = (pet_data * curr_roi_mask_left).sum() / curr_roi_mask_left.sum() # roi total suv / roi total volume
            suvr_roi_left = suv_roi_left / (suv_ref_global)
            
            suv_roi_right = (pet_data * curr_roi_mask_right).sum() / curr_roi_mask_right.sum() # roi total suv / roi total volume
            suvr_roi_right = suv_roi_right / (suv_ref_global)
            res[p][roi] = {'left': suvr_roi_left,
                           'right': suvr_roi_right,
                           'diff': (suvr_roi_left - suvr_roi_right) / (suvr_roi_left + suvr_roi_right)}
            if include_diff:
                if p not in res_diff:
                    res_diff[p] = {}
                res_diff[p][roi] = {'left': res['truth'][roi]['left'] - res[p][roi]['left'],
                                   'right': res['truth'][roi]['right'] - res[p][roi]['right'],
                                    'diff': res['truth'][roi]['diff'] - res[p][roi]['diff'],
                                   }

    congruence_index = get_congruence_index(res, area_info)
    congruence_mae_scaled = get_congruence_mae_scaled(res, area_info)
    
    return res, res_diff, congruence_index, congruence_mae_scaled


def get_histogram(data, num_bins=128, scale=10, output='norm'):
     # calculate histogram
    histograms, _ = np.histogram(data, bins=num_bins, range=(0.001, 1))
    normalized_histograms = histograms / (histograms.sum(keepdims=True) + 1e-4)
    normalized_histograms *= scale
    if output == 'norm':
        return normalized_histograms
    cum_hist = np.cumsum(normalized_histograms)
    hist_diff = np.diff(normalized_histograms)
    hist_diff = np.insert(hist_diff, 0, hist_diff[0])
    hist_diff *= scale
    combined_histogram = np.stack((normalized_histograms, cum_hist, hist_diff), axis=0) # 3, 128
    # assert combined_histogram.shape == (3, 128)
    return combined_histogram
    
def compare_histograms_w_plot_asym(vol1, vol2, roi_name, pet_type, hist_truth_left, hist_truth_right, emd_truth, axes, num_bins=128):
    '''
        Evaluate histogram with Wasserstein loss, can be a single slice or a whole volume
    '''
    hist1 = get_histogram(vol1, num_bins, output='all')[0]
    hist2 = get_histogram(vol2, num_bins, output='all')[0]
    emd = wasserstein_distance(hist1, hist2)
    emd_diff = emd_truth - emd
    if axes is not None:
        axes[0].bar(range(num_bins), hist1, width=3, label=roi_name+'_pred')
        axes[1].bar(range(num_bins), hist2, width=3, label=roi_name+'_pred')
        axes[2].bar(range(num_bins), hist_truth_left, width=3, label=roi_name)
        axes[3].bar(range(num_bins), hist_truth_right, width=3, label=roi_name)
    # plt.suptitle(f"{pet_type} {roi_name} Asymmetry Wasserstein {emd} VS {emd_truth}")
    # Wasserstein distance
    print(f'{pet_type} {roi_name} EMD: {emd}')
    return emd, emd_diff

def compare_histogram_asymmetry(roi_masks, pets, target_rois, axes, num_bins=128):
    res = {}
    res_diff = {}
    for roi in target_rois:
        val = roi_masks[roi]
        roi_mask_left = val['left']
        roi_mask_right = val['right']
        hist_truth_left = get_histogram(pets['truth'] * roi_mask_left, num_bins=num_bins)
        hist_truth_right = get_histogram(pets['truth'] * roi_mask_right, num_bins=num_bins)
        emd_truth = wasserstein_distance(hist_truth_left, hist_truth_right)
        for pet_type, pet_vol in pets.items():
            if pet_type not in res:
                res[pet_type] = {}
                res_diff[pet_type] = {}
            if pet_type == 'truth':
                continue
            vol_left = pet_vol * roi_mask_left
            vol_right = pet_vol * roi_mask_right
            emd, emd_diff = compare_histograms_w_plot_asym(vol_left, vol_right, roi, pet_type, hist_truth_left, hist_truth_right, emd_truth, axes)
            res[pet_type][roi] = emd
            res_diff[pet_type][roi] = emd_diff
    return res, res_diff

def compare_histogram_roi(roi_masks, pets, target_rois, axes=None, num_bins=128):
    res = {}
    for i, roi in enumerate(target_rois):
        m = roi_masks[roi]
        hist_truth = get_histogram(pets['truth'] * m, num_bins=num_bins)
        for pet_type, pet_vol in pets.items():
            if pet_type == 'truth':
                continue
            if pet_type not in res:
                res[pet_type] = {}
            curr_hist = get_histogram(pet_vol * m, num_bins=num_bins)
            if axes is not None:
                axes[i].bar(range(num_bins), hist_truth, width=3, label=roi+'_acq', color='red')
                axes[i].bar(range(num_bins), curr_hist, width=3, label=roi+'_synth', color='blue')
            emd = wasserstein_distance(hist_truth, curr_hist)
            res[pet_type] = emd
    return res
    
    

def analyze_one_subj(subj, pet_types, save_dest, seg, *test_dirs):
    '''
        Computes SUVR, PSNR, RMSE, MSSSIM, SUVR-asymmetry, histogram-asymmetry
    '''
    seg_file = os.path.join("xx", subj, seg)
    roi_masks_global, area_info = compute_masks(seg_file)
    roi_masks_left_right = compute_hemispheric_masks(seg_file)
    pet_truth = sorted(glob.glob(os.path.join("/data/jiaqiw01/preprocessed_cases", subj, 'reslice_PET_full.nii')))
    if len(pet_truth) == 0:
        pet_truth = sorted(glob.glob(os.path.join("/data/jiahong/data/FDG_PET_preprocessed", subj, 'reslice_PET_*.nii')))
        
    pet_truth = pet_truth[0]
    print(pet_truth)
    pets = {'truth': nib.load(pet_truth).get_fdata()}
    # normalize to 0-1
    temp = np.zeros((256, 256, 89))
    temp[:, :, slice_start:slice_end] = pets['truth'][:, :, slice_start:slice_end]
    pets['truth'] = norm(temp)

    # fig3, axes = plt.subplots(nrows=len(test_subjects), ncols=4, figsize=(15, 25))
    # axes_flatten_histogram = axes.ravel()
    
    # idx = 1
    for i, directory in enumerate(test_dirs):
        pet_data = list(glob.glob(os.path.join(directory, subj) + "/*.nii"))
        if pet_data == []:
            print(f"Could not find predicted pet file for {subj}")
            continue
        curr_pred_pet = nib.load(pet_data[0]).get_fdata()
        assert curr_pred_pet.shape == (256, 256, 89), "Synthetic pet shape is not 256, 256, 89"
        temp = np.zeros((256, 256, 89))
        temp[:, :, slice_start:slice_end] = curr_pred_pet[:, :, slice_start:slice_end]
        # normalize to 0-1
        pets[pet_types[i]] = norm(temp)

    suvr, suvr_diff = compute_suvr(roi_masks_global, 'cerebellum', pets, include_diff=True)
    # all_metrics = compute_metrics(pets)
    all_metrics = None
    suvr_asym, suvr_asym_diff, asym_congruency, asym_se = compute_asymmetry(roi_masks_left_right, roi_masks_global, pets, area_info, per_roi=True)
    # emd_roi = compare_histogram_roi(roi_masks_global, pets, target_rois)
    
    # one subject stats only!                                
    return {'suvr': suvr, 
            'suvr_diff': suvr_diff, 
            'all_metrics': all_metrics, 
            'suvr_asym': suvr_asym, 
            'suvr_asym_diff': suvr_asym_diff, 
            'asym_congruency': asym_congruency,
            'asym_mae': asym_se,
            'pets': pets
           }


def write_to_h5(subj, roi_mask_global, roi_mask_left_right, area_info, fname='subject_masks_lateralized.h5'):
    with h5py.File(fname, 'a') as f:
        # for subj, data in all_subj_mask_info.items():
        if subj in f.keys():
            subj_group = f[subj]
        else:
            subj_group = f.create_group(subj)
        
        mask_group = subj_group.create_group('mask_global')
        for key, array in roi_mask_global.items():
            mask_group.create_dataset(key, data=array)
        lr_group = subj_group.create_group('mask_left_right')
        for roi_key, vals in roi_mask_left_right.items():
            print(roi_key, vals.keys())
            sub = lr_group.create_group(roi_key)
            sub.create_dataset('left', data=vals['left'])          
            sub.create_dataset('right', data=vals['right'])

        if 'area_info' in subj_group.keys():
            area_group = subj_group['area_info']
        else:
            area_group = subj_group.create_group('area_info')
        for key, val in area_info.items():
            area_group.create_dataset(key, data=val)
        print("Added to h5...")

        