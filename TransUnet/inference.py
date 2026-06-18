# new version, 2020/08/19

import os
import glob
import time
import json
import torch
import torch.optim as optim
import numpy as np
import yaml
import pdb
import psutil
import nibabel as nib
import skimage.metrics
import argparse
from model_simple import *
from model import *

# set seed
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic=True

parser = argparse.ArgumentParser(prog='ProgramName', description='What the program does')
parser.add_argument('-c', '--config')
args = parser.parse_args()
yml = args.config
print(f"Using {yml} as model yaml")

# set config param
def load_config_yaml(yaml_path):
    if os.path.exists(yaml_path):
        with open(yaml_path, 'r') as file:
            config = yaml.safe_load(file)
        return True, config
    else:
        return False, None
    
_, config = load_config_yaml(yml)
config['in_num_ch'] = len(config['contrast_list']) * (2*config['block_size']+1)
config['device'] = torch.device('cuda:'+ config['gpu'])
if config['is_template']:
    config['input_height'], config['input_width'] = 160, 192
else:
    config['input_height'], config['input_width'] = 256, 256
    if config['is_symmetry']:
        print('Raw space should not use symmetry')
        config['is_symmetry'] = False

# define model
if config['norm_type'] == 'z-score':
    config['target_output_act'] = 'no'
else:
    config['target_output_act'] = 'softplus'

if config['model_type'] == 'TransUNET':
    print("Instantiating a trans unet model..")
    model = TransUNet(in_num_ch=config['in_num_ch'], out_num_ch=config['out_num_ch'], 
                    first_num_ch=64, input_size=(config['input_height'], config['input_width']), 
                    is_symmetry=config['is_symmetry'], 
                    output_activation=config['target_output_act'], is_transformer=config['is_transformer']).to(config['device'])
elif config['model_type'] == 'GANSplit': # this uses symmetry gated, not suitable for rawspace
    print(f"Input channel number: {config['in_num_ch']}")
    model = GANShortGeneratorWithSplitInputChannelAttentionAllAndSpatialAttention(in_num_ch=config['in_num_ch'], out_num_ch=config['out_num_ch'],).to(config['device'])
    print("Using GAN Split Model....")
else:
    print("Instantiating a GAN model...")
    model = GANShortGeneratorWithChannelAttentionAllAndSymmetrySpatialAttention(in_num_ch=config['in_num_ch'], out_num_ch=config['out_num_ch'], 
                                                                                first_num_ch=64, input_size=(config['input_height'], config['input_width']), 
                                                                                output_activation=config['target_output_act'], is_transformer=config['is_transformer']).to(config['device'])
model.eval()

# load pretrained model
if os.path.isfile(config['pretrained_ckpt_path']):
    checkpoint = torch.load(config['pretrained_ckpt_path'], map_location=config['device'])
    try:
        def load_checkpoint_model(model, pretrained_dict):
            model_dict = model.state_dict()
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape==model_dict[k].shape}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # 3. load the new state dict
            model.load_state_dict(model_dict)
            return model

        model = load_checkpoint_model(model, checkpoint['model'])
        print('loading pretrained model success!' + config['pretrained_ckpt_path'])
    except:
        print('loading pretrained model failed!')
else:
    raise ValueError('No correct checkpoint')


brain_mask_nib = nib.load(config['brain_mask_path'])
brain_mask = brain_mask_nib.get_fdata()

def preprocess_data(img, config):
    raw_h, raw_w = img.shape[0], img.shape[1]

    # check invalid image value
    if np.nanmax(img) == 0 or np.isnan(img[:,:,20:-20]).sum()>100000:
        raise ValueError('Invalid image nanmax or nan count')

    img = np.nan_to_num(img, nan=0.)

    if config['is_template']:
        img = img * brain_mask
    img[img<0] = 0

    # check image size
    if config['input_height'] == 160 and config['input_width'] == 192:
        if img.shape != (157, 189, 156):
            raise ValueError('Invalid image shape')
        img = np.concatenate([img, np.zeros((3,189,156))], 0)     # pad (157,189) -> (160,192)
        img = np.concatenate([img, np.zeros((160,3,156))], 1)
    else:
        if img.shape != (256, 256, 89):
            raise ValueError('Invalid image shape')

    norm = img.mean()
    # img = img / norm  # norm by dividing mean
    std = img.std()
    img_max = np.percentile(img, 99.9)
    img_min = np.percentile(img, 0.1)
    if config['norm_type'] == 'z-score':
        img = (img - norm) / (std + 1e-8)
    elif config['norm_type'] == 'max':
        img = (img - img_min) / (img_max - img_min)
        img = np.clip(img, a_max=1., a_min=0.)
    elif config['norm_type'] == 'mean':
        img  = img / norm
    else:
        raise ValueError('Invalid norm type')
    return img, raw_h, raw_w, [norm, std, img_max, img_min]

def postprocess_data(img, img_stat, norm_type):
    if norm_type == 'z-score':
        img = img * img_stat[1] + img_stat[0]
    elif norm_type == 'max':
        img = np.clip(img, a_max=1., a_min=0.)
        img = img * (img_stat[2] - img_stat[3]) + img_stat[3]
    elif norm_type == 'mean':
        img  = img * img_stat[0]
    return img


def select_input_contrast(case_path, config):
    contrast_selected_list = []
    for contrast_idx in range(len(config['contrast_list'])):
        contrast, contrast_name = config['contrast_list'][contrast_idx], config['contrast_name_list'][contrast_idx]
        # print("++++++++++++ Contrast ++++++++++++")
        img_paths = glob.glob(os.path.join(case_path, contrast_name + '.nii'))

        if len(img_paths) == 0:
            img_path = "blank"
            print("Blank image path?")
        elif len(img_paths) == 1:
            img_path = img_paths[0]
        else:        
            # choose T1_GRE over T1_SE if both exist    
            if contrast == 'T1' or contrast_name == 'T1c':
                pdb.set_trace()
                if os.path.exists(os.path.join(case_path, contrast_name.split('*')[0] + '_GRE.nii')):
                    img_path = os.path.join(case_path, contrast_name.split('*')[0] + '_GRE.nii')
                else:
                    img_path = img_paths[0]
            # choose T2_FLAIR over T2_FLAIR_2D if both exist    
            elif contrast == 'T2_FLAIR':
                if os.path.exists(os.path.join(case_path, contrast_name.split('*')[0] + '.nii')):
                    img_path = os.path.join(case_path, contrast_name.split('*')[0] + '.nii')
                else:
                    img_path = img_paths[0]
        contrast_selected_list.append(img_path)
    print("------Selected Contrast------")
    print(contrast_selected_list)
    return contrast_selected_list
            

def select_target_contrast(case_id, case_path, config):
    if len(case_id) == 5:
        img_paths = glob.glob(os.path.join(case_path, 'PET_full.nii'))
    else:
        img_paths = glob.glob(os.path.join(case_path, config['target_contrast_name'] + '.nii'))

    if len(img_paths) == 0:
        img_path = None
    elif len(img_paths) == 1:
        img_path = img_paths[0] # for new sdc cases
    else: # for old cases with multiple PET recons
        if os.path.exists(os.path.join(case_path, config['target_contrast_name'].split('*')[0] + '_MAC.nii')):
            img_path = os.path.join(case_path, config['target_contrast_name'].split('*')[0] + '_MAC.nii')
        elif os.path.exists(os.path.join(case_path, config['target_contrast_name'].split('*')[0] + '_QCLEAR.nii')):
            img_path = os.path.join(case_path, config['target_contrast_name'].split('*')[0] + '_QCLEAR.nii')
        elif os.path.exists(os.path.join(case_path, config['target_contrast_name'].split('*')[0] + '_full.nii')):
            img_path = os.path.join(case_path, config['target_contrast_name'].split('*')[0] + '_full.nii')
        else:
            img_path = img_paths[0]
    return img_path
        

# get list of tested cases
# pdb.set_trace()
# if config['test_cases_list'][0] == '*': 
#     cases_paths = glob.glob(config['test_cases_dir']+'*')
# else:
#     cases_paths = []
#     for case_id in config['test_cases_list']:
#         if len(case_id) == 5:
#             cases_paths.append(os.path.join("/data/jiaqiw01/preprocessed_cases", case_id))
#         else:
#             cases_paths.append(os.path.join("/data/jiahong/data/FDG_PET_preprocessed", case_id))
cases_paths = []
curr_contrast = 'T1'
with open("/data/jiaqiw01/PET_MRI/data/slices/debug/T1_PET_rawspace_fdg_only/info.json", 'r') as fr:
    info = json.load(fr)
    train_subj = info['all_train_subj']
    val_subj = info['all_val_subj']
    all_subjects = train_subj + val_subj
    for case_id in all_subjects:
        if len(case_id) == 5:
            cases_paths.append(os.path.join("/data/jiaqiw01/preprocessed_cases", case_id))
        else:
            cases_paths.append(os.path.join("/data/jiahong/data/FDG_PET_preprocessed", case_id))

if len(cases_paths) == 0:
    print('None tested cases found')
else:
    print('Total number of tested cases: ', len(cases_paths))

# loop on selected cases in the folder
for case_path in cases_paths:
    case_id = os.path.basename(case_path)
    print('Start', case_id )

    # preprocess input
    input_list = []
    input_path_list = select_input_contrast(case_path, config)
    # pdb.set_trace()
    for input_path in input_path_list:
        if input_path == 'blank':
            img = np.zeros((256, 256, 89))
        elif 'predicted_10p' in input_path:
            img = nib.load(input_path).get_fdata()
            print(f"Skipped preprocessing for {input_path}")
        else:
            img = nib.load(input_path)
            affine = img.affine
            img = img.get_fdata()
            img, raw_h, raw_w, _ = preprocess_data(img, config)
            # print(img.shape)
        input_list.append(img)
    
    input_list = np.stack(input_list, axis=0)  # (4, 160, 192, 156)
    # print("++++++++++++++++++")
    # print(input_list.shape)

    # make preduction
    pred_list = []
    for slc_idx in range(input_list.shape[-1]):
        slc_idx_sel = np.clip(np.arange(slc_idx-config['block_size'], slc_idx+config['block_size']+1), 0, input_list.shape[-1]-1)
        input = input_list[:,:,:,slc_idx_sel]
        input = torch.from_numpy(input).to(config['device'], dtype=torch.float)  # (4, 160, 192, 3)
        input = input.permute(0,3,1,2).contiguous().view(-1, config['input_height'], config['input_width']).unsqueeze(0)  # (1, 12, 160, 192)
        pred, _ = model(input)
        pred_list.append(pred.detach().cpu().numpy().squeeze(0).squeeze(0))  # (160, 192)
        # print("Checking nan value...")
        if np.isnan(pred_list[-1]).any():
            print(f"Got a nan value in {slc_idx}")

    pred_pre = np.stack(pred_list, axis=0) # 89, 256, 256
    assert pred_pre.shape == (89, 256, 256)
    max_normed_pred = (pred_pre - pred_pre.min()) / (pred_pre.max() - pred_pre.min())

    f = h5py.File("/data/jiaqiw01/PET_MRI/data/all_case_synthetics.h5", 'a')
    subj_data = f.require_group(case_id)  # Create subject group if missing
    transunet_group = subj_data.require_group('TransUnet')  # Ensure TransUnet exists
    transunet_group.create_dataset(curr_contrast, data=max_normed_pred)
    # if case_id not in f:
    #     subj_data = f.create_group(case_id)
    # else:
    #     subj_data = f[case_id]
    # if 'TransUnet' not in subj_data:
    #     transunet_group = subj_data.create_group('TransUnet')
    # else:
    #     transunet_group = subj_data['TransUnet']
    # transunet_group.create_dataset(curr_contrast, data=max_normed_pred)

    # evaluate metrics if available
    # target_path = select_target_contrast(case_id, case_path, config)
    # if target_path:
    #     target = nib.load(target_path)
    #     affine = target.affine
    #     target = target.get_fdata()
    #     target_pre, raw_h, raw_w, target_norm_stat = preprocess_data(target, config)  # (160, 192, 156)
    #     pred_list = pred_pre[:raw_h, :raw_w]  # (157, 189, 156)
    #     target_list = target_pre
    #     # target_list = postprocess_data(target_pre, target_norm_stat, config['norm_type'])[:raw_h, :raw_w]  # (157, 189, 156)
    #     # print(len(pred_list), len(target_list))
    #     print(skimage.metrics.peak_signal_noise_ratio(target_list, pred_list, data_range=target_list.max()))
    #     print(skimage.metrics.structural_similarity(target_list, pred_list, data_range=target_list.max()))

    # # save prediction
    # if not os.path.exists(os.path.join(config['test_save_dir'], case_id)):
    #     os.makedirs(os.path.join(config['test_save_dir'], case_id))
    # pred_nib = nib.Nifti1Image(pred_list, affine)
    # nib.save(pred_nib, os.path.join(config['test_save_dir'], case_id, 'pred.nii'))

