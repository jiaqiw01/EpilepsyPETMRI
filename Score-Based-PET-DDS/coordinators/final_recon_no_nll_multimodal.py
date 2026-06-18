import hydra
import torch
import functools
import numpy as np 
import yaml
import nibabel as nib
import glob
import sys, os
sys.path.append(os.path.dirname(os.getcwd()))
from src import (BrainWebOSEM, PETMRTest, get_standard_score, get_standard_sde, 
        get_standard_sampler, osem_nll, get_osem, get_map, get_anchor, kl_div)
from omegaconf import DictConfig, OmegaConf
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import time
import matplotlib.pyplot as plt 

import cupy as xp

detector_efficiency = 1./30

def get_acq_model():
    import pyparallelproj.coincidences as coincidences
    import pyparallelproj.petprojectors as petprojectors
    import pyparallelproj.resolution_models as resolution_models
    import cupyx.scipy.ndimage as ndi
    """
    create forward operator
    """
    coincidence_descriptor = coincidences.GEDiscoveryMICoincidenceDescriptor(
        num_rings=1,
        sinogram_spatial_axis_order=coincidences.SinogramSpatialAxisOrder['RVP'],xp=xp)
    acq_model = petprojectors.PETJosephProjector(coincidence_descriptor,
        (128, 128, 1), (-127.0, -127.0, 0.0), (2., 2., 2.))
    res_model = resolution_models.GaussianImageBasedResolutionModel(
        (128, 128, 1), tuple(4.5 / (2.35 * x) for x in (2., 2., 2.)), xp, ndi)
    acq_model.image_based_resolution_model = res_model
    return acq_model

def estimate_scale_factor(osem, measurements, contamination, normalisation_type):
    scale_factors = []		
    for i in range(osem.shape[0]):
        if normalisation_type == "data_scale":
            emission_volume = torch.where(osem[i] > 0.01*osem[i].max(), 1, 0).sum() * 8
            scale_factor = (measurements[i] - contamination[i]).sum()/emission_volume
            scale_factors.append(scale_factor)
        elif normalisation_type == "image_scale":
            emission_volume = torch.where(osem[i] > 0.01*osem[i].max(), 1, 0).sum()
            scale_factor = osem[i].sum()/emission_volume
            scale_factors.append(scale_factor)		
        else:
            raise NotImplementedError
    return torch.tensor(scale_factors)


def to_nii(img_list, subj, dest):
    path = os.path.join("/data/jiaqiw01/preprocessed_cases", subj, 'reslice_PET_full.nii')
    if not os.path.exists(path):
        path = glob.glob(os.path.join("/data/jiahong/data/FDG_PET_preprocessed", subj, 'reslice_PET*.nii'))[0]
        print(f"Using {path} as true pet..")
    true_pet = nib.load(path)
    aff = true_pet.affine
    img_arr = np.stack(img_list, axis=2)
    print(img_arr.shape)
    # assert img_arr.shape == (256, 256, 89)
    nif = nib.Nifti1Image(img_arr, affine=aff)
    d = os.path.join(dest, subj)
    if not os.path.exists(d):
        os.makedirs(d)
    nib.save(nif, os.path.join(d, 'pred.nii'))
    print(f"Nifti saved at {d}")


@hydra.main(config_path='../configs', config_name='final_reconstruction_petmr_multimodal')
def reconstruction(config : DictConfig) -> None:
    print(OmegaConf.to_yaml(config))

    # Generate a unique filename, the file directories specify the SDE and data
    name = ""
    results = {}
    if config.sampling.use_osem_nll:
        name = "OSEMNLL_"
    if config.sampling.name == "dps" or config.sampling.name == "naive":
        name = name + "penalty_" + str(config.sampling.penalty) + "_"
        results["penalty"] = config.sampling.penalty

    if "guided" in config.score_based_model.name:
        name = name  + "_gstrength_" + str(config.sampling.guidance_strength) + "_"
        results["gstrength"] = config.sampling.guidance_strength

    timestr = time.strftime("%Y%m%d_%H%M%S_")
    config.device = 'cuda:1'
    ###### SET SEED ######
    torch.manual_seed(42)
    np.random.seed(42)
    ###### GET SCORE MODEL ######
    # open the yaml config file
    with open(os.path.join(config.score_based_model.path, "report.yaml"), "r") as stream:
        ml_collection = yaml.load(stream, Loader=yaml.UnsafeLoader)
    guided = False if ml_collection.guided_p_uncond is None else True
    with open(os.path.join(config.t2f_model_path, "report.yaml"), "r") as stream:
        ml_collection_2 = yaml.load(stream, Loader=yaml.UnsafeLoader)
    # get the sde
    sde = get_standard_sde(ml_collection)
    # get the score model
    score_model_t1 = get_standard_score(ml_collection, sde, 
                    ema_ckpt=config.ema_ckpt,
                    model_ckpt=config.model_ckpt,	
                    use_ema = config.score_based_model.ema, 
                    load_path = config.score_based_model.path)
    score_model_t1.eval()
    score_model_t1.to(config.device)
    
    score_model_t2f = get_standard_score(ml_collection_2, sde, 
                    ema_ckpt=config.ema_ckpt,
                    model_ckpt=config.model_ckpt,	
                    use_ema = True, 
                    load_path = config.t2f_model_path)
    score_model_t2f.eval()
    score_model_t2f.to(config.device)

    ###### GET ACQUISITION MODEL AND DATA ######
    # get the data
    dataset = PETMRTest(data_h5="/data/jiaqiw01/PET_MRI/data/all_cases_rawspace_complete_max.h5", 
                    subjects=['25013', '25120', '25074', '25182', '25131',],
                    contrast_list=config.contrast,
                    slice_offset=0,
                    normalization_method="0-1",
                    is_test=True,
                    aug=False)
    
    test_loader = torch.utils.data.DataLoader(dataset, 
        batch_size=1, shuffle=False)
    # as there are 10 realisations then batch = 10
    config.sampling.batch_size = 1

    ###### SOLVING REVERSE SDE ######
    img_shape = (config.dataset.img_z_dim, 
        config.dataset.img_xy_dim, config.dataset.img_xy_dim)

    save_recon = []
    save_ref = []

    if guided:
        save_guided = []
    print("Normalisation type: ", ml_collection.normalisation)
    print("Length of test loader: ", len(test_loader))

    subj_pred_list = {}
    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            # [0] reference, [1] gt, [2] osem, [3] norm, [4] measurements,
            # [5] contamination_factor, [6] attn_factors
            # FIRST STEP
            data, subj, slc = batch
            subj = subj[0]
            slc = slc[0]
            if subj not in subj_pred_list:
                subj_pred_list[subj] = []
            if slc < 18 or slc >= 70:
                # blank slice
                recon = np.zeros((256, 256))
                subj_pred_list[subj].append(recon)
            else:
                print("====> Currently <====")
                print(subj, slc)
                gt = data[:, [0], ...]
                # print(f"gt shape: {gt.shape}")
                if guided:
                    guided_img = data[:, 1:, ...].to(config.device)
                    # print(f"guided_img shape: {guided_img.shape}")
                nll_partial = None

                logg_kwargs = {'log_dir': "./tb", 
                    'num_img_in_log': config.sampling.batch_size, 'sample_num':idx,
                    'ground_truth': gt, 'osem': None}
                
                sampler = get_standard_sampler(
                    config=config,
                    score=score_model_t1,
                    sde=sde,
                    nll=nll_partial, 
                    im_shape=img_shape,
                    multimodal=True,
                    score_2=score_model_t2f,
                    guidance_imgs=guided_img if guided else None,
                    device=config.device
                )
                
                recon, _ = sampler.sample(logg_kwargs=logg_kwargs, logging=False)
                recon = torch.clamp(recon, min=0, max=1.0)
                save_recon.append(recon.squeeze().cpu())
                subj_pred_list[subj].append(recon.squeeze().cpu())

            if len(subj_pred_list[subj]) == 89:
                # save to nifti
                print(f"Saving nifti file for subj {subj}.....")
                to_nii(subj_pred_list[subj], subj, config.nii_save_path)
        
if __name__ == '__main__':
    reconstruction()