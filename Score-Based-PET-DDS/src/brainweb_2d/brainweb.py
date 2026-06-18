import torch 

import random
import numpy as np
import torch
import h5py
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import v2
import matplotlib.pyplot as plt

class PETMRTest(Dataset):
    def __init__(self, data_h5, subjects, img_size=256, slice_offset=1, contrast_list=['T1_GRE', 'T2_FLAIR'], normalization_method="-1-1", is_test=True, aug=False):
        self.data = h5py.File(data_h5, 'r')
        self.contrast_list = contrast_list
        self.slices = []
        self.subjects = subjects
        with h5py.File(data_h5, "r") as f:
            self.subject_list = list(f.keys())
        for subj in subjects:
            assert subj in self.subject_list, "This subject is not found in h5py... exiting.."
            for i in range(89):
                self.slices.append([subj, i])
        self.image_size = img_size
        self.slice_offset = slice_offset
        self.aug = aug
        self.normalization_method = normalization_method
        self.is_test = is_test


    def __len__(self):
        print(f"Total training slices: {len(self.slices)}")
        return len(self.slices)
    

    def __getitem__(self, idx):
        subj = self.slices[idx][0]
        slice_idx = int(self.slices[idx][1])
        slc_idx_sel = np.clip(np.arange(slice_idx-self.slice_offset, slice_idx+self.slice_offset+1), 0, 88)
        imgs = []
        for contrast in self.contrast_list:
            if 'T1' in contrast:
                if subj+'/'+'T1_GRE' in self.data.keys():
                    mri = np.array(self.data[subj+'/T1_GRE'])[:,:, slc_idx_sel]
                elif subj+'/T1_SE' in self.data.keys():
                    mri = np.array(self.data[subj+'/T1_SE'])[:,:,slc_idx_sel]
                else:
                    print(f"T1 not found for sub {subj}..?????")
                    exit(0)           
                imgs.append(mri)         
                mri = torch.Tensor(np.transpose(mri, (2, 0, 1)))
            elif contrast == 'T2_FLAIR':
                if subj+'/'+'T2_FLAIR' in self.data.keys():
                    mri = np.array(self.data[subj+'/T2_FLAIR'])[:,:, slc_idx_sel]
                elif subj+'/'+'T2_FLAIR_2D' in self.data.keys():
                    mri = np.array(self.data[subj+'/T2_FLAIR_2D'])[:,:, slc_idx_sel]
                else:
                    print(f"T2 FlAIR not found for subject {subj}")
                    exit(0)
                imgs.append(mri)
                mri = torch.Tensor(np.transpose(mri, (2, 0, 1)))
            elif contrast == 'PET_1p':
                if subj+'/'+'PET_1p' in self.data.keys():
                    mri = np.array(self.data[subj+'/PET_1p'])[:,:, slc_idx_sel]
                else:
                    print(f"PET_1p not found for subject {subj}")
                    exit(0)
                imgs.append(mri)
                mri = torch.Tensor(np.transpose(mri, (2, 0, 1)))

        if len(imgs) > 1:
            img = np.concatenate(imgs, axis=-1)
            mri = np.transpose(img, (2, 0, 1))
            mri = torch.Tensor(mri).float()
            
        if subj+'/PET' in self.data.keys():
            reference = self.data[subj+'/PET'][:,:,slice_idx:slice_idx+1]
        elif subj+'/PET_MAC' in self.data.keys():
            reference = self.data[subj+'/PET_MAC'][:,:,slice_idx:slice_idx+1]
        elif subj+'/PET_QCLEAR' in self.data.keys():
            reference = self.data[subj+'/PET_QCLEAR'][:,:,slice_idx:slice_idx+1]
        elif subj+'/PET_TOF' in self.data.keys():
            reference = self.data[subj+'/PET_TOF'][:,:,slice_idx:slice_idx+1]
        else:
            print("Ops, PET not found??? Exiting..")
            exit(0)
        reference = torch.Tensor(np.transpose(reference, (2, 0, 1))).float()
        res = torch.cat((reference, mri), dim=0) # 2, 256, 256
        return res, subj, slice_idx





class PETMR(Dataset):
    def __init__(self, data_h5, slice_file, save_root=None, img_size=(160, 192), slice_offset=0, contrast_list=['T2_FLAIR'], normalization_method="0-1", use_low_dose=False, is_test=False, aug=False):
        self.data = h5py.File(data_h5, 'r')
        self.contrast_list = contrast_list
        self.use_low_dose = use_low_dose
        self.slices = []
        self.image_size = img_size
        self.slice_offset = slice_offset
        self.aug = aug
        self.save_transform = 0
        self.save_root = save_root
        with open(slice_file, 'r') as fr:
            for line in fr:
                res = line.strip('\n').split(" ")
                self.slices.append(res)
        self.normalization_method = normalization_method
        self.is_test = is_test
        with h5py.File(data_h5, "r") as f:
            self.subject_list = list(f.keys())
        self.fill = -1 if self.normalization_method == "-1-1" else 0
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
        return torch.Tensor(res_input).float(), torch.Tensor(res_target).float()

    def __getitem__(self, idx):
        subj = self.slices[idx][0]
        slice_idx = int(self.slices[idx][1])
        output  = {}
        imgs = []
        for contrast in self.contrast_list:
            if 'T1' in contrast:
                if subj+'/'+'T1_GRE' in self.data.keys():
                    mri = self.data[subj+'/T1_GRE'][:,:,slice_idx-self.slice_offset:slice_idx+self.slice_offset+1]
                elif subj+'/T1_SE' in self.data.keys():
                    mri = self.data[subj+'/T1_SE'][:,:,slice_idx-self.slice_offset:slice_idx+self.slice_offset+1]
                else:
                    print(f"T1 not found for sub {subj}..?????")
                    exit(0)           
                imgs.append(mri)         
                mri = torch.Tensor(np.transpose(mri, (2, 0, 1)))
            elif contrast == 'T2_FLAIR':
                if subj+'/'+'T2_FLAIR' in self.data.keys():
                    mri = self.data[subj+'/T2_FLAIR'][:,:,slice_idx-self.slice_offset:slice_idx+self.slice_offset+1]
                elif subj+'/'+'T2_FLAIR_2D' in self.data.keys():
                    mri = self.data[subj+'/T2_FLAIR_2D'][:,:,slice_idx-self.slice_offset:slice_idx+self.slice_offset+1]
                else:
                    print(f"T2 FlAIR not found for subject {subj}")
                    exit(0)
                imgs.append(mri)
                mri = torch.Tensor(np.transpose(mri, (2, 0, 1)))
            elif contrast == 'PET_1p':
                if subj+'/'+'PET_1p' in self.data.keys():
                    mri = self.data[subj+'/PET_1p'][:,:,slice_idx-self.slice_offset:slice_idx+self.slice_offset+1]
                else:
                    print(f"PET_1p not found for subject {subj}")
                    exit(0)
                imgs.append(mri)
                mri = torch.Tensor(np.transpose(mri, (2, 0, 1)))
            elif contrast == 'PET_10p':
                if subj+'/'+'PET_10p' in self.data.keys():
                    mri = self.data[subj+'/PET_10p'][:,:,slice_idx-self.slice_offset:slice_idx+self.slice_offset+1]
                else:
                    print(f"PET_10p not found for subject {subj}")
                    exit(0)
                imgs.append(mri)
                mri = torch.Tensor(np.transpose(mri, (2, 0, 1)))
        if len(imgs) > 1:
            img = np.concatenate(imgs, axis=-1)
            mri = np.transpose(img, (2, 0, 1))
            mri = torch.Tensor(mri).float()
        if subj+'/PET' in self.data.keys():
            reference = self.data[subj+'/PET'][:,:,slice_idx:slice_idx+1]
        elif subj+'/PET_MAC' in self.data.keys():
            reference = self.data[subj+'/PET_MAC'][:,:,slice_idx:slice_idx+1]
        elif subj+'/PET_QCLEAR' in self.data.keys():
            reference = self.data[subj+'/PET_QCLEAR'][:,:,slice_idx:slice_idx+1]
        elif subj+'/PET_TOF' in self.data.keys():
            reference = self.data[subj+'/PET_TOF'][:,:,slice_idx:slice_idx+1]
        else:
            print("Ops, PET not found??? Exiting..")
            exit(0)
        reference = torch.Tensor(np.transpose(reference, (2, 0, 1))).float()
        if self.aug:
            mri, reference = self.augment(mri, reference, subj, slice_idx)
        res = torch.cat((reference, mri), dim=0) # 2, 256, 256
        if self.is_test:
            return res, subj, slice_idx
        else:
            return res



class BrainWebClean(torch.utils.data.Dataset):
    def __init__(self, path_to_files="path_to/test_dict.pt", mri=False):

        self.path_to_files = path_to_files
        self.data = torch.load(path_to_files, map_location=torch.device('cpu'))
        self.mri = mri
    
    def __len__(self):
        return self.data['clean_measurements'].shape[0]

    def __getitem__(self, idx):
        y = self.data["clean_measurements"][idx, ...]
        mu = self.data["mu"][idx, ...]
        gt = self.data["reference"][idx, ...]
        if self.mri:
            mri = self.data["mri"][idx, ...]
            return y, mu, gt, mri
        return y, mu, gt

class BrainWebOSEM(torch.utils.data.Dataset):
    def __init__(self, part, noise_level, base_path="path_to/src/brainweb_2d/", static_path = None, device="cpu", guided=False):
        assert noise_level in [2.5, 5, 7.5, 10, 50, 100, "2.5", "5", "7.5", "10", "50", "100"], "noise level has to be 2.5, 5, 7.5, 10, 50, 100"
        assert part in ["train", "test", "test_tumour", "subset_test_tumour", "subset_test", "validation"], 'part has to be "train", "test", "test_tumour", "subset_test_tumour", "subset_test", "validation"'

        self.part = part 
        self.noise_level = noise_level
        self.guided = guided

        self.base_path = base_path
        # dict_keys(['osem', 'scale_factor', 'measurements', 'contamination_factor', 'attn_factors'])
        self.noisy = torch.load(base_path+"noisy/noisy_"+ self.part + "_" + str(noise_level)+".pt", map_location=torch.device(device))
        # dict_keys(['clean_measurements', 'mu', 'reference']) 
        self.clean = torch.load(base_path+"clean/clean_"+part+".pt", map_location=torch.device(device))
        if static_path is not None:
            # dict_keys(['osem', 'scale_factor', 'measurements', 'contamination_factor', 'attn_factors'])
            self.noisy = torch.load(static_path, map_location=torch.device(device))
            # dict_keys(['clean_measurements', 'mu', 'reference']) 
            self.clean = torch.load(base_path+"clean/"+part+"_clean.pt", map_location=torch.device(device))
        if "tumour" in part:
            self.tumour = True
        else:
            self.tumour = False
    def __len__(self):
        return self.clean["reference"].shape[0]

    def __getitem__(self, idx):
        
        reference = self.clean["reference"][idx, ...].float()
        scale_factor = self.noisy["scale_factor"][idx]

        reference = reference*scale_factor[[0]] 

        if self.guided:
            reference = torch.cat((reference, self.clean["mri"][idx, ...].float()), dim=0)
        osem = self.noisy["osem"][idx, ...].float()


        norm = 1

        measurements = self.noisy["measurements"][idx, ...].float()
        contamination_factor = self.noisy["contamination_factor"][idx]
        attn_factors = self.noisy["attn_factors"][idx, ...].float()
        
        if self.part == "subset_test":
            measurements = measurements
            contamination_factor = contamination_factor
            attn_factors = attn_factors
            osem = osem

        if norm == 0:
            norm = torch.ones_like(norm)
            osem = torch.zeros_like(osem)
            reference = torch.zeros_like(reference)
            attn_factors = torch.ones_like(attn_factors)

        if self.tumour:
            background = self.clean["background"][idx, ...].float()
            tumour_rois = self.clean["tumour_rois"][idx, ...].float()
            return reference, scale_factor, osem, norm, measurements, contamination_factor, attn_factors, background, tumour_rois
        return reference, scale_factor, osem, norm, measurements, contamination_factor, attn_factors


class BrainWebSupervisedTrain(torch.utils.data.Dataset):
    def __init__(self, noise_level, base_path="path_to/pyparallelproj/examples/data/", device="cpu", guided=False):
        assert noise_level in [5, 10, 50, "5", "10", "50"], "noise level has to be 5, 10, 50"
        self.base_path = base_path

        # dict_keys(['clean_measurements', 'mu', 'reference', 'mri]) 
        clean = torch.load(base_path+"clean/clean_train.pt", map_location=torch.device(device))
        # dict_keys(['osem', 'scale_factor', 'measurements', 'contamination_factor', 'attn_factors'])
        self.noisy = torch.load(base_path+"noisy/noisy_train_"+str(noise_level)+".pt", map_location=torch.device(device))
        self.reference = clean["reference"]
        self.guided = guided
        if self.guided:
            self.mri = clean["mri"]

    def __len__(self):
        return self.reference

    def __getitem__(self, idx):
        reference = self.reference[idx,...].float()*self.noisy["scale_factor"][idx]

        osem = self.noisy["osem"][idx, ...].float()

        measurements = self.noisy["measurements"][idx, ...].float()

        contamination_factor = self.noisy["contamination_factor"][idx]

        attn_factors = self.noisy["attn_factors"][idx, ...].float()
        
        if self.guided:
            mri = self.mri[idx,...].float()
            return reference, mri, osem, measurements, contamination_factor, attn_factors

        return reference, osem, measurements, contamination_factor, attn_factors



class BrainWebScoreTrain(torch.utils.data.Dataset):
    def __init__(self, base_path="path_to/pyparallelproj/examples/data/", device="cpu", guided=False, normalisation="data_scale"):

        self.base_path = base_path
        # dict_keys(['clean_measurements', 'mu', 'reference', 'mri]) 
        clean = torch.load(base_path+"clean/clean_train.pt", map_location=torch.device(device))
        print(clean.keys())
        self.reference = clean["reference"]
        self.clean_measurements = clean["clean_measurements"]
        self.mri = clean["mri"]
        self.guided = guided

        self.normalisation = normalisation

    def __len__(self):
        return self.reference.shape[0]

    def __getitem__(self, idx):
        reference = self.reference[idx, ...].float()

        if self.normalisation == "data_scale":
            emission_volume = torch.where(reference > 0)[0].shape[0] * 8 # 2 x 2 x 2
            current_trues_per_volume = float(self.clean_measurements[idx].sum() / emission_volume)
        elif self.normalisation == "image_scale":
            emission_volume = torch.where(reference > 0)[0].shape[0]
            current_trues_per_volume = float(reference.sum() / emission_volume)
        else:
            raise NotImplementedError

        reference = reference/current_trues_per_volume 

        reference = reference* (0.5 + torch.rand(1))

        mri = self.mri[idx, ...].float()
        if self.guided:
            return torch.cat((reference, mri), dim=0)
        return reference

    
if __name__ == "__main__":

    dataset = BrainWebScoreTrain(base_path="path_to/", normalisation="image_scale")
    import matplotlib.pyplot as plt 
    import numpy as np 

    for i in range(10):
        batch = dataset[i]
        print(batch.min(), batch.max())