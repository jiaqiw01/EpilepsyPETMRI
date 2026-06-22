# EpilepsyPETMRI
PET MRI translation for epilepsy population.

Code repo for paper "Score-based generative diffusion models to synthesize full-dose FDG brain PET from MRI in epilepsy patients"

It may be easier for you to go to the original repo for SGM models and adapt them to your needs :)

# Data Preparation
- Our volumes **do not** use template. We co-register all MRI volumes to PET volume with shape 89, 256, 256. The top 20 and last 20 slices are almost empty and noisy.
- Please have your PET volumes in 1.17mm, 1.17mm, 2.78mm and coregister T1/T2 to PET
- Save your PET/MRI volumes into a big H5 file for faster data loading, don't forget to normalize to 0-1 or -1-1 range:
```text
  dataset.h5
  ├── subject_001
  │   ├── t1
  │   │   └── data
  │   ├── t2
  │   │   └── data
  │   └── pet
  │       └── data
  ├── subject_002
  │   ├── t1
  │   │   └── data
  │   ├── t2
  │   │   └── data
  │   └── pet
  │       └── data
  └── ...
```
- Run train_test_split.py such as `python train_test_split.py --src /data/PET_MRI/data/all_cases_rawspace_complete_max.h5 --dest ./train_test_split --contrast T1 PET --restart`
- The train/val slice files will be generated

# TransUnet
- The default code loads volumes from .h5 files and uses keys such as 'T1', 'PET_QCLEAR' to index. It is best to only use the code from model.py/model_simple.py and use your own training pipeline

# SGM-KD
- Original repo: https://github.com/crowsonkb/k-diffusion
- Our results used the cosine-interpolation sampling method (the default one), with sigma_min = 0.01, sigma_max = 10, sigma_data = 0.5
- The network is not too deep in our experiment due to memory constrain
- More details located in the README file in SGM-KD-Simplified

# SGM-VP
- Original repo: https://github.com/Imraj-Singh/Score-Based-Generative-Models-for-PET-Image-Reconstruction
- In general slower to train than SGM-KD
- The original method used Poisson loss backpropagation for data consistency, but this is removed in our experiment as our method is purely image-based
- More details located in the README file in Score-Based-PET-DDS

# Evaluation
- Requires segmentation file and color maps
- Evaluates asymmetry scores, congruency scores, SUVR errors
  
