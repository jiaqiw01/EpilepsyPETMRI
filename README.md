# EpilepsyPETMRI
PET MRI translation for epilepsy population

# TransUnet
- The default code loads volumes from .h5 files and uses keys such as 'T1', 'PET_QCLEAR' to index. It is best to only use the code from model.py/model_simple.py and use your own training pipeline

# SGM-KD
- Our results used the cosine-interpolation sampling method (the default one), with sigma_min = 0.01, sigma_max = 10, sigma_data = 0.5
- The network is not too deep in our experiment due to memory constrain
- More details located in the README file in SGM-KD-Simplified

# SGM-VP
- In general slower to train than SGM-KD
- The original method used Poisson loss backpropagation for data consistency, but this is removed in our experiment as our method is purely image-based
- More details located in the README file in Score-Based-PET-DDS