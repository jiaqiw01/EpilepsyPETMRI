# Train
1. Use your own dataloader and add it in train_all.py line 212 (marked as TODO)
2. Create a training config in ./configs/your_config_name.yaml
    - input_channel is for PET
    - unet_cond_channel is for all MRI slices
    - replace "dataset" fields with your own
3. Replace some of the argparse stuff with your own
    - block_size
    - output_slice
    - contrast
4. Experiment on some hyperparameters such as 
    - sigma_min/sigma_max: these control the amount of noise added, if we add excessive noise the task may be unnecessarily hard
    - depth/channels/attention: these will demand more memory 
3. CUDA_VISIBLE_DEVICES=2 python train_all.py \
    --config ./configs/your_config \
    --name t1_t2f_zerodose_all_v1_3ch_3class_2neighbor_large (your exp name) \
    --contrast T1 T2_FLAIR  \
    --block_size 1 \
    --batch-size 32 \
    --output_slice 3 \

# Sampling
1. Modify sampling.py by replacing the test_dataset with your own
CUDA_VISIBLE_DEVICES=3 python sample.py \
    --config ./configs/config_petmr_t1_t2f_zerodose_3ch_class_emb_v1.json (use the same training config) \
    --checkpoint /data/jiaqiw01/PET_MRI/experiments/k-diffusion/results/T1_1p/ckpt_00060000.pth \
    --block_size 1 \
    --contrast T1 T2_FLAIR \
    --nii_save_dest /data/jiaqiw01/debug_tests/kdiff/kdiff_t1_t2f 

# Notes
- Train for longer to alleviate slice inconsistency issue
- sampling.py is 3slice-3slice prediction, there is no averaging operations done



