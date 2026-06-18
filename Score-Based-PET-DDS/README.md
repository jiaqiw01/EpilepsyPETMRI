## Configs you need to modify
- dataset/PETMR.yaml sets image dimension
- score_based_model/vpsde_image_scale_guided_t1_tp.yaml sets the checkpoint to use
- final_reconstruction_petmr.yaml sets the contrast to use and model checkpoint

## Training
- Modify train_zerodose/train_lowdose.py for dataloader (marked as TODO)
- Modify training configs in configs/default_config.py (needs some hardcode)
- python train_lowdose.py 

## Sampling
- Modify coordinators/final_recon_no_nll.py "@hydra.main(config_path='../configs', config_name='final_reconstruction_petmr')", replace with your config name

