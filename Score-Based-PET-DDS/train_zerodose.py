import os
import numpy as np 
from datetime import datetime 
import yaml 
from torch.utils.data import DataLoader
import torch
from src import (get_standard_sde, get_standard_score, BrainWebScoreTrain, PETMR,
			score_model_simple_trainer)



#from configs.ellipses_configs import get_config
from configs.default_config import get_default_configs

def coordinator():

	config = get_default_configs()
	config.contrast = ['T1']
	config.slice_offset = 1
	config.model.in_channels = 4 # (2 * config.slice_offset + 1) * len(config.contrast) + len(target_slice) (1 in our case)
	config.label = '_t1_2_pet_zerodose' # tailor this specifically for one experiment

	if config.guided_p_uncond is not None:
		print("Train Guided Score Model")

	sde = get_standard_sde(config)

	score_model = get_standard_score(config, sde, use_ema=True, load_path=None, load_model=False)
	# TODO: change here to load data
	brain_dataset = PETMR(data_h5="xx", 
					  	 slice_file="xx",
						 contrast_list=config.contrast,
						 slice_offset=config.slice_offset,
						 normalization_method="0-1",
						 save_root=None,
						 aug=True)
	
	brain_dataset_val = PETMR(data_h5="xx", 
					  	 slice_file="xx",
						 contrast_list=config.contrast,
						 slice_offset=config.slice_offset,
						 normalization_method="0-1",
						 save_root=None,
						 aug=False)
	
	train_dl = DataLoader(brain_dataset,batch_size=8, num_workers=2)

	val_dl = DataLoader(brain_dataset_val,batch_size=4, num_workers=2)

	print(f" # Parameters: {sum([p.numel() for p in score_model.parameters()]) }")
	today = datetime.now()
	
	if config.guided_p_uncond is not None: # TODO: change here too
		log_dir = './guided_p_uncond/' + config.sde.type + config.label + "_beta_max_" + str(config.sde.beta_max)

	else:
		log_dir = './no_guided/' + config.sde.type

	if not os.path.exists(log_dir):
		os.makedirs(log_dir)

	found_version = False 
	version_num = 0
	while not found_version:
		if os.path.isdir(os.path.join(log_dir, "version_" + str(version_num))):
			version_num += 1
		else:
			found_version = True 

	log_dir = os.path.join(log_dir, "version_" + str(version_num))
	os.makedirs(log_dir)
	print("log dir is: ", log_dir)

	with open(os.path.join(log_dir,'report.yaml'), 'w') as file:
		yaml.dump(config, file)

	score_model_simple_trainer(
		score=score_model.to(config.device),
		sde=sde, 
		train_dl=train_dl,
		val_dl=val_dl,
		optim_kwargs={
			'epochs': config.training.epochs,
			'lr': config.training.lr,
			'ema_warm_start_steps': config.training.ema_warm_start_steps,
			'log_freq': config.training.log_freq,
			'ema_decay': config.training.ema_decay
			},
			val_kwargs={
			'batch_size': config.validation.batch_size,
			'num_steps': config.validation.num_steps,
			'snr': config.validation.snr,
			'eps': config.validation.eps,
			'sample_freq' : config.validation.sample_freq
			},
		device=config.device,
		log_dir=log_dir,
		guided_p_uncond=config.guided_p_uncond,
		num_target_slices=config.num_target_slices
	)


if __name__ == '__main__': 
	coordinator()