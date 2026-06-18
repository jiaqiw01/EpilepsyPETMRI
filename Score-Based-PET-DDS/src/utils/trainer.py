"""
Adapted from: https://github.com/educating-dip/score_based_model_baselines/blob/main/src/utils/trainer.py

"""


from typing import Optional, Any, Dict
import os 
import torch 
import torchvision
import numpy as np 
import functools 

from tqdm import tqdm 
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from torch.utils.data import DataLoader
from .losses import loss_fn
from .ema import ExponentialMovingAverage
from .sde import SDE

from ..third_party_models import OpenAiUNetModel
from ..samplers import BaseSampler, Euler_Maruyama_sde_predictor, Langevin_sde_corrector, soft_diffusion_momentum_sde_predictor


def score_model_simple_trainer(
	score: OpenAiUNetModel,
	sde: SDE, 
	train_dl: DataLoader, 
	val_dl: DataLoader,
	optim_kwargs: Dict,
	val_kwargs: Dict,
	device: Optional[Any] = None, 
	log_dir: str ='./',
	guided_p_uncond: Optional[Any] = None,
	num_target_slices: int = 1
	) -> None:

	writer = SummaryWriter(log_dir=log_dir, comment='training-score-model')
	optimizer = Adam(score.parameters(), lr=optim_kwargs['lr'])
	for epoch in range(optim_kwargs['epochs']):
		avg_loss, num_items = 0, 0
		score.train()
		for _ in range(3):
			for idx, batch in tqdm(enumerate(train_dl), total = len(train_dl)):
				x = batch.to(device)
				loss = loss_fn(score, x, sde, num_target_slices=num_target_slices)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				avg_loss += loss.item() * x.shape[0]
				num_items += x.shape[0]
				if idx % optim_kwargs['log_freq'] == 0:
					writer.add_scalar('train/loss', loss.item(), epoch*len(train_dl) + idx) 
				if epoch == 0 and idx == optim_kwargs['ema_warm_start_steps']:
					ema = ExponentialMovingAverage(score.parameters(), decay=optim_kwargs['ema_decay'])
				if idx > optim_kwargs['ema_warm_start_steps'] or epoch > 0:
					ema.update(score.parameters())

		print('Average Loss: {:5f}'.format(avg_loss / num_items))
		writer.add_scalar('train/mean_loss_per_epoch', avg_loss / num_items, epoch + 1)
		# save every 10 epochs
		if (epoch + 1) % 10 == 0:
			torch.save(score.state_dict(), os.path.join(log_dir,f'model_{epoch+1}.pt'))
			torch.save(ema.state_dict(), os.path.join(log_dir, f'ema_model_{epoch+1}.pt'))
			print(f"Model ckpt saved at {os.path.join(log_dir,f'model_{epoch+1}.pt')}")

		if val_kwargs['sample_freq'] > 0:
			if epoch % val_kwargs['sample_freq']== 0:
				print("Inside Validation.....")
				score.eval()

				predictor = functools.partial(Euler_Maruyama_sde_predictor, nloglik = None)
				corrector = functools.partial(Langevin_sde_corrector, nloglik = None) 
				for idx, batch in enumerate(val_dl):
					if (idx + 1) % 10 == 0:
						x = batch.to(device)
						sample_kwargs={
								'num_steps': val_kwargs['num_steps'],
								'start_time_step': 0,
								'batch_size': val_kwargs['batch_size'] if guided_p_uncond is None else x.shape[0],
								'im_shape': [1, *x.shape[2:]],
								'eps': val_kwargs['eps'],
								'predictor': {'aTweedy': False},
								'corrector': {'corrector_steps': 1}
								}
				
						log_kwargs = {
							'log_dir': log_dir,
							'sample_num': 1,
							'ground_truth': x[:, 0,...].unsqueeze(1),
							'num_img_in_log': 20,
							'osem': None
						}

						if guided_p_uncond is not None:
							sample_kwargs['predictor'] = {
								"guidance_imgs": x[:,1:,...],
								"guidance_strength": 1
							}
							sample_kwargs['corrector'] = {
								"guidance_imgs": x[:,1:,...],
								"guidance_strength": 1
							}

						sampler = BaseSampler(
							score=score,
							sde=sde,
							predictor=predictor,
							corrector=corrector,
							init_chain_fn=None,
							sample_kwargs=sample_kwargs,
							device=device)
						x_mean, x_init = sampler.sample(logg_kwargs=log_kwargs, logging=False)
						print("Finished sampling x_mean in validation!")
						print(x_mean.shape)

						if guided_p_uncond is not None: # note that by default x[:, 0] is our target slice, x[:, 1:] is our conditional inputs (MRI)
							x_mean = torch.cat([x_mean[:,[0],...], x[:, [1],...]], dim=0)
							sample_grid = torchvision.utils.make_grid(x_mean, normalize=True, scale_each=True, nrow = x.shape[0])
							writer.add_image('unconditional samples in validation set', sample_grid, global_step=epoch)
						else:
							sample_grid = torchvision.utils.make_grid(x_mean, normalize=True, scale_each=True)
							writer.add_image('unconditional samples in validation set', sample_grid, global_step=epoch)
						break