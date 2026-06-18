# new version, 2020/08/19

import os
import glob
import time
import torch
import torch.optim as optim
import numpy as np
import yaml
import pdb
import psutil
import argparse
import logging
from model import *
from util import *
from model_simple import *

# set seed
seed = 10
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic=True


parser = argparse.ArgumentParser(prog='ProgramName', description='What the program does')
parser.add_argument('-c', '--config')
parser.add_argument('-g', '--gpu', type=int, default=0)
args = parser.parse_args()
yml = args.config
print(f"Using {yml} as model yaml")

_, config = load_config_yaml(yml)
config['in_num_ch'] = len(config['contrast_list']) * (2*config['block_size']+1)
config['device'] = torch.device(f'cuda:{args.gpu}')

if config['ckpt_timelabel'] and (config['phase'] == 'test' or config['continue_train'] == True):
    time_label = config['ckpt_timelabel']
else:
    localtime = time.localtime(time.time())
    time_label = str(localtime.tm_year) + '_' + str(localtime.tm_mon) + '_' + str(localtime.tm_mday) + '_' + str(localtime.tm_hour) + '_' + str(localtime.tm_min) + '_' + str(config['model_label'])


# ckpt folder, load yaml config
# config['ckpt_path'] = os.path.join('../ckpt/', 'BraTS', config['model_name'], time_label)
config['ckpt_path'] = os.path.join('./ckpt/', config['dataset_name'], config['model_name'], time_label)
print(config['ckpt_path'])

if not os.path.exists(config['ckpt_path']):     # test, not exists
    os.makedirs(config['ckpt_path'])
    save_config_yaml(config['ckpt_path'], config)
elif config['load_yaml']:       # exist and use yaml config
    flag, config_load = load_config_yaml(os.path.join(config['ckpt_path'], 'config.yaml'))
    if flag:    # load yaml success
        print('load yaml config file')
        for key in config_load.keys():  # if yaml has, use yaml's param, else use config
            if key == 'phase':
                continue
            if key in config.keys():
                config[key] = config_load[key]
            else:
                print('current config do not have yaml param')
        config['in_num_ch'] = len(config['contrast_list']) * (2*config['block_size']+1)
    else:
        save_config_yaml(config['ckpt_path'], config)

logging.basicConfig(level=logging.DEBUG, filename=os.path.join(config['ckpt_path'], "logfile"), filemode='a+', format="%(asctime)-15s %(levelname)-8s %(message)s")
logging.info(config['model_name'])
# config['ckpt_name'] = 'model_best.pth.tar'


# if config['phase'] == 'train':
#     aug = True
# else:
#     aug = False
aug = True

transforms = v2.Compose([
    v2.RandomHorizontalFlip(p=0.5),
    # v2.RandomResizedCrop(size=(256, 256), scale=(0.7,1)),
    v2.RandomRotation(degrees=(-10, 10))
])

Data = ZeroDoseDataAll(config['dataset_name'], config['data_path'], config['data_h5_path'], config['train_txt_path'], config['val_txt_path'], config['test_txt_path'],
                        postfix=config['postfix'], norm_type=config['norm_type'], batch_size=config['batch_size'], num_fold=config['num_fold'], \
                        fold=config['fold'], shuffle=config['shuffle'], num_workers=0, block_size=config['block_size'], \
                        contrast_list=config['contrast_list'], aug=aug, dropoff=config['dropoff'], skull_strip=config['skull_strip'], \
                        pred_low=config['pred_low'], transform=transforms, save_root=config['ckpt_path'])

trainDataLoader = Data.trainLoader
valDataLoader = Data.valLoader
testDataLoader = Data.testLoader



# define model
if config['norm_type'] == 'z-score':
    config['target_output_act'] = 'no'
else:
    config['target_output_act'] = 'softplus'
    # config['target_output_act'] = 'no'

if config['target_model_name'] == 'U':
    model = GANShortGenerator(in_num_ch=config['in_num_ch'], out_num_ch=config['out_num_ch'], first_num_ch=64, input_size=(config['input_height'], config['input_width']), output_activation=config['target_output_act']).to(config['device'])
elif config['target_model_name'] == 'U+SA':
    model = GANShortGeneratorWithSpatialAttention(in_num_ch=config['in_num_ch'], out_num_ch=config['out_num_ch'], first_num_ch=64, input_size=(config['input_height'], config['input_width']), output_activation=config['target_output_act']).to(config['device'])
elif config['target_model_name'] == 'U+SA+CA':
    model = GANShortGeneratorWithChannelAttentionAllAndSpatialAttention(in_num_ch=config['in_num_ch'], out_num_ch=config['out_num_ch'], first_num_ch=64, input_size=(config['input_height'], config['input_width']), output_activation=config['target_output_act']).to(config['device'])
elif config['target_model_name'] == 'U+SSA+CA':
    model = GANShortGeneratorWithChannelAttentionAllAndSymmetrySpatialAttention(in_num_ch=config['in_num_ch'], out_num_ch=config['out_num_ch'], 
                                                                                first_num_ch=64, input_size=(config['input_height'], config['input_width']), 
                                                                                output_activation=config['target_output_act'], is_transformer=config['is_transformer']).to(config['device'])
elif config['target_model_name'] == 'TransUNET':
    model = TransUNet(in_num_ch=config['in_num_ch'], out_num_ch=config['out_num_ch'], first_num_ch=64, input_size=(config['input_height'], config['input_width']), 
                      is_symmetry=config['is_symmetry'], output_activation=config['target_output_act'], is_transformer=config['is_transformer']).to(config['device'])
    print("Using Trans UNET as model!")
elif config['target_model_name'] == 'GANSplit': # this uses symmetry gated, not suitable for rawspace
    print(f"Input channel number: {config['in_num_ch']}")
    model = GANShortGeneratorWithSplitInputChannelAttentionAllAndSpatialAttention(in_num_ch=config['in_num_ch'], out_num_ch=config['out_num_ch'],).to(config['device'])
    print("Using GAN Split Model....")
else:
    raise ValueError('Not implemented')

class PolyLR(torch.optim.lr_scheduler._LRScheduler):
    """Set the learning rate of each parameter group to the initial lr decayed
    by gamma every epoch. When last_epoch=-1, sets initial lr as lr.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        gamma (float): Multiplicative factor of learning rate decay.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(self, optimizer, max_epoch, power=0.9, last_epoch=-1):
        self.max_epoch = max_epoch
        self.power = power
        super(PolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * (1 - self.last_epoch / self.max_epoch) ** self.power
                for base_lr in self.base_lrs]

# define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=1e-5, amsgrad=True)
# scheduler = PolyLR(optimizer, max_epoch=300, power=0.9)
scheduler = PolyLR(optimizer, max_epoch=50, power=0.9)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, min_lr=1e-5)

if config['model_name'] == 'GAN':
    discriminator = Discriminator(in_num_ch=config['in_num_ch']+config['out_num_ch'], inter_num_ch=64, input_shape=(config['input_height'], config['input_width']), is_patch_gan=True).to(config['device'])
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=config['lr'], betas=(0.5, 0.999))

# load pretrained model
if config['continue_train'] or config['phase'] == 'test':
    [optimizer, scheduler, model], start_epoch = load_checkpoint_by_key([optimizer, scheduler, model], config['ckpt_path'], ['optimizer', 'scheduler', 'model'], config['device'], config['ckpt_name'])
    # [optimizer, model], start_epoch = load_checkpoint_by_key([optimizer, model], config['ckpt_path'], ['optimizer', 'model'], config['device'], config['ckpt_name'])
    # [model], start_epoch = load_checkpoint_by_key([model], config['ckpt_path'], ['model'], config['device'], config['ckpt_name'])
else:
    start_epoch = -1

if config['phase'] == 'train':
    save_config_file(config)

# train
def train():
    global_iter = 0
    monitor_metric_best = 100
    start_time = time.time()

    # stat = evaluate(phase='val', set='val', save_res=False)
    # print(stat)
    for epoch in range(start_epoch+1, config['epochs']):
        print(f"-----Epoch {epoch}-----")
        logging.info("===> Epoch {epoch}")
        model.train()
        loss_all_dict = {'recon_y': 0., 'all': 0.}
        global_iter0 = global_iter
        for _ in range(2):
            for iter, sample in enumerate(trainDataLoader, 0):
                global_iter += 1

                inputs = sample['inputs'].to(config['device'], dtype=torch.float)
                targets = sample['targets'].to(config['device'], dtype=torch.float)
                mask = sample['mask'].to(config['device'], dtype=torch.float)
                pred, _ = model(inputs)

                if 'GAN' in config['model_name']:
                    output_d_real = discriminator(torch.cat([inputs, targets], dim=1))
                    output_d_fake = discriminator(torch.cat([inputs, pred.detach()], dim=1))
                    loss_d = 0.5 * (F.binary_cross_entropy_with_logits(output_d_real, torch.ones_like(output_d_real)) + \
                            F.binary_cross_entropy_with_logits(output_d_fake, torch.zeros_like(output_d_fake)))
                    optimizer_D.zero_grad()
                    loss_d.backward()
                    optimizer_D.step()
                    

                if config['p'] == 1:
                    loss_recon_y = torch.abs(pred - targets).mean()
                else:
                    loss_recon_y = torch.pow(pred - targets, 2).mean()
                loss = config['lambda_recon_y'] * loss_recon_y

                if config['lambda_gan'] > 0 and config['model_name'] == 'GAN':
                    output_d_fake = discriminator(torch.cat([inputs, pred], dim=1))
                    loss_gan = F.binary_cross_entropy_with_logits(output_d_fake.detach(), torch.ones_like(output_d_fake))
                    loss += config['lambda_gan'] * loss_gan
                    logging.info("Recon loss: %.2f, Gan Loss: %.2f, Dis Loss: %.2f", loss_recon_y, loss_gan, loss_d)

                loss_all_dict['recon_y'] += loss_recon_y.item()
                loss_all_dict['all'] += loss.item()
                # check nan
                if np.isnan(loss_all_dict['all']) or np.isnan(loss_all_dict['recon_y']) or np.isinf(loss_all_dict['all']):
                    logging.info("Got a nan value in loss....")
                    logging.info("Exiting...")
                    exit(0)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                for name, param in model.named_parameters():
                    try:
                        if not torch.isfinite(param.grad).all():
                            pdb.set_trace()
                    except:
                        continue

                optimizer.step()
                

                if global_iter % 10 == 0:
                    logging.info('Epoch[%3d], iter[%3d]: loss=[%.4f], recon y=[%.4f]' \
                            % (epoch, iter, loss.item(), loss_recon_y.item()))

        # save train result
        num_iter = global_iter - global_iter0
        for key in loss_all_dict.keys():
            loss_all_dict[key] /= num_iter
        save_result_stat(loss_all_dict, config, info='epoch[%2d]'%(epoch))
        logging.info("Loss all dict: %s", loss_all_dict)

        # validation
        # pdb.set_trace()
        stat = evaluate(phase='val', set='val', save_res=False)
        monitor_metric = stat['recon_y']
        # scheduler.step(monitor_metric)
        scheduler.step()
        save_result_stat(stat, config, info='val')
        logging.info("Stat: ")
        logging.info(stat)

        # save ckp
        is_best = False
        if monitor_metric <= monitor_metric_best:
            is_best = True
            monitor_metric_best = monitor_metric if is_best == True else monitor_metric_best
        state = {'epoch': epoch, 'monitor_metric': monitor_metric, 'stat': stat, \
                'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), \
                'model': model.state_dict()}
        save_checkpoint(state, is_best, config['ckpt_path'])

def evaluate(phase='val', set='val', save_res=True, info=''):
    model.eval()
    if phase == 'val':
        loader = valDataLoader
    else:
        if set == 'train':
            loader = trainDataLoader
        elif set == 'val':
            loader = valDataLoader
        elif set == 'test':
            loader = testDataLoader
        else:
            raise ValueError('Undefined loader')

    loss_all_dict = {'recon_y': 0., 'all': 0.}

    subj_id_list = []
    slice_idx_list = []
    input_list = []
    target_list = []
    mask_list = []
    y_fake_fused_list = []
    # att_map_list2 = []
    # att_map_list3 = []
    metrics_list_dict = {}

    with torch.no_grad():
        for iter, sample in enumerate(loader, 0):
            print("Validation Dataloader....")
            subj_id = sample['subj_id']
            inputs = sample['inputs'].to(config['device'], dtype=torch.float)
            targets = sample['targets'].to(config['device'], dtype=torch.float)
            mask = sample['mask'].to(config['device'], dtype=torch.float)
            slice_idx = sample['slice_idx'].to(config['device'], dtype=torch.float)
            print(subj_id, slice_idx)
            pred, att_map_dict = model(inputs)

            if config['p'] == 1:
                loss_recon_y = torch.abs(pred - targets).mean()
            else:
                loss_recon_y = torch.pow(pred - targets, 2).mean()

            loss = config['lambda_recon_y'] * loss_recon_y

            loss_all_dict['recon_y'] += loss_recon_y.item()
            loss_all_dict['all'] += loss.item()

            metrics = compute_reconstruction_metrics(targets.detach().cpu().numpy(), pred.detach().cpu().numpy())

            print(metrics)

            for key in metrics.keys():
                if key in metrics_list_dict.keys():
                    metrics_list_dict[key].extend(metrics[key])
                else:
                    metrics_list_dict[key] = metrics[key]
            # print(metrics)
            # pdb.set_trace()

            if phase == 'test' and save_res:
                input_list.append(inputs.detach().cpu().numpy())
                target_list.append(targets.detach().cpu().numpy())
                y_fake_fused_list.append(pred.detach().cpu().numpy())
                print("====> Saving test result in default folder ./test_outputs/.... Please replace")
                os.makedirs(os.path.join("./test_outputs", subj_id[0]), exist_ok=True)
                
                subj_id_list.append(subj_id)
                slice_idx_list.append(slice_idx.detach().cpu().numpy())
                print(y_fake_fused_list[-1].shape)
                plt.imsave(os.path.join("./test_outputs", subj_id[0], str(slice_idx_list[-1][0]) + '.png'), y_fake_fused_list[-1].squeeze(), cmap='gray')
                mask_list.append(mask.detach().cpu().numpy())
                # if 'alpha_3' in att_map_dict:
                #     att_map_list2.append(att_map_dict['alpha_2'].detach().cpu().numpy())
                #     att_map_list3.append(att_map_dict['alpha_3'].detach().cpu().numpy())

            #
            # if iter > 10:
            #     break

    for key in loss_all_dict.keys():
        loss_all_dict[key] /= (iter + 1)

    for key in metrics_list_dict:
        loss_all_dict[key] = np.array(metrics_list_dict[key]).mean()

    if phase == 'test' and save_res:
        input_list = np.concatenate(input_list, axis=0)
        target_list = np.concatenate(target_list, axis=0)
        slice_idx_list = np.concatenate(slice_idx_list, axis=0)
        subj_id_list = np.concatenate(subj_id_list, axis=0)
        y_fake_fused_list = np.concatenate(y_fake_fused_list, axis=0)
        mask_list = np.concatenate(mask_list, axis=0)

        path = os.path.join(res_path, 'results_all'+info+config['postfix']+'.h5')
        if os.path.exists(path):
            print('Already saved h5')
        else:
            h5_file = h5py.File(path, 'w')
            h5_file.create_dataset('subj_id', data=np.string_(subj_id_list))
            h5_file.create_dataset('slice_idx', data=slice_idx_list)
            h5_file.create_dataset('inputs', data=input_list)
            h5_file.create_dataset('targets', data=target_list)
            h5_file.create_dataset('mask', data=mask_list)
            h5_file.create_dataset('y_fake_fused', data=y_fake_fused_list)
            # if len(att_map_list2) > 0:
            #     att_map_list2 = np.concatenate(att_map_list2, axis=0)
            #     att_map_list3 = np.concatenate(att_map_list3, axis=0)
            #     h5_file.create_dataset('att_map_2', data=att_map_list2)
            #     h5_file.create_dataset('att_map_3', data=att_map_list3)

        # save nifti
        nifti_path = os.path.join(res_path, 'nifti'+config['postfix']+'/')
        if not os.path.exists(nifti_path):
            os.mkdir(nifti_path)

        norm_data = h5py.File(config['data_norm_h5_path'], 'r')

        def renormalize(volume, norm_target, norm_type):
            if norm_type == 'z-score':
                volume_renormed = norm_target[1] * volume + norm_target[0]
            elif norm_type  == 'mean':
                volume_renormed = norm_target[0] * volume
            elif norm_type  == 'max':
                volume = np.clip(volume, a_min=0., a_max=1.0)
                volume_renormed = (norm_target[1]-norm_target[0]) * volume + norm_target[0]
            else:
                raise ValueError('Do not support this type of norm')
            return volume_renormed

        subj_id_list_uni, ind = np.unique(subj_id_list, return_index=True)
        subj_id_list_uni = subj_id_list_uni[np.argsort(ind)]

        for idx, subj_id in enumerate(subj_id_list_uni):
            pred = y_fake_fused_list[subj_id_list==subj_id].squeeze(1)
            target = target_list[subj_id_list==subj_id].squeeze(1)
            inputs = input_list[subj_id_list==subj_id]

            if 'PET' in norm_data[subj_id].keys():
                norm_target = np.array(norm_data[subj_id]['PET'])
            elif 'PET_MAC' in norm_data[subj_id].keys():
                norm_target = np.array(norm_data[subj_id]['PET_MAC'])
            elif 'PET_QCLEAR' in norm_data[subj_id].keys():
                norm_target = np.array(norm_data[subj_id]['PET_QCLEAR'])
            else:
                norm_target = np.array(norm_data[subj_id]['PET_TOF'])

            pred_renormed = renormalize(pred, norm_target, config['norm_type'])
            print("Saving normed....")
            print(pred_renormed.shape)
            for i in range(pred_renormed.shape[-1]):
                plt.imsave(os.path.join("./test_outputs", subj_id[0], str(i) + '_normed.png'), pred_renormed[:, :, i], cmap='gray')

            if 'rawspace' not in config['dataset_name']:
                pred_renormed = pred_renormed[:,:157,:189]

            os.makedirs(os.path.join(nifti_path, subj_id), exist_ok=True)
            # save_volume_nifti(os.path.join(nifti_path, subj_id, 'pred.nii'), pred_renormed)
            target_renormed = renormalize(target, norm_target, config['norm_type'])
            if 'rawspace' not in config['dataset_name']:
                target_renormed = target_renormed[:,:157,:189]
            save_volume_nifti(os.path.join(nifti_path, subj_id, 'target.nii'), target_renormed)

            for ic, contrast in enumerate(config['contrast_list']):
                if contrast == 'T1':
                    if subj_id+'/T1_GRE' in norm_data:
                        contrast = 'T1_GRE'
                    else:
                        contrast = 'T1_SE'
                elif contrast == 'T1c':
                    if subj_id+'/T1c_GRE' in norm_data:
                        contrast = 'T1c_GRE'
                    else:
                        contrast = 'T1c_SE'
                norm_contrast = np.array(norm_data[subj_id+'/'+contrast])
                contrast_renormed = renormalize(inputs[:,ic*(2*config['block_size']+1)+config['block_size']], norm_contrast, config['norm_type'])
                if 'rawspace' not in config['dataset_name']:
                    contrast_renormed = contrast_renormed[:,:157,:189]
                # volume = np.transpose(volume, (2, 0, 1))
                save_volume_nifti(os.path.join(nifti_path, subj_id, contrast + '.nii'), contrast_renormed)


    return loss_all_dict

if config['phase'] == 'train':
    train()
else:
    stat = evaluate(phase='test', set='test', save_res=True)
    print(stat)
