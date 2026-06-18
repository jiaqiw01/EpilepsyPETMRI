import ml_collections
import os
# TODO: change here
def get_default_configs():
    gpu = os.environ['GPU']
    config = ml_collections.ConfigDict()
    config.device = f"cuda:{gpu}"
    config.seed = 1

    config.contrast = ['PET_1p'] # change this
    config.slice_file = "" # change this
    config.slice_offset = 0
    config.guided_p_uncond = 0.3 # 0.1 # None 
    config.num_target_slices = 1 # change this

    config.label = 'pet_1p_2_pet' # change this

    config.normalisation = "image_scale"
    # sde configs
    config.sde = sde = ml_collections.ConfigDict()
    sde.type = "vpsde" # "vpsde", "vesde" "heatdiffusion"

    # the largest noise scale sigma_max was choosen according to Technique 1 from [https://arxiv.org/pdf/2006.09011.pdf], 
    if sde.type == "vesde":
        sde.sigma_min = 0.01
        sde.sigma_max = 40. #for 40 vpsde, 0.1 for heatidffusion
    if sde.type == "vpsde":
        # only for vpsde
        sde.beta_min = 0.1
        sde.beta_max = 5

    if sde.type == "heatdiffusion":
        # used for HeatDiffusion
        sde.T_max = 64

    # training configs
    config.training = training = ml_collections.ConfigDict()
    training.batch_size = 16
    training.epochs = 150
    training.log_freq = 25
    training.lr = 1e-4
    training.ema_decay = 0.999
    training.ema_warm_start_steps = 50 # only start updating ema after this amount of steps 

    # validation configs
    config.validation = validation = ml_collections.ConfigDict()
    validation.batch_size = 4
    validation.snr = 0.05
    validation.num_steps = 1000
    validation.eps = 1e-4
    validation.sample_freq = 10 #10

    # sampling configs 
    config.sampling = sampling = ml_collections.ConfigDict()
    sampling.batch_size = 1
    sampling.snr = 0.05
    sampling.num_steps = 1000 
    sampling.eps = 1e-4
    sampling.sampling_strategy = "predictor_corrector"
    sampling.start_time_step = 0

    sampling.load_model_from_path = "" # not used..
    sampling.model_name = "model.pt"


    # data configs - specify in other configs
    config.data = ml_collections.ConfigDict()
    config.data.im_size = 256

    # forward operator config - specify in other configs
    config.forward_op = ml_collections.ConfigDict()

    # model configs
    config.model = model = ml_collections.ConfigDict()
    model.model_name = 'OpenAiUNetModel'
    if config.guided_p_uncond == None:
        model.in_channels = 1
    else:
        model.in_channels = 2 * config.slice_offset + 1 + 1
    model.model_channels = 64
    model.out_channels = 1
    model.num_res_blocks = 3
    model.attention_resolutions = [32, 16]
    model.channel_mult = (1, 2, 2, 4, 4)
    model.conv_resample = True
    model.dims = 2
    model.num_heads = 4
    model.num_head_channels = -1
    model.num_heads_upsample = -1
    model.use_scale_shift_norm = True 
    model.resblock_updown = False
    model.use_new_attention_order = False
    model.max_period = 0.005


    return config