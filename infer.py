# import torch
# import data as Data
# import model as Model
# import argparse
# import logging
# import core.logger as Logger
# import core.metrics as Metrics
# from core.wandb_logger import WandbLogger
# from tensorboardX import SummaryWriter
# import os

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-c', '--config', type=str, default="/home/yuhu/experiments/Dehaze_NH_221024_163156/checkpoint/I80000_E1455", help='JSON file for configuration')
#     parser.add_argument('-p', '--phase', type=str, choices=['val'], help='val(generation)', default='val') 
#     parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
#     parser.add_argument('-debug', '-d', action='store_true')
#     parser.add_argument('-enable_wandb', action='store_true')
#     parser.add_argument('-log_infer', action='store_true')
    
#     # parse configs
#     args = parser.parse_args()
#     opt = Logger.parse(args)
#     # Convert to NoneDict, which return None for missing key.
#     opt = Logger.dict_to_nonedict(opt)

#     # logging
#     torch.backends.cudnn.enabled = True
#     torch.backends.cudnn.benchmark = True

#     Logger.setup_logger(None, opt['path']['log'], 'train', level=logging.INFO, screen=True)
#     Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
#     logger = logging.getLogger('base')
#     logger.info(Logger.dict2str(opt))
#     tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

#     # Initialize WandbLogger
#     if opt['enable_wandb']:
#         wandb_logger = WandbLogger(opt)
#     else:
#         wandb_logger = None

#     # dataset
#     for phase, dataset_opt in opt['datasets'].items():
#         if phase == 'val':
#             val_set = Data.create_dataset(dataset_opt, phase)
#             val_loader = Data.create_dataloader(val_set, dataset_opt, phase)
#     logger.info('Initial Dataset Finished')

#     # model
#     diffusion = Model.create_model(opt)
#     logger.info('Initial Model Finished')

#     diffusion.set_new_noise_schedule(opt['model']['beta_schedule']['val'], schedule_phase='val')
    
#     logger.info('Begin Model Inference.')
#     current_step = 0
#     current_epoch = 0
#     idx = 0

#     result_path = '{}'.format(opt['path']['results'])
#     os.makedirs(result_path, exist_ok=True)
#     avg_psnr = 0.0
#     for _,  val_data in enumerate(val_loader):
#         idx += 1

#         diffusion.feed_data(val_data)
#         diffusion.test(continous=True)
#         visuals = diffusion.get_current_visuals()
#         hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
#         lr_img = Metrics.tensor2img(visuals['LR'])  # uint8
#         sr_img_mode = 'grid'
#         if sr_img_mode == 'single':
#             # single img series
#             sr_img = visuals['Out']  # uint8
#             sample_num = sr_img.shape[0]
#             for iter in range(0, sample_num):
#                 Metrics.save_img(Metrics.tensor2img(sr_img[iter]), '{}/{}_{}_out_{}.png'.format(result_path, current_step, idx, iter))
#         else:
#             # grid img
#             sr_img = Metrics.tensor2img(visuals['Out'])  # uint8
#             #Metrics.save_img(sr_img, '{}/{}_{}_out_process.png'.format(result_path, current_step, idx))
#             Metrics.save_img(Metrics.tensor2img(visuals['Out'][-1]), '{}/{}_{}_out.png'.format(result_path, current_step, idx))

#         #Metrics.save_img(hr_img, '{}/{}_{}_hr.png'.format(result_path, current_step, idx))
#         Metrics.save_img(lr_img, '{}/{}_{}_lr.png'.format(result_path, current_step, idx))
#         avg_psnr += Metrics.calculate_psnr(Metrics.tensor2img(visuals['Out'][-1]), hr_img)

#         if wandb_logger and opt['log_infer']:
#             wandb_logger.log_eval_data(lr_img, Metrics.tensor2img(visuals['Out'][-1]), hr_img)

#     avg_psnr = avg_psnr / idx
#     diffusion.set_new_noise_schedule(opt['model']['beta_schedule']['train'], schedule_phase='train')
#     # log
#     logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))

#     if wandb_logger and opt['log_infer']:
#         wandb_logger.log_eval_table(commit=True)

import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter
import os
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default="/home/yuhu/experiments/Dehaze_NH_221024_163156/checkpoint/I80000_E1455", help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['val'], help='val(generation)', default='val') 
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', default = True,  action='store_true')
    parser.add_argument('-log_infer', action='store_true')
    
    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'], 'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # Initialize WandbLogger
    if opt['enable_wandb']:
        import wandb
        wandb_logger = WandbLogger(opt)
        wandb.define_metric('validation/val_step')
        wandb.define_metric('epoch')
        wandb.define_metric("validation/*", step_metric="val_step")
        val_step = 0
    else:
        wandb_logger = None

    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'val':
            val_set = Data.create_dataset(dataset_opt, phase)
            val_loader = Data.create_dataloader(val_set, dataset_opt, phase)
    logger.info('Initial Dataset Finished')

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    diffusion.set_new_noise_schedule(opt['model']['beta_schedule']['val'], schedule_phase='val')
    
    logger.info('Begin Model Inference.')
    current_step = 0
    current_epoch = 0
    idx = 0

    avg_psnr_only_asm = 0.0
    avg_ssim_only_asm = 0.0

    avg_psnr_only_sr3 = 0.0
    avg_ssim_only_sr3 = 0.0

    avg_psnr = 0.0
    avg_ssim = 0.0

    result_path = '{}'.format(opt['path']['results'])
    os.makedirs(result_path, exist_ok=True)
    for _,  val_data in enumerate(val_loader):
        idx += 1
        diffusion.feed_data(val_data)
        # diffusion.test(continous=True)
        diffusion.test_commbine(continous=True)
        # diffusion.test(continous=False)
        visuals = diffusion.get_current_visuals()
        out_img = Metrics.tensor2img(visuals['Out'])  # uint8
        hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
        lr_img = Metrics.tensor2img(visuals['LR'])  # uint8
        asm_img = Metrics.tensor2img(visuals['ASM'])  # uint8
        sr3_base_line = Metrics.tensor2img(visuals['SR3_baseline'])  # uint8
        
        #Generation
        Metrics.save_img(out_img, '{}/{}_{}_out.png'.format(result_path, current_step, idx))
        Metrics.save_img(hr_img, '{}/{}_{}_hr.png'.format(result_path, current_step, idx))
        Metrics.save_img(lr_img, '{}/{}_{}_lr.png'.format(result_path, current_step, idx))
        Metrics.save_img(asm_img, '{}/{}_{}_asm.png'.format(result_path, current_step, idx))
        Metrics.save_img(sr3_base_line, '{}/{}_{}_sr3_baseline.png'.format(result_path, current_step, idx))
        tb_logger.add_image('Iter_{}'.format(current_step), np.transpose(np.concatenate((lr_img, out_img, hr_img, asm_img, sr3_base_line), axis=1), [2, 0, 1]), idx)
        
        # avg_psnr += Metrics.calculate_psnr(sr_img, hr_img)
        # avg_ssim += Metrics.calculate_ssim(sr_img, hr_img)
        avg_psnr_only_asm += Metrics.calculate_psnr(asm_img, hr_img)
        avg_ssim_only_asm += Metrics.calculate_ssim(asm_img, hr_img)

        avg_psnr_only_sr3 += Metrics.calculate_psnr(sr3_base_line, hr_img)
        avg_ssim_only_sr3 += Metrics.calculate_ssim(sr3_base_line, hr_img)

        avg_psnr += Metrics.calculate_psnr(out_img, hr_img)
        avg_ssim += Metrics.calculate_ssim(out_img, hr_img)

        if wandb_logger:
            wandb_logger.log_image(f'validation_{idx}', np.concatenate((lr_img, asm_img, sr3_base_line, out_img, hr_img), axis=1))

        # if wandb_logger and opt['log_infer']:
        #     wandb_logger.log_eval_data(lr_img, Metrics.tensor2img(visuals['Out'][-1]), hr_img)
        if wandb_logger and opt['log_infer']:
            wandb_logger.log_eval_data(lr_img, out_img, hr_img, avg_psnr, avg_ssim)
            wandb_logger.log_eval_data(lr_img, asm_img, hr_img, avg_psnr_only_asm, avg_ssim_only_asm)
            wandb_logger.log_eval_data(lr_img, sr3_base_line, hr_img, avg_psnr_only_sr3, avg_ssim_only_sr3)

    avg_psnr_only_asm = avg_psnr_only_asm / idx
    avg_ssim_only_asm = avg_ssim_only_asm / idx

    avg_psnr_only_sr3 = avg_psnr_only_sr3 / idx
    avg_ssim_only_sr3 = avg_ssim_only_sr3 / idx

    avg_psnr = avg_psnr / idx
    avg_ssim = avg_ssim / idx

    # log
    # logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
    logger.info('# Inference Result Only ASM # PSNR: {:.4e}, SSIM: {:.4e}'.format(avg_psnr_only_asm, avg_ssim_only_asm))
    logger.info('# Inference Result Only SR3 Base Line # PSNR: {:.4e}, SSIM: {:.4e}'.format(avg_psnr_only_sr3, avg_ssim_only_sr3))
    logger.info('# Inference Result Full Pipeline # PSNR: {:.4e}, SSIM: {:.4e}'.format(avg_psnr, avg_ssim))

    logger_val = logging.getLogger('val')
    logger_val.info('<inference> Full Pipeline PSNR: {:.4e}, SSIM: {:.4e}'.format(avg_psnr, avg_ssim))
    logger_val.info('<inference> Only ASM PSNR: {:.4e}, SSIM: {:.4e}'.format(avg_psnr_only_asm, avg_ssim_only_asm))
    logger_val.info('<inference> Only SR3 Base Line PSNR: {:.4e}, SSIM: {:.4e}'.format(avg_psnr_only_sr3, avg_ssim_only_sr3))

    if wandb_logger and opt['log_infer']:
        wandb_logger.log_eval_table(commit=True)

    if wandb_logger:
        wandb_logger.log_metrics({
        'PSNR/ASM': float(avg_psnr_only_asm),
        'SSIM/ASM': float(avg_ssim_only_asm),
        'PSNR/SR3_Baseline': float(avg_psnr_only_sr3),
        'SSIM/SR3_Baseline': float(avg_ssim_only_sr3),
        'PSNR/Full_Pipeline': float(avg_psnr),
        'SSIM/Full_Pipeline': float(avg_ssim),
        'validation/val_step': val_step
        })
        val_step += 1
        if opt['log_infer']:
            wandb_logger.log_eval_table(commit=True)



