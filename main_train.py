


import torch
import argparse
import logging
from tqdm import tqdm
from model import DDPM
from Data import Dataset, Dataset_debug
from torch.utils.data import Subset
from tensorboardX import SummaryWriter
import os
import numpy as np
import yaml
from pathlib import Path
from model import data_transform_reverse
from guided_diffusion.train_util import tensor2img, save_img, calculate_psnr, load_partial_weights
from mc_dataset import datasets
import random
import json
from guided_diffusion.h_net.network import Classify_loss_time
import pandas as pd

def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def select_load(model, pretrained_path, exclude_key=['A']):
    pretrained_dict = torch.load(pretrained_path)


    model_dict = model.state_dict()
    
    filtered_dict = {}
    for k, v in pretrained_dict.items():
        exclude = False
        for pattern in exclude_key:
            if isinstance(pattern, str):
                if pattern in k:
                    exclude = True
                    break
            else: 
                import re
                if re.search(pattern, k):
                    exclude = True
                    break
        if not exclude:
            if k in model_dict:
                filtered_dict[k] = v
    
    model.load_state_dict(filtered_dict, strict=False)
    return model


def get_min_indices_sorted(lst):
    indexed_lst = list(enumerate(lst))
    sorted_indices = sorted(indexed_lst, key=lambda x: (x[1], x[0]))
    return [idx for idx, _ in sorted_indices[:3]]


def min_max_normalize_cols(tensor):
    min_vals = tensor.amin(dim=1, keepdim=True)  
    max_vals = tensor.amax(dim=1, keepdim=True)  
    return (tensor - min_vals) / (max_vals - min_vals + 1e-8)  

def data_normalization(data_dict, batch_num):
    label_matrix = torch.zeros((batch_num, 5))
    for i in range(1, 6):
        label_matrix[:, i-1] = data_dict[f'data{i}'][-1].view(-1)
    label_matrix = min_max_normalize_cols(label_matrix)
    return label_matrix

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="/mnt/ssd3/gyh/HomoDiff/config/test_pipeline_flir.yaml",
                        help="Path to config file (yaml/json)")


    args = parser.parse_args()
    cfg = load_config(args.config)

    for k, v in vars(args).items():
        if v is not None:
            cfg[k] = v 

    class Dict2Obj:
        def __init__(self, d): self.__dict__.update(d)
    args = Dict2Obj(cfg)
    seed_everything(args.seed)
    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True


    # model
    diffusion = DDPM(args)
    train_loader = datasets.fetch_dataloader(args, split="train")
    val_loader = datasets.fetch_dataloader(args, split="val")
    test_loader = datasets.fetch_dataloader(args, split="val")
    if args.single_pair:
        test_loader = datasets.fetch_dataloader(args, split="single_pair")
    current_step = 0
    current_epoch = 0
    n_iter = args.n_iter


    phase = args.phase
    if phase == 'train':
        total_batches = len(train_loader)
        total_epochs = (n_iter + total_batches - 1) // total_batches
        for epoch in range(current_epoch, total_epochs):
            current_epoch = epoch + 1  # 从1开始计数
            progress_bar = tqdm(
                train_loader,
                desc=f"Epoch {current_epoch}/{total_epochs}",
                unit="batch",
                leave=False
            )



            for _, train_data in enumerate(train_loader):


                if current_step % args.val_freq == 0 and args.test_level:
                    print('start validation for pipeline!')
                    avg_psnr = 0.0
                    idx = 0
                    result_path = '{}/{}'.format(args.result_path, current_epoch)
                    os.makedirs(result_path, exist_ok=True)

                    # diffusion.set_new_noise_schedule(args.beta_schedule, schedule_phase='val')
                    delta1 = 0
                    delta2 = 0
                    delta3 = 0
                    delta4 = 0
                    delta5 = 0
                    delta_exit = 0
                    delta_total = 0
                    delta_average_total = 0
                    delta_random_total = 0

                    delta1_mae = 0
                    delta2_mae = 0
                    delta3_mae = 0
                    delta4_mae = 0
                    delta5_mae = 0
                    delta_exit_mae = 0

                    delta1_mee = 0
                    delta2_mee = 0
                    delta3_mee = 0
                    delta4_mee = 0
                    delta5_mee = 0
                    delta_exit_mee = 0

                    px3_samples1 = 0
                    px3_samples2 = 0
                    px3_samples3 = 0
                    px3_samples4 = 0
                    px3_samples5 = 0
                    px3_samples_exit = 0
                    px3_samples_average = 0
                    px3_samples_random = 0

                    px5_samples1 = 0
                    px5_samples2 = 0
                    px5_samples3 = 0
                    px5_samples4 = 0
                    px5_samples5 = 0
                    px5_samples_exit = 0
                    px5_samples_average = 0
                    px5_samples_random = 0

                    px10_samples1 = 0
                    px10_samples2 = 0
                    px10_samples3 = 0
                    px10_samples4 = 0
                    px10_samples5 = 0
                    px10_samples_exit = 0
                    px10_samples_average = 0
                    px10_samples_random = 0
 

                    for _,  val_data in enumerate(test_loader):
                        idx += 1
                        # print('idx', idx-1)
                        # if idx >5:
                        #     exit()
                        diffusion.feed_data(val_data)
                        diffusion.test_pipeline(index=idx)
                        visuals = diffusion.get_current_visuals()

                        truth_img = tensor2img(data_transform_reverse(visuals['truth']))  # uint8
                        noisy_img = tensor2img(data_transform_reverse(visuals['noisy']))  # uint8
                        clean_img = tensor2img(data_transform_reverse(visuals['clean']))  # uint8

                        delta_per1 = visuals['homo_loss1']['rmse']
                        delta_per2 = visuals['homo_loss2']['rmse']
                        delta_per3 = visuals['homo_loss3']['rmse']
                        delta_per4 = visuals['homo_loss4']['rmse']
                        delta_per5 = visuals['homo_loss5']['rmse']
                        all_deltas = [delta_per1, delta_per2, delta_per3, delta_per4, delta_per5]
                        delta_average = (delta_per1 + delta_per2 + delta_per3 + delta_per4 + delta_per5)/5
                        delta_random = random.choice(all_deltas)
                        if args.load_exit:
                            delta_exit_per = visuals['exit_rmse']
                            


                        if delta_per1.item() <= 3:
                            px3_samples1 += 1
                        elif delta_per1.item() <= 5:
                            px5_samples1 += 1
                        elif delta_per1.item() <= 10:
                            px10_samples1 += 1  

                        if delta_per2.item() <= 3:
                            px3_samples2 += 1
                        elif delta_per2.item() <= 5:
                            px5_samples2 += 1
                        elif delta_per2.item() <= 10:
                            px10_samples2 += 1 

                        if delta_per3.item() <= 3:
                            px3_samples3 += 1
                        elif delta_per3.item() <= 5:
                            px5_samples3 += 1
                        elif delta_per3.item() <= 10:
                            px10_samples3 += 1 

                        if delta_per4.item() <= 3:
                            px3_samples4 += 1
                        elif delta_per4.item() <= 5:
                            px5_samples4 += 1
                        elif delta_per4.item() <= 10:
                            px10_samples4 += 1 

                        if delta_per5.item() <= 3:
                            px3_samples5 += 1
                        elif delta_per5.item() <= 5:
                            px5_samples5 += 1
                        elif delta_per5.item() <= 10:
                            px10_samples5 += 1 

                        if delta_average.item() <= 3:
                            px3_samples_average += 1
                        elif delta_average.item() <= 5:
                            px5_samples_average += 1
                        elif delta_average.item() <= 10:
                            px10_samples_average += 1 

                        if delta_random.item() <= 3:
                            px3_samples_random += 1
                        elif delta_random.item() <= 5:
                            px5_samples_random += 1
                        elif delta_random.item() <= 10:
                            px10_samples_random += 1 

                        if args.load_exit:
                            if delta_exit_per.item() <= 3:
                                px3_samples_exit += 1
                            elif delta_exit_per.item() <= 5:
                                px5_samples_exit += 1
                            elif delta_exit_per.item() <= 10:
                                px10_samples_exit += 1                        

                        delta_per1_mae = visuals['homo_loss1']['mae']
                        delta_per2_mae = visuals['homo_loss2']['mae']
                        delta_per3_mae = visuals['homo_loss3']['mae']
                        delta_per4_mae = visuals['homo_loss4']['mae']
                        delta_per5_mae = visuals['homo_loss5']['mae']
                        if args.load_exit:
                            delta_exit_per_mae = visuals['exit_mae']

                        delta_per1_mee = visuals['homo_loss1']['mee']
                        delta_per2_mee = visuals['homo_loss2']['mee']
                        delta_per3_mee = visuals['homo_loss3']['mee']
                        delta_per4_mee = visuals['homo_loss4']['mee']
                        delta_per5_mee = visuals['homo_loss5']['mee']
                        if args.load_exit:
                            delta_exit_per_mee = visuals['exit_mee']
                            print('homo1_per, homo2_per, homo3_per, homo4_per, homo5_per, delta_exit_per', delta_per1, delta_per2, delta_per3, delta_per4, delta_per5, delta_exit_per)
                        else:
                            print('homo1_per, homo2_per, homo3_per, homo4_per, homo5_per', delta_per1, delta_per2, delta_per3, delta_per4, delta_per5)

                        # generation
                        save_img(
                            truth_img, '{}/{}_{}_truth.png'.format(result_path, current_step, idx))
                        # save_img(
                        #     denoise_img, '{}/{}_{}_denoise.png'.format(result_path, current_step, idx))
                        save_img(
                            noisy_img, '{}/{}_{}_noisy.png'.format(result_path, current_step, idx))
                        save_img(
                            clean_img, '{}/{}_{}_clean.png'.format(result_path, current_step, idx))
                        # save_img(
                        #     x0_pred_img, '{}/{}_{}_x0_pred_time{}.png'.format(result_path, current_step, idx, t))
                        # tb_logger.add_image(
                        #     'Iter_{}'.format(current_step),
                        #     np.transpose(np.concatenate(
                        #         (fake_img, sr_img, hr_img), axis=1), [2, 0, 1]),
                        #     idx)
                        delta1 += delta_per1
                        delta2 += delta_per2
                        delta3 += delta_per3
                        delta4 += delta_per4
                        delta5 += delta_per5
                        delta_average_total += delta_average
                        delta_random_total += delta_random
                        if args.load_exit:
                            delta_exit += delta_exit_per
                        # avg_psnr += calculate_psnr(
                        #     denoise_img, truth_img)
                        delta1_mae += delta_per1_mae
                        delta2_mae += delta_per2_mae
                        delta3_mae += delta_per3_mae
                        delta4_mae += delta_per4_mae
                        delta5_mae += delta_per5_mae
                        if args.load_exit:
                            delta_exit_mae += delta_exit_per_mae

                        delta1_mee += delta_per1_mee
                        delta2_mee += delta_per2_mee
                        delta3_mee += delta_per3_mee
                        delta4_mee += delta_per4_mee
                        delta5_mee += delta_per5_mee
                        if args.load_exit:
                            delta_exit_mee += delta_exit_per_mee


                    # avg_psnr = avg_psnr / idx
                    avg_delta1 = delta1 / idx
                    avg_delta2 = delta2 / idx
                    avg_delta3 = delta3 / idx
                    avg_delta4 = delta4 / idx
                    avg_delta5 = delta5 / idx
                    avg_delta_average = delta_average_total / idx
                    avg_delta_random = delta_random_total / idx
                    if args.load_exit:
                        avg_delta_exit = delta_exit / idx

                    avg_delta1_mae = delta1_mae / idx
                    avg_delta2_mae = delta2_mae / idx
                    avg_delta3_mae = delta3_mae / idx
                    avg_delta4_mae = delta4_mae / idx
                    avg_delta5_mae = delta5_mae / idx
                    if args.load_exit:
                        avg_delta_exit_mae = delta_exit_mae / idx

                    avg_delta1_mee = delta1_mee / idx
                    avg_delta2_mee = delta2_mee / idx
                    avg_delta3_mee = delta3_mee / idx
                    avg_delta4_mee = delta4_mee / idx
                    avg_delta5_mee = delta5_mee / idx
                    if args.load_exit:
                        avg_delta_exit_mee = delta_exit_mee / idx
                    with open(args.result_txt, 'w') as f:
                    # print('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}'.format(current_epoch, current_step, avg_psnr))
                        print('<epoch:{:3d}, iter:{:8,d}> homo1_rmse: {:.4e}'.format(current_epoch, current_step, avg_delta1.item()), file=f)
                        print('<epoch:{:3d}, iter:{:8,d}> homo2_rmse: {:.4e}'.format(current_epoch, current_step, avg_delta2.item()), file=f)
                        print('<epoch:{:3d}, iter:{:8,d}> homo3_rmse: {:.4e}'.format(current_epoch, current_step, avg_delta3.item()), file=f)
                        print('<epoch:{:3d}, iter:{:8,d}> homo4_rmse: {:.4e}'.format(current_epoch, current_step, avg_delta4.item()), file=f)
                        print('<epoch:{:3d}, iter:{:8,d}> homo5_rmse: {:.4e}'.format(current_epoch, current_step, avg_delta5.item()), file=f)
                        print('<epoch:{:3d}, iter:{:8,d}> homoavg_rmse: {:.4e}'.format(current_epoch, current_step, avg_delta_average.item()), file=f)
                        print('<epoch:{:3d}, iter:{:8,d}> homorandom_rmse: {:.4e}'.format(current_epoch, current_step, avg_delta_random.item()), file=f)
                        if args.load_exit:
                            print('<epoch:{:3d}, iter:{:8,d}> homo_exit_rmse: {:.4e}'.format(current_epoch, current_step, avg_delta_exit.item()), file=f)

                        print('<epoch:{:3d}, iter:{:8,d}> homo1_mae: {:.4e}'.format(current_epoch, current_step, avg_delta1_mae.item()), file=f)
                        print('<epoch:{:3d}, iter:{:8,d}> homo2_mae: {:.4e}'.format(current_epoch, current_step, avg_delta2_mae.item()), file=f)
                        print('<epoch:{:3d}, iter:{:8,d}> homo3_mae: {:.4e}'.format(current_epoch, current_step, avg_delta3_mae.item()), file=f)
                        print('<epoch:{:3d}, iter:{:8,d}> homo4_mae: {:.4e}'.format(current_epoch, current_step, avg_delta4_mae.item()), file=f)
                        print('<epoch:{:3d}, iter:{:8,d}> homo5_mae: {:.4e}'.format(current_epoch, current_step, avg_delta5_mae.item()), file=f)
                        if args.load_exit:
                            print('<epoch:{:3d}, iter:{:8,d}> homo_exit_mae: {:.4e}'.format(current_epoch, current_step, avg_delta_exit_mae.item()), file=f)

                        print('<epoch:{:3d}, iter:{:8,d}> homo1_mee: {:.4e}'.format(current_epoch, current_step, avg_delta1_mee.item()), file=f)
                        print('<epoch:{:3d}, iter:{:8,d}> homo2_mee: {:.4e}'.format(current_epoch, current_step, avg_delta2_mee.item()), file=f)
                        print('<epoch:{:3d}, iter:{:8,d}> homo3_mee: {:.4e}'.format(current_epoch, current_step, avg_delta3_mee.item()), file=f)
                        print('<epoch:{:3d}, iter:{:8,d}> homo4_mee: {:.4e}'.format(current_epoch, current_step, avg_delta4_mee.item()), file=f)
                        print('<epoch:{:3d}, iter:{:8,d}> homo5_mee: {:.4e}'.format(current_epoch, current_step, avg_delta5_mee.item()), file=f)
                        if args.load_exit:
                            print('<epoch:{:3d}, iter:{:8,d}> homo_exit_mee: {:.4e}'.format(current_epoch, current_step, avg_delta_exit_mee.item()), file=f)

                        print('3px_samples1, 5px_samples1', '10px_samples1', 'total_samples', px3_samples1, px5_samples1+px3_samples1, px10_samples1+px5_samples1+px3_samples1, len(test_loader), file=f)
                        print('3px_samples2, 5px_samples2', '10px_samples2', 'total_samples', px3_samples2, px5_samples2+px3_samples2, px10_samples2+px5_samples2+px3_samples2, len(test_loader), file=f)
                        print('3px_samples3, 5px_samples3', '10px_samples3', 'total_samples', px3_samples3, px5_samples3+px3_samples3, px10_samples3+px5_samples3+px3_samples3, len(test_loader), file=f)
                        print('3px_samples4, 5px_samples4', '10px_samples4', 'total_samples', px3_samples4, px5_samples4+px3_samples4, px10_samples4+px5_samples4+px3_samples4, len(test_loader), file=f)
                        print('3px_samples5, 5px_samples5', '10px_samples5', 'total_samples', px3_samples5, px5_samples5+px3_samples5, px10_samples5+px5_samples5+px3_samples5, len(test_loader), file=f)
                        print('average', file=f)
                        print('3px_samples5, 5px_samples5', '10px_samples5', 'total_samples', px3_samples_average, px5_samples_average+px3_samples_average, px10_samples_average+px5_samples_average+px3_samples_average, len(test_loader), file=f)
                        print('random', file=f)
                        print('3px_samples5, 5px_samples5', '10px_samples5', 'total_samples', px3_samples_random, px5_samples_random+px3_samples_random, px10_samples_random+px5_samples_random+px3_samples_random, len(test_loader), file=f)
                        print('3px_samples_exit, 5px_samples_exit', '10px_samples_exit', 'total_samples', px3_samples_exit, px5_samples_exit+px3_samples_exit, px10_samples_exit+px5_samples_exit+px3_samples_exit, len(test_loader), file=f)
                    exit()

                # validation
                if current_step % args.val_freq == 1:
                    print('start validation!')
                    avg_psnr = 0.0
                    idx = 0
                    result_path = '{}/{}'.format(args.result_path, current_epoch)
                    os.makedirs(result_path, exist_ok=True)

                    # diffusion.set_new_noise_schedule(args.beta_schedule, schedule_phase='val')
                    delta = 0
                    for _,  val_data in enumerate(val_loader):
                        idx += 1
                        diffusion.feed_data(val_data)
                        diffusion.test(continous=False)
                        visuals = diffusion.get_current_visuals(get_denoise=True, get_t=True)
                        denoise_img = tensor2img(visuals['denoise'])  # uint8
                        truth_img = tensor2img(data_transform_reverse(visuals['truth']))  # uint8
                        noisy_img = tensor2img(data_transform_reverse(visuals['noisy']))  # uint8
                        clean_img = tensor2img(data_transform_reverse(visuals['clean']))  # uint8
                        x0_pred_img = tensor2img(data_transform_reverse(visuals['x0_pred']))
                        delta_per = visuals['homo']
                        t = visuals['t']

                        # generation
                        save_img(
                            truth_img, '{}/{}_{}_truth.png'.format(result_path, current_step, idx))
                        save_img(
                            denoise_img, '{}/{}_{}_denoise.png'.format(result_path, current_step, idx))
                        save_img(
                            noisy_img, '{}/{}_{}_noisy.png'.format(result_path, current_step, idx))
                        save_img(
                            clean_img, '{}/{}_{}_clean.png'.format(result_path, current_step, idx))
                        save_img(
                            x0_pred_img, '{}/{}_{}_x0_pred_time{}.png'.format(result_path, current_step, idx, t))
                        # tb_logger.add_image(
                        #     'Iter_{}'.format(current_step),
                        #     np.transpose(np.concatenate(
                        #         (fake_img, sr_img, hr_img), axis=1), [2, 0, 1]),
                        #     idx)
                        delta += delta_per
                        avg_psnr += calculate_psnr(
                            denoise_img, truth_img)


                    avg_psnr = avg_psnr / idx
                    avg_delta = delta / idx
                    # print(avg_delta)
                    print('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}'.format(current_epoch, current_step, avg_psnr))
                    print('<epoch:{:3d}, iter:{:8,d}> homo: {:.4e}'.format(current_epoch, current_step, avg_delta.item()))
                current_step += 1

                if current_step > n_iter:
                    break
                diffusion.feed_data(train_data)
                loss_dict = diffusion.optimize_parameters()
                if current_step % args.print_freq == 0:
                    print('current step:%d'%current_step, end='\t')
                    for key, value in loss_dict.items():
                        print(key, ': %.4f'%value, end='\t')
                    print('')
                if current_step % args.save_freq == 0:
                    checkpoint_path = os.path.join(
                        args.save_path,
                        f"checkpoint_epoch{current_epoch}_step{current_step}.pth"
                    )

                    if args.diffusion_phase[:-1] == 'homo':
                        torch.save(diffusion.netG.h_model_layer1.state_dict(), checkpoint_path)
                    elif args.diffusion_phase == 'denoise':
                        torch.save(diffusion.netG.denoise_model.state_dict(), checkpoint_path)
                    print(f"Saved checkpoint at step {current_step} → {checkpoint_path}")

    elif phase == 'exit_train':
            device = torch.device('cuda')
            exit_model = Classify_loss_time()
            select_load(exit_model, args.feature_path, exclude_key=['update_block_4', 'fc'])
            exit_model.to(device)
            criterion = torch.nn.MSELoss()
            optim_params = []

            # for k, v in exit_model.update_block_4.named_parameters():
            for k, v in exit_model.named_parameters():
                optim_params.append(v)
            for k, v in exit_model.named_parameters():
                v.requires_grad = True


            optG = torch.optim.Adam(
                optim_params, lr=1e-4)
            # Train
            # current_step = diffusion.begin_step
            # current_epoch = diffusion.begin_epoch
            current_step = 0
            current_epoch = 0
            n_iter = args.n_iter
            total_batches = len(train_loader)
            total_epochs = (n_iter + total_batches - 1) // total_batches
            for epoch in range(current_epoch, total_epochs):
                current_epoch = epoch + 1  
                progress_bar = tqdm(
                    train_loader,
                    desc=f"Epoch {current_epoch}/{total_epochs}",
                    unit="batch",
                    leave=False
                )
            while current_step < n_iter:
                current_epoch += 1
                for _, train_data in enumerate(train_loader):
                    print('current step:', current_step)

                    # result_path = '{}/{}'.format(args.result_path, current_epoch)
                    # os.makedirs(result_path, exist_ok=True)

                    data = {}
                    delta_total = 0
                    diffusion.feed_data(train_data)
                    diffusion.test_pipeline()
                    visuals = diffusion.get_current_visuals()
                    batch_num = visuals[f'homo_loss1']['rmse'].shape[0]
                    data['refer'] = train_data['clean'].to(device)


                    # homo_losses_compare = [visuals[f'homo_loss{i}_compare'] for i in range(1, 6)]
                    # print('homo_losses', homo_losses)
                    # print('homo_losses_compare', homo_losses_compare)
                    

                    homo_losses = torch.zeros(batch_num, 5)
                    # homo_losses = torch.tensor([visuals[f'homo_loss{j}'] for j in range(1, 6)])
                    
                    for b in range(batch_num):
                        homo_losses[b] = torch.tensor([visuals[f'homo_loss{j}']['rmse'][b] for j in range(1, 6)])

                    
                    for i in range(1, 6):
                        x_pred = visuals[f'x0_pred'][i-1].to(device)
                        loss_i = torch.zeros(batch_num, 1).to(device)
                        for b in range(batch_num):
                            loss_i[b] = homo_losses[b][i-1].to(device) 
                        data[f'data{i}'] = [x_pred, loss_i]
                        # print('label', data[f'data{i}'][-1])
                    
                    # print('homo_losses', homo_losses)
                    label_matrix = data_normalization(data, batch_num)
                    # print('label matrix for train', label_matrix)
                    for i in range(1, 6):
                        data[f'data{i}'][-1] = label_matrix[:, i-1].view(batch_num, 1).to(device)
                    

                    # t = visuals['t']
                    for i in range(1, 6):
                        optG.zero_grad()
                        sample, refer, label = data[f'data{i}'][0], data['refer'], data[f'data{i}'][-1]
                        pred = exit_model(refer, sample, four_point_disp=visuals[f'delta{i}'], time=torch.tensor([200*i]).to(device), idx=2)
                        loss = criterion(pred, label)
                        # print('pred shape:', pred.shape, 'label shape:', label.shape)
                        with torch.no_grad():
                            if current_step % args.print_freq == 0:
                                print('pred:', '\n', np.array(pred.detach().cpu()), '\n', 'label:', '\n', np.array(label.detach().cpu()), '\n', 'loss:', '\n', np.array(loss.detach().cpu()))
                            # print('loss now:', loss.item(), 'compare now:', loss.item())
                        loss.backward()
                        optG.step()
                    
                


                    # validation
                    if current_step % args.val_freq == 0:
                        print('start validation!')
                        exit_model.eval()
                        # result_path = '{}/{}'.format(args.result_path, current_epoch)
                        # os.makedirs(result_path, exist_ok=True)

                        # diffusion.set_new_noise_schedule(args.beta_schedule, schedule_phase='val')
                        delta = 0
                        idx = 0
                        ridx1 = 0
                        ridx3 = 0
                        for _,  val_data in enumerate(val_loader):
                            idx += 1
                            data = {}
                            delta_total = 0
                            diffusion.feed_data(val_data)
                            diffusion.test_pipeline()
                            visuals = diffusion.get_current_visuals()
                            homo_losses = [visuals[f'homo_loss{i}']['rmse'][0] for i in range(1, 6)]
                            min_loss = min(homo_losses).to(device)
                            min_index1, min_index2, min_index3 = get_min_indices_sorted(homo_losses)
                            print('homo_losses:', homo_losses)
                            print('min_index1, min_index2, min_index3', min_index1, min_index2, min_index3)
                            data['refer'] = val_data['clean'].to(device)
                            for i in range(1, 6):
                                x_pred = visuals[f'x0_pred'][i-1].to(device)
                                loss_i = homo_losses[i-1].to(device) 
                                data[f'data{i}'] = [x_pred, loss_i]
                            # t = visuals['t']
                            min_idx = 0
                            min_value = 100
                            label_matrix = data_normalization(data, 1)
                            for i in range(1, 6):
                                data[f'data{i}'][-1] = label_matrix[:, i-1].view(1, 1).to(device)
                                # print('label', data[f'data{i}'][-1].detach().cpu())
                            for i in range(1, 6):
                                sample, refer, label = data[f'data{i}'][0], data['refer'], data[f'data{i}'][-1]
                                
                                with torch.no_grad():
                                    # print('refer.shape, sample.shape, visuals[delta1].shape', refer.shape, sample.shape, visuals['delta1'].shape)
                                    pred = exit_model(refer, sample, four_point_disp=visuals[f'delta{i}'], time=torch.tensor([200*i]).to(device), idx=2)
                                    # pred = exit_model(refer, sample)
                                    # print('abs(pred.item()-1), min_value', abs(pred.item()-1), min_value)
                                    if pred.item() < min_value:
                                        # print('start iter')
                                        min_value = pred.item()
                                        min_idx = i-1
                            # if idx%10 == 0:
                            print('min_idx, min_index1', min_idx, min_index1)
                            if min_idx == min_index1:
                                ridx1 += 1
                                ridx3 += 1
                            elif min_idx == min_index2 or min_idx == min_index3:
                                ridx3 += 1
                        acc1 = ridx1/idx
                        acc3 = ridx3/idx
                        print('acc1:', acc1, 'acc3:', acc3)
                                
                        exit_model.train()
                    current_step += 1

                    if current_step > n_iter:
                        break
                    if current_step % args.save_freq == 0:
                        checkpoint_path = os.path.join(
                            args.save_path,
                            f"checkpoint_epoch{current_epoch}_step{current_step}.pth"
                        )
                        torch.save({
                            'epoch': current_epoch,
                            'step': current_step,
                            'model_classify': exit_model.state_dict(),
                            'args': args  
                        }, checkpoint_path)
                        print(f"Saved checkpoint at step {current_step} → {checkpoint_path}")
                    # log
                    # if current_step % opt['train']['print_freq'] == 0:
                    #     logs = diffusion.get_current_log()
                    #     message = '<epoch:{:3d}, iter:{:8,d}> '.format(
                    #         current_epoch, current_step)
                    #     for k, v in logs.items():
                    #         message += '{:s}: {:.4e} '.format(k, v)
                    #         tb_logger.add_scalar(k, v, current_step)
                    #     logger.info(message)

                    #     if wandb_logger:
                    #         wandb_logger.log_metrics(logs)


                        # diffusion.set_new_noise_schedule(
                        #     opt['model']['beta_schedule']['train'], schedule_phase='train')
                        # log
                        # logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
                        # logger_val = logging.getLogger('val')  # validation logger
                        # logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}'.format(
                        #     current_epoch, current_step, avg_psnr))
                        # tensorboard logger
                        # tb_logger.add_scalar('psnr', avg_psnr, current_step)

                        # if wandb_logger:
                        #     wandb_logger.log_metrics({
                        #         'validation/val_psnr': avg_psnr,
                        #         'validation/val_step': val_step
                        #     })
                        #     val_step += 1

                #     if current_step % opt['train']['save_checkpoint_freq'] == 0:
                #         logger.info('Saving models and training states.')
                #         diffusion.save_network(current_epoch, current_step)

                #         if wandb_logger and opt['log_wandb_ckpt']:
                #             wandb_logger.log_checkpoint(current_epoch, current_step)

                # if wandb_logger:
                #     wandb_logger.log_metrics({'epoch': current_epoch-1})

            # save model
            # logger.info('End of training.')
        # else:
        #     logger.info('Begin Model Evaluation.')
        #     avg_psnr = 0.0
        #     avg_ssim = 0.0
        #     idx = 0
        #     result_path = '{}'.format(opt['path']['results'])
        #     os.makedirs(result_path, exist_ok=True)
        #     for _,  val_data in enumerate(val_loader):
        #         idx += 1
        #         diffusion.feed_data(val_data)
        #         diffusion.test(continous=True)
        #         visuals = diffusion.get_current_visuals()

        #         hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
        #         lr_img = Metrics.tensor2img(visuals['LR'])  # uint8
        #         fake_img = Metrics.tensor2img(visuals['INF'])  # uint8

        #         sr_img_mode = 'grid'
        #         if sr_img_mode == 'single':
        #             # single img series
        #             sr_img = visuals['SR']  # uint8
        #             sample_num = sr_img.shape[0]
        #             for iter in range(0, sample_num):
        #                 Metrics.save_img(
        #                     Metrics.tensor2img(sr_img[iter]), '{}/{}_{}_sr_{}.png'.format(result_path, current_step, idx, iter))
        #         else:
        #             # grid img
        #             sr_img = Metrics.tensor2img(visuals['SR'])  # uint8
        #             Metrics.save_img(
        #                 sr_img, '{}/{}_{}_sr_process.png'.format(result_path, current_step, idx))
        #             Metrics.save_img(
        #                 Metrics.tensor2img(visuals['SR'][-1]), '{}/{}_{}_sr.png'.format(result_path, current_step, idx))

        #         Metrics.save_img(
        #             hr_img, '{}/{}_{}_hr.png'.format(result_path, current_step, idx))
        #         Metrics.save_img(
        #             lr_img, '{}/{}_{}_lr.png'.format(result_path, current_step, idx))
        #         Metrics.save_img(
        #             fake_img, '{}/{}_{}_inf.png'.format(result_path, current_step, idx))

        #         # generation
        #         eval_psnr = Metrics.calculate_psnr(Metrics.tensor2img(visuals['SR'][-1]), hr_img)
        #         eval_ssim = Metrics.calculate_ssim(Metrics.tensor2img(visuals['SR'][-1]), hr_img)

        #         avg_psnr += eval_psnr
        #         avg_ssim += eval_ssim

        #         if wandb_logger and opt['log_eval']:
        #             wandb_logger.log_eval_data(fake_img, Metrics.tensor2img(visuals['SR'][-1]), hr_img, eval_psnr, eval_ssim)

        #     avg_psnr = avg_psnr / idx
        #     avg_ssim = avg_ssim / idx

        #     # log
        #     logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
        #     logger.info('# Validation # SSIM: {:.4e}'.format(avg_ssim))
        #     logger_val = logging.getLogger('val')  # validation logger
        #     logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}, ssim：{:.4e}'.format(
        #         current_epoch, current_step, avg_psnr, avg_ssim))

        #     if wandb_logger:
        #         if opt['log_eval']:
        #             wandb_logger.log_eval_table()
        #         wandb_logger.log_metrics({
        #             'PSNR': float(avg_psnr),
        #             'SSIM': float(avg_ssim)
        #         })
