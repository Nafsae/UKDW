import argparse
import os
import re
import shutil
import tempfile
from collections import OrderedDict
from glob import glob
import glob as glob_module
import random
import numpy as np

import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml

from albumentations import HorizontalFlip, VerticalFlip, RandomRotate90, Resize
from albumentations import transforms

from albumentations.core.composition import Compose, OneOf
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from tqdm import tqdm

from models import UKDW
from models import UKAN_ASFF

from utils.losses import BCEDiceLoss, LovaszHingeLoss, FocalDiceLoss
from utils.metrics import iou_score, dice_coef, indicators
from datasets import Dataset

from utils.misc import AverageMeter, str2bool

# from tensorboardX import SummaryWriter

import subprocess

from pdb import set_trace as st


LOSS_NAMES = ['BCEDiceLoss', 'LovaszHingeLoss']
LOSS_NAMES.append('BCEWithLogitsLoss')
LOSS_NAMES.append('FocalDiceLoss')


def list_type(s):
    str_list = s.split(',')
    int_list = [int(a) for a in str_list]
    return int_list


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=400, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=8, type=int,
                        metavar='N', help='mini-batch size (default: 16)')

    parser.add_argument('--dataseed', default=2981, type=int,
                        help='')
    
    # parser.add_argument('--arch', '-a', metavar='ARCH', default='UKAN')
    
    parser.add_argument('--deep_supervision', default=False, type=str2bool)
    parser.add_argument('--input_channels', default=3, type=int,
                        help='input channels')
    parser.add_argument('--num_classes', default=1, type=int,
                        help='number of classes')
    parser.add_argument('--input_w', default=256, type=int,
                        help='image width')
    parser.add_argument('--input_h', default=256, type=int,
                        help='image height')
    parser.add_argument('--input_list', type=list_type, default=[256, 320, 512])

    # loss
    parser.add_argument('--loss', default='BCEDiceLoss',
                        choices=LOSS_NAMES,
                        help='loss: ' +
                        ' | '.join(LOSS_NAMES) +
                        ' (default: BCEDiceLoss)')
    
    # dataset
    parser.add_argument('--dataset', default='busi', help='dataset name')      
    parser.add_argument('--data_dir', default='inputs', help='dataset dir')

    parser.add_argument('--output_dir', default='/root/autodl-tmp/U-KAN/Seg_UKAN_SDI_v3/outputs', help='ouput dir')


    # optimizer
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD', 'AdamW'],
                        help='loss: ' +
                        ' | '.join(['Adam', 'SGD', 'AdamW']) +
                        ' (default: Adam)')

    parser.add_argument('--lr', '--learning_rate', default=1e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=5e-5, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')

    parser.add_argument('--kan_lr', default=5e-3, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--kan_weight_decay', default=1e-4, type=float,
                        help='weight decay')

    # scheduler
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-5, type=float,
                        help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2/3, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int,
                        metavar='N', help='early stopping (default: -1)')
    parser.add_argument('--cfg', type=str, metavar="FILE", help='path to config file', )
    parser.add_argument('--num_workers', default=4, type=int)

    parser.add_argument('--no_kan', action='store_true')


    
    parser.add_argument('--detail_asff_channel', default=24, type=int,
                        help='ASFF channel dimension for detail group (t1, t2)')
    parser.add_argument('--semantic_asff_channel', default=36, type=int,
                        help='ASFF channel dimension for semantic group (t3, t4)')
    parser.add_argument('--use_detail_attention', default=False, type=str2bool,
                        help='Whether to use residual attention in detail group')
    parser.add_argument('--attention_alpha', default=0.1, type=float,
                        help='Initial alpha value for residual attention')
    
    parser.add_argument('--dropout', default=0.3, type=float,
                        help='Dropout probability for neural network layers')
    parser.add_argument('--drop_path', default=0.1, type=float,
                        help='Drop path probability for neural network paths')
    
    parser.add_argument('--focal_gamma', default=2.0, type=float,
                        help='Gamma parameter for Focal Loss component when using FocalDiceLoss')
    parser.add_argument('--focal_alpha', default=0.25, type=float,
                        help='Alpha parameter for Focal Loss component when using FocalDiceLoss')

    parser.add_argument('--augmentation', default=False, type=str2bool,
                        help='Whether to use enhanced data augmentation')

    parser.add_argument('--enable_wavelet', default=True, type=str2bool,
                        help='Whether to enable wavelet preprocessing (v4 innovation)')
    parser.add_argument('--wavelet_denoise_threshold', default=0.08, type=float,
                        help='Threshold for wavelet denoising (0.05-0.15 recommended)')

    config = parser.parse_args()

    return config


def train(config, train_loader, model, criterion, optimizer):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter()}

    model.train()

    pbar = tqdm(total=len(train_loader))
    for input, target, _ in train_loader:
        input = input.cuda()
        target = target.cuda()

        # compute output
        if config['deep_supervision']:
            outputs = model(input)
            loss = 0
            for output in outputs:
                loss += criterion(output, target)
            loss /= len(outputs)
            iou, _, _ = iou_score(outputs[0], target)
        else:
            output = model(input)
            loss = criterion(output, target)
            iou, _, _ = iou_score(output, target)

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg)])


def validate(config, val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'dice': AverageMeter(),
                  'hd': AverageMeter(),
                  'hd95': AverageMeter(),
                  'recall': AverageMeter(),
                  'specificity': AverageMeter(),
                  'precision': AverageMeter()}

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target, _ in val_loader:
            input = input.cuda()
            target = target.cuda()

            # compute output
            if config['deep_supervision']:
                outputs = model(input)
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
                output = outputs[0]
            else:
                output = model(input)
                loss = criterion(output, target)

            iou_, dice_, hd_, hd95_, recall_, specificity_, precision_ = indicators(output, target)

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou_, input.size(0))
            avg_meters['dice'].update(dice_, input.size(0))
            avg_meters['hd'].update(hd_, input.size(0))
            avg_meters['hd95'].update(hd95_, input.size(0))
            avg_meters['recall'].update(recall_, input.size(0))
            avg_meters['specificity'].update(specificity_, input.size(0))
            avg_meters['precision'].update(precision_, input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
                ('dice', avg_meters['dice'].avg)
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice', avg_meters['dice'].avg),
                        ('hd', avg_meters['hd'].avg),
                        ('hd95', avg_meters['hd95'].avg),
                        ('recall', avg_meters['recall'].avg),
                        ('specificity', avg_meters['specificity'].avg),
                        ('precision', avg_meters['precision'].avg)])

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    seed_torch()
    config = vars(parse_args())

    exp_name = config.get('name')
    output_dir = config.get('output_dir')

    if config['name'] is None:
        base_name = config['dataset']
        
        sdi_suffix = f"_HierASFF_{config['detail_asff_channel']}_{config['semantic_asff_channel']}"
        if config['use_detail_attention']:
            sdi_suffix += f"_Att{config['attention_alpha']}"
        
        if config['deep_supervision']:
            config['name'] = f"{base_name}{sdi_suffix}_wDS"
        else:
            config['name'] = f"{base_name}{sdi_suffix}_woDS"
    
    class DummyWriter:
        def add_scalar(self, *args, **kwargs):
            pass
        
    my_writer = DummyWriter()

    os.makedirs(f'{output_dir}/{exp_name}', exist_ok=True)

    print('-' * 20)
    print('🔧 :')
    for key in config:
        print('%s: %s' % (key, config[key]))
    
    print('\n📊 SDI:')
    print(f"   : SDI ()")
    print(f"   : {config['detail_asff_channel']} (t1, t2)")
    print(f"   : {config['semantic_asff_channel']} (t3, t4)")
    print(f"   : {'' if config['use_detail_attention'] else ''}")
    if config['use_detail_attention']:
        print(f"   : {config['attention_alpha']}")
    print('-' * 20)

    with open(f'{output_dir}/{exp_name}/config.yml', 'w') as f:
        yaml.dump(config, f)

    # define loss function (criterion)
    if config['loss'] == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().cuda()
    elif config['loss'] == 'BCEDiceLoss':
        criterion = BCEDiceLoss().cuda()
    elif config['loss'] == 'LovaszHingeLoss':
        criterion = LovaszHingeLoss().cuda()
    elif config['loss'] == 'FocalDiceLoss':
        criterion = FocalDiceLoss(gamma=config['focal_gamma'], alpha=config['focal_alpha']).cuda()
    else:
        raise NotImplementedError(f"Loss {config['loss']} not implemented")

    cudnn.benchmark = True

    print("🌊 v4_waveletSDI：UKAN_SDI")
    print(f"   : {config.get('detail_dsff_channel', config.get('detail_asff_channel', 24))}")
    print(f"   : {config.get('semantic_dsff_channel', config.get('semantic_asff_channel', 36))}")
    print(f"   : {'' if config['use_detail_attention'] else ''}")

    model = UKDW(
        num_classes=config['num_classes'],
        input_channels=config['input_channels'],
        deep_supervision=config['deep_supervision'],
        img_size=config['input_h'],
        embed_dims=config['input_list'],
        no_kan=config['no_kan'],
        drop_rate=config['dropout'],
        drop_path_rate=config['drop_path'],
        enable_wavelet=config['enable_wavelet'],
        wavelet_denoise_threshold=config['wavelet_denoise_threshold'],
        detail_dsff_channel=config.get('detail_dsff_channel', config.get('detail_asff_channel', 24)),
        semantic_dsff_channel=config.get('semantic_dsff_channel', config.get('semantic_asff_channel', 36)),
        use_detail_attention=config['use_detail_attention'],
        attention_alpha=config['attention_alpha'],
        dataset_type=config['dataset']
    )

    model = model.cuda()

    param_groups = []

    kan_fc_params = []
    other_params = []

    for name, param in model.named_parameters():
        if 'layer' in name.lower() and 'fc' in name.lower(): # higher lr for kan layers
            param_groups.append({'params': param, 'lr': config['kan_lr'], 'weight_decay': config['kan_weight_decay']}) 
        else:
            param_groups.append({'params': param, 'lr': config['lr'], 'weight_decay': config['weight_decay']})  
    
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(param_groups)
    elif config['optimizer'] == 'AdamW':
        optimizer = optim.AdamW(param_groups)
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(param_groups, momentum=config['momentum'], nesterov=config['nesterov'])
    else:
        raise NotImplementedError

    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'], verbose=1, min_lr=config['min_lr'])
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')], gamma=config['gamma'])
    elif config['scheduler'] == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError

    script_path = os.path.abspath(__file__)
    shutil.copy2(script_path, f'{output_dir}/{exp_name}/')
    
    os.makedirs(f'{output_dir}/{exp_name}/models', exist_ok=True)
    os.makedirs(f'{output_dir}/{exp_name}/datasets', exist_ok=True)
    os.makedirs(f'{output_dir}/{exp_name}/utils', exist_ok=True)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_files = {
        os.path.join(current_dir, 'models/ukdw.py'): f'{output_dir}/{exp_name}/models/ukdw.py',
        os.path.join(current_dir, 'models/layers.py'): f'{output_dir}/{exp_name}/models/layers.py',
        os.path.join(current_dir, 'models/kan.py'): f'{output_dir}/{exp_name}/models/kan.py',
        os.path.join(current_dir, 'models/__init__.py'): f'{output_dir}/{exp_name}/models/__init__.py',
        os.path.join(current_dir, 'datasets/__init__.py'): f'{output_dir}/{exp_name}/datasets/__init__.py',
        os.path.join(current_dir, 'datasets/dataset.py'): f'{output_dir}/{exp_name}/datasets/dataset.py',
        os.path.join(current_dir, 'utils/__init__.py'): f'{output_dir}/{exp_name}/utils/__init__.py',
        os.path.join(current_dir, 'utils/losses.py'): f'{output_dir}/{exp_name}/utils/losses.py',
        os.path.join(current_dir, 'utils/metrics.py'): f'{output_dir}/{exp_name}/utils/metrics.py',
        os.path.join(current_dir, 'utils/misc.py'): f'{output_dir}/{exp_name}/utils/misc.py'
    }
    
    for src, dst in model_files.items():
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)

    dataset_name = config['dataset']
    img_ext = '.png'

    if dataset_name == 'busi':
        mask_ext = '_mask.png'
    elif dataset_name == 'glas':
        mask_ext = '.png'
    elif dataset_name == 'cvc':
        mask_ext = '.png'

    # Data loading code
    img_ids = sorted(glob(os.path.join(config['data_dir'], config['dataset'], 'images', '*' + img_ext)))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=config['dataseed'])

    basic_transform = [
        Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ]
    
    basic_augmentations = [
        RandomRotate90(),
        OneOf([HorizontalFlip(), VerticalFlip()], p=0.5),
    ]
    
    enhanced_augmentations = []
    if config['augmentation']:
        from albumentations import (
            RandomBrightnessContrast, GridDistortion, ElasticTransform, 
            RandomGamma, ShiftScaleRotate, Blur, MotionBlur
        )
        enhanced_augmentations = [
            RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5),
            OneOf([
                GridDistortion(p=0.5),
                ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.5)
            ], p=0.25),
            OneOf([
                Blur(blur_limit=3, p=0.5),
                MotionBlur(blur_limit=3, p=0.5),
            ], p=0.25),
            RandomGamma(gamma_limit=(80, 120), p=0.25),
        ]
    
    train_transform = Compose(basic_augmentations + enhanced_augmentations + basic_transform)
    val_transform = Compose(basic_transform)

    train_dataset = Dataset(
        img_ids=train_img_ids,
        img_dir=os.path.join(config['data_dir'], config['dataset'], 'images'),
        mask_dir=os.path.join(config['data_dir'], config['dataset'], 'masks'),
        img_ext=img_ext,
        mask_ext=mask_ext,
        num_classes=config['num_classes'],
        transform=train_transform)
    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join(config['data_dir'] ,config['dataset'], 'images'),
        mask_dir=os.path.join(config['data_dir'], config['dataset'], 'masks'),
        img_ext=img_ext,
        mask_ext=mask_ext,
        num_classes=config['num_classes'],
        transform=val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('loss', []),
        ('iou', []),
        ('val_loss', []),
        ('val_iou', []),
        ('val_dice', []),
        ('val_hd', []),
        ('val_hd95', []),
        ('val_recall', []),
        ('val_specificity', []),
        ('val_precision', []),
    ])


    best_iou = 0
    best_dice = 0
    best_hd95 = float('inf')
    best_score = 0
    trigger = 0
    
    top_models = []
    n_best_models = 5
    
    for epoch in range(config['epochs']):
        current_lr = optimizer.param_groups[0]['lr']
        print('Epoch [%d/%d], Current LR: %.7f' % (epoch, config['epochs'], current_lr))

        # train for one epoch
        train_log = train(config, train_loader, model, criterion, optimizer)
        # evaluate on validation set
        val_log = validate(config, val_loader, model, criterion)

        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()
        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_log['loss'])

        print('loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f - val_dice %.4f - val_hd95 %.4f'
              % (train_log['loss'], train_log['iou'], val_log['loss'], val_log['iou'], val_log['dice'], val_log['hd95']))

        log['epoch'].append(epoch)
        log['lr'].append(optimizer.param_groups[0]['lr'])
        log['loss'].append(train_log['loss'])
        log['iou'].append(train_log['iou'])
        log['val_loss'].append(val_log['loss'])
        log['val_iou'].append(val_log['iou'])
        log['val_dice'].append(val_log['dice'])
        log['val_hd'].append(val_log['hd'])
        log['val_hd95'].append(val_log['hd95'])
        log['val_recall'].append(val_log['recall'])
        log['val_specificity'].append(val_log['specificity'])
        log['val_precision'].append(val_log['precision'])

        pd.DataFrame(log).to_csv(f'{output_dir}/{exp_name}/log.csv', index=False)

        my_writer.add_scalar('train/loss', train_log['loss'], global_step=epoch)
        my_writer.add_scalar('train/iou', train_log['iou'], global_step=epoch)
        my_writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_step=epoch)
        my_writer.add_scalar('val/loss', val_log['loss'], global_step=epoch)
        my_writer.add_scalar('val/iou', val_log['iou'], global_step=epoch)
        my_writer.add_scalar('val/dice', val_log['dice'], global_step=epoch)
        my_writer.add_scalar('val/hd', val_log['hd'], global_step=epoch)
        my_writer.add_scalar('val/hd95', val_log['hd95'], global_step=epoch)
        my_writer.add_scalar('val/recall', val_log['recall'], global_step=epoch)
        my_writer.add_scalar('val/specificity', val_log['specificity'], global_step=epoch)
        my_writer.add_scalar('val/precision', val_log['precision'], global_step=epoch)

        my_writer.add_scalar('val/best_score', best_score, global_step=epoch)
        my_writer.add_scalar('val/best_iou_value', best_iou, global_step=epoch)
        my_writer.add_scalar('val/best_dice_value', best_dice, global_step=epoch)
        my_writer.add_scalar('val/best_hd95_value', best_hd95, global_step=epoch)

        trigger += 1

        current_score = val_log['iou']

        model_info = (current_score, epoch, val_log['iou'], val_log['dice'], val_log['hd95'])
        
        should_save = False
        
        if len(top_models) < n_best_models:
            top_models.append(model_info)
            should_save = True
        elif current_score > top_models[-1][0]:
            top_models.append(model_info)
            should_save = True
        
        if should_save:
            top_models.sort(reverse=True, key=lambda x: x[0])
            
            if len(top_models) > n_best_models:
                models_to_remove = top_models[n_best_models:]
                top_models = top_models[:n_best_models]
                
                for old_score, old_epoch, _, _, _ in models_to_remove:
                    old_files = glob_module.glob(f'{output_dir}/{exp_name}/model_top*_epoch{old_epoch}.pth')
                    for old_file in old_files:
                        try:
                            os.remove(old_file)
                            print(f"=> removed old model: {os.path.basename(old_file)}")
                        except:
                            pass
            
            temp_model_file = f'{output_dir}/{exp_name}/model_temp_epoch{epoch}.pth'
            torch.save(model.state_dict(), temp_model_file)
            
            current_rank = None
            for i, (score, ep, iou, dice, hd95) in enumerate(top_models):
                rank = i + 1
                correct_file = f'{output_dir}/{exp_name}/model_top{rank}_epoch{ep}.pth'
                
                if ep == epoch:
                    try:
                        shutil.move(temp_model_file, correct_file)
                        current_rank = rank
                    except Exception as e:
                        print(f":  {correct_file}: {str(e)}")
                        continue
                else:
                    existing_files = glob_module.glob(f'{output_dir}/{exp_name}/model_top*_epoch{ep}.pth')
                    if existing_files:
                        existing_file = existing_files[0]
                        if existing_file != correct_file:
                            try:
                                shutil.move(existing_file, correct_file)
                            except Exception as e:
                                print(f":  {existing_file} -> {correct_file}: {str(e)}")
            
            temp_files = glob_module.glob(f'{output_dir}/{exp_name}/model_temp_*.pth')
            for temp_file in temp_files:
                try:
                    os.remove(temp_file)
                except:
                    pass
            
            if current_rank:
                print(f"=> saved top-{current_rank} model at epoch {epoch} (ranked by IoU)")
                print(f'IoU Score: {current_score:.4f}, Rank: {current_rank}/{n_best_models}')
                print(f'IoU: {val_log["iou"]:.4f}, Dice: {val_log["dice"]:.4f}, HD95: {val_log["hd95"]:.4f}')
                
                print(f"Top-{min(5, len(top_models))}:")
                for i, (score, ep, iou, dice, hd95) in enumerate(top_models[:5]):
                    print(f"  Rank {i+1}: Epoch {ep} - IoU: {score:.4f}")
                if len(top_models) > 5:
                    print(f"  ...  {len(top_models)-5} ")
            
            if current_score > best_score:
                best_iou = val_log['iou']
                best_dice = val_log['dice']
                best_hd95 = val_log['hd95']
                best_score = current_score
                print("=> IoU!")
                trigger = 0

        # early stopping
        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print("=> early stopping")
            break

        torch.cuda.empty_cache()
    
    torch.save(model.state_dict(), f'{output_dir}/{exp_name}/model_final.pth')
    
    if top_models:
        top_models_info = []
        for i, (score, epoch, iou, dice, hd95) in enumerate(top_models):
            top_models_info.append({
                'rank': i + 1,
                'epoch': epoch,
                'iou_score': score,
                'iou': iou,
                'dice': dice,
                'hd95': hd95,
                'model_path': f'model_top{i+1}_epoch{epoch}.pth'
            })
        pd.DataFrame(top_models_info).to_csv(f'{output_dir}/{exp_name}/top_models_by_iou.csv', index=False)
        print(f"{len(top_models)}IoU {output_dir}/{exp_name}/top_models_by_iou.csv")

if __name__ == '__main__':
    main() 