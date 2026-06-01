#!/usr/bin/env python3
"""
U-KAN SDI v3 
SDI
"""

import argparse
import os
from glob import glob
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import yaml
from albumentations import transforms, Normalize
from albumentations.core.composition import Compose
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from albumentations import Resize
import warnings
warnings.filterwarnings('ignore')

from models import UKDW
from models import UKAN_ASFF
from datasets import Dataset
from utils.metrics import iou_score, indicators
from utils.misc import AverageMeter, str2bool

def list_type(s):
    str_list = s.split(',')
    int_list = [int(a) for a in str_list]
    return int_list

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None, help='model name')
    parser.add_argument('--model_path', default=None, help='path to trained model')
    parser.add_argument('-b', '--batch_size', default=1, type=int, metavar='N', help='mini-batch size (default: 16)')

    parser.add_argument('--deep_supervision', default=False, type=str2bool)
    parser.add_argument('--input_channels', default=3, type=int, help='input channels')
    parser.add_argument('--num_classes', default=1, type=int, help='number of classes')
    parser.add_argument('--input_w', default=256, type=int, help='image width')
    parser.add_argument('--input_h', default=256, type=int, help='image height')
    parser.add_argument('--input_list', type=list_type, default=[256, 320, 512])

    parser.add_argument('--dataset', default='busi', help='dataset name')
    parser.add_argument('--data_dir', default='inputs', help='dataset dir')
    parser.add_argument('--output_dir', default='outputs', help='ouput dir')
    parser.add_argument('--cfg', type=str, metavar="FILE", help='path to config file')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--no_kan', action='store_true')

    parser.add_argument('--sdi_channel', default=32, type=int, help='SDI module channel dimension')

    parser.add_argument('--use_hierarchical_sdi', default=True, type=str2bool)
    parser.add_argument('--detail_asff_channel', default=24, type=int)
    parser.add_argument('--semantic_asff_channel', default=32, type=int)

    parser.add_argument('--use_detail_attention', default=False, type=str2bool)
    parser.add_argument('--attention_alpha', default=0.1, type=float)

    parser.add_argument('--enable_wavelet', default=True, type=str2bool, help='Enable wavelet preprocessing')
    parser.add_argument('--wavelet_denoise_threshold', default=0.08, type=float, help='Wavelet denoising threshold')

    config = parser.parse_args()
    return config

def main():
    config = parse_args()

    if config.model_path is not None:
        model_path = config.model_path
    else:
        model_path = os.path.join(config.output_dir, config.name, 'model.pth')

    if config.name is not None:
        config_path = os.path.join(config.output_dir, config.name, 'config.yml')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_dict = yaml.load(f, Loader=yaml.FullLoader)
                for key in config_dict.keys():
                    if key in vars(config) and getattr(config, key) is None:
                        setattr(config, key, config_dict[key])

    config_dict = vars(config)

    print('-'*20)
    for key in config_dict.keys():
        print('%s: %s' % (key, config_dict[key]))
    print('-'*20)

    cudnn.benchmark = True

    model = UKDW(
        num_classes=config.num_classes,
        input_channels=config.input_channels,
        deep_supervision=config.deep_supervision,
        img_size=config.input_h,
        embed_dims=config.input_list,
        no_kan=config.no_kan,
        detail_dsff_channel=getattr(config, 'detail_dsff_channel', getattr(config, 'detail_asff_channel', 24)),
        semantic_dsff_channel=getattr(config, 'semantic_dsff_channel', getattr(config, 'semantic_asff_channel', 36)),
        use_detail_attention=config.use_detail_attention,
        attention_alpha=config.attention_alpha,
        dataset_type=config.dataset,
        enable_wavelet=config.enable_wavelet,
        wavelet_denoise_threshold=config.wavelet_denoise_threshold
    )

    model = model.cuda()

    print(f"Loading model from {model_path}")
    model.load_state_dict(torch.load(model_path), strict=False)
    model.eval()

    dataset_name = config.dataset
    img_ext = '.png'

    if dataset_name == 'busi':
        mask_ext = '_mask.png'
    elif dataset_name == 'glas':
        mask_ext = '.png'
    elif dataset_name == 'cvc':
        mask_ext = '.png'

    # Data loading code
    img_ids = sorted(glob(os.path.join(config.data_dir, config.dataset, 'images', '*' + img_ext)))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    _, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=2981)

    val_transform = Compose([
        Resize(config.input_h, config.input_w),
        Normalize(),
    ])

    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join(config.data_dir, config.dataset, 'images'),
        mask_dir=os.path.join(config.data_dir, config.dataset, 'masks'),
        img_ext=img_ext,
        mask_ext=mask_ext,
        num_classes=config.num_classes,
        transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        drop_last=False)

    avg_meters = {'iou': AverageMeter(),
                  'dice': AverageMeter(),
                  'hd': AverageMeter(),
                  'hd95': AverageMeter(),
                  'recall': AverageMeter(),
                  'specificity': AverageMeter(),
                  'precision': AverageMeter()}

    with torch.no_grad():
        for input, target, meta in tqdm(val_loader, total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()

            if config.deep_supervision:
                output = model(input)[-1]
            else:
                output = model(input)

            iou = iou_score(output, target)
            iou_, dice_, hd_, hd95_, recall_, specificity_, precision_ = indicators(output, target)

            avg_meters['iou'].update(iou_, input.size(0))
            avg_meters['dice'].update(dice_, input.size(0))
            avg_meters['hd'].update(hd_, input.size(0))
            avg_meters['hd95'].update(hd95_, input.size(0))
            avg_meters['recall'].update(recall_, input.size(0))
            avg_meters['specificity'].update(specificity_, input.size(0))
            avg_meters['precision'].update(precision_, input.size(0))

    print('IoU: %.4f' % avg_meters['iou'].avg)
    print('Dice: %.4f' % avg_meters['dice'].avg)
    print('HD: %.4f' % avg_meters['hd'].avg)
    print('HD95: %.4f' % avg_meters['hd95'].avg)
    print('Recall: %.4f' % avg_meters['recall'].avg)
    print('Specificity: %.4f' % avg_meters['specificity'].avg)
    print('Precision: %.4f' % avg_meters['precision'].avg)

    output_file = os.path.join(config.output_dir, f"{config.dataset}_evaluation_results.txt")
    os.makedirs(config.output_dir, exist_ok=True)

    with open(output_file, 'w') as f:
        f.write(f"Model: {model_path}\n")
        f.write(f"Dataset: {config.dataset}\n")
        f.write(f"Input size: {config.input_h}x{config.input_w}\n")
        f.write(f"IoU: {avg_meters['iou'].avg:.4f}\n")
        f.write(f"Dice: {avg_meters['dice'].avg:.4f}\n")
        f.write(f"HD: {avg_meters['hd'].avg:.4f}\n")
        f.write(f"HD95: {avg_meters['hd95'].avg:.4f}\n")
        f.write(f"Recall: {avg_meters['recall'].avg:.4f}\n")
        f.write(f"Specificity: {avg_meters['specificity'].avg:.4f}\n")
        f.write(f"Precision: {avg_meters['precision'].avg:.4f}\n")

    print(f"Results saved to: {output_file}")

if __name__ == '__main__':
    main()
