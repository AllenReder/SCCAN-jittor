import os
import numpy as np
from PIL import Image
import random
import logging
import cv2
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from matplotlib.ticker import FuncFormatter, FormatStrFormatter
from matplotlib import font_manager
from matplotlib import rcParams
import seaborn as sns
import pandas as pd
import math
from seaborn.distributions import distplot
from tqdm import tqdm
from scipy import ndimage

# from util.get_weak_anns import find_bbox

import jittor as jt
from jittor import nn


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def step_learning_rate(optimizer, base_lr, epoch, step_epoch, multiplier=0.1):
    """Sets the learning rate to the base LR decayed by 10 every step epochs"""
    lr = base_lr * (multiplier ** (epoch // step_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def poly_learning_rate(
        optimizer, 
        base_lr, 
        curr_iter, 
        max_iter, 
        power=0.9, 
        index_split=-1, # always -1
        scale_lr=10., 
        warmup=False,
        warmup_step=500
        ):
    """poly learning rate policy"""
    if warmup and curr_iter < warmup_step:
        lr = base_lr * (0.1 + 0.9 * (curr_iter / warmup_step))
    else:
        lr = base_lr * (1 - float(curr_iter) / max_iter) ** power

    # if curr_iter % 50 == 0:   
    #     print('Base LR: {:.4f}, Curr LR: {:.4f}, Warmup: {}.'.format(base_lr, lr, (warmup and curr_iter < warmup_step)))     

    optimizer.lr = lr * scale_lr
    for index, param_group in enumerate(optimizer.param_groups):
        if index <= index_split:
            param_group['lr'] = lr
        else:
            param_group['lr'] = lr * scale_lr  # 10x LR
    
    return lr * scale_lr ### DEBUG


def intersectionAndUnion(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.ndim in [1, 2, 3])
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K + 1))
    area_output, _ = np.histogram(output, bins=np.arange(K + 1))
    area_target, _ = np.histogram(target, bins=np.arange(K + 1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def jt_histc(input, bins, min=0., max=0.):
    if min == 0 and max == 0:
        min, max = input.min(), input.max()
    assert min < max
    bin_length = (max - min) / bins
    
    # 筛选在范围内的值
    valid_input = input[jt.logical_and(input >= min, input <= max)]
    
    # 特殊处理最大值，确保它落在最后一个箱子中
    indices = jt.floor((valid_input - min) / bin_length).int()
    indices = jt.minimum(indices, jt.array(bins-1))  # 确保索引在有效范围内
    
    # 使用 jittor 的 one_hot 和 sum 来实现直方图统计
    hist = jt.zeros(bins).float()
    for i in range(bins):
        hist[i] = (indices == i).sum().float()
    
    return hist

def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.dim() in [1, 2, 3])
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    intersection = intersection.float()
    output = output.float()
    target = target.float()
    area_intersection = jt_histc(intersection, bins=K, min=0, max=K - 1)
    area_output = jt_histc(output, bins=K, min=0, max=K - 1)
    area_target = jt_histc(target, bins=K, min=0, max=K - 1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def check_makedirs(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def del_file(path):
    for i in os.listdir(path):
        path_file = os.path.join(path, i)
        if os.path.isfile(path_file):
            os.remove(path_file)
        else:
            del_file(path_file)


def init_weights(model, conv='kaiming', batchnorm='normal', linear='kaiming', lstm='kaiming'):
    """
    :param model: Jittor Model which is nn.Module
    :param conv:  'kaiming' or 'xavier'
    :param batchnorm: 'normal' or 'constant'
    :param linear: 'kaiming' or 'xavier'
    :param lstm: 'kaiming' or 'xavier'
    """
    for m in model.modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            if conv == 'kaiming':
                nn.init.kaiming_normal_(m.weight)
            elif conv == 'xavier':
                nn.init.xavier_normal_(m.weight)
            else:
                raise ValueError("init type of conv error.\n")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            if batchnorm == 'normal':
                nn.init.normal_(m.weight, 1.0, 0.02)
            elif batchnorm == 'constant':
                nn.init.constant_(m.weight, 1.0)
            else:
                raise ValueError("init type of batchnorm error.\n")
            nn.init.constant_(m.bias, 0.0)

        elif isinstance(m, nn.Linear):
            if linear == 'kaiming':
                nn.init.kaiming_normal_(m.weight)
            elif linear == 'xavier':
                nn.init.xavier_normal_(m.weight)
            else:
                raise ValueError("init type of linear error.\n")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    if lstm == 'kaiming':
                        nn.init.kaiming_normal_(param)
                    elif lstm == 'xavier':
                        nn.init.xavier_normal_(param)
                    else:
                        raise ValueError("init type of lstm error.\n")
                elif 'bias' in name:
                    nn.init.constant_(param, 0)


def colorize(gray, palette):
    # gray: numpy array of the label and 1*3N size list palette
    color = Image.fromarray(gray.astype(np.uint8)).convert('P')
    color.putpalette(palette)
    return color


# ------------------------------------------------------
def get_model_para_number(model):
    total_number = 0
    learnable_number = 0
    for param in model.parameters():
        total_number += param.numel()
        if param.requires_grad:
            learnable_number += param.numel()
    return total_number, learnable_number


def setup_seed(seed=2021, deterministic=False):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    jt.set_global_seed(seed)


def get_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def get_save_path(args):
    backbone_str = 'vgg' if args.vgg else 'resnet' + str(args.layers)
    args.snapshot_path = 'exp/{}/{}/split{}_{}shot/{}/snapshot'.format(args.data_set, args.arch, args.split, args.shot, backbone_str)
    args.result_path = 'exp/{}/{}/split{}_{}shot/{}/result'.format(args.data_set, args.arch, args.split, args.shot, backbone_str)


def get_train_val_set(args):
    if args.data_set == 'pascal':
        class_list = list(range(1, 21))  # [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
        if args.split == 3:
            sub_list = list(range(1, 16))  # [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
            sub_val_list = list(range(16, 21))  # [16,17,18,19,20]
        elif args.split == 2:
            sub_list = list(range(1, 11)) + list(range(16, 21))  # [1,2,3,4,5,6,7,8,9,10,16,17,18,19,20]
            sub_val_list = list(range(11, 16))  # [11,12,13,14,15]
        elif args.split == 1:
            sub_list = list(range(1, 6)) + list(range(11, 21))  # [1,2,3,4,5,11,12,13,14,15,16,17,18,19,20]
            sub_val_list = list(range(6, 11))  # [6,7,8,9,10]
        elif args.split == 0:
            sub_list = list(range(6, 21))  # [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
            sub_val_list = list(range(1, 6))  # [1,2,3,4,5]

    elif args.data_set == 'coco':
        if args.use_split_coco:
            print('INFO: using SPLIT COCO (FWB)')
            class_list = list(range(1, 81))
            if args.split == 3:
                sub_val_list = list(range(4, 81, 4))
                sub_list = list(set(class_list) - set(sub_val_list))
            elif args.split == 2:
                sub_val_list = list(range(3, 80, 4))
                sub_list = list(set(class_list) - set(sub_val_list))
            elif args.split == 1:
                sub_val_list = list(range(2, 79, 4))
                sub_list = list(set(class_list) - set(sub_val_list))
            elif args.split == 0:
                sub_val_list = list(range(1, 78, 4))
                sub_list = list(set(class_list) - set(sub_val_list))
        else:
            print('INFO: using COCO (PANet)')
            class_list = list(range(1, 81))
            if args.split == 3:
                sub_list = list(range(1, 61))
                sub_val_list = list(range(61, 81))
            elif args.split == 2:
                sub_list = list(range(1, 41)) + list(range(61, 81))
                sub_val_list = list(range(41, 61))
            elif args.split == 1:
                sub_list = list(range(1, 21)) + list(range(41, 81))
                sub_val_list = list(range(21, 41))
            elif args.split == 0:
                sub_list = list(range(21, 81))
                sub_val_list = list(range(1, 21))

    return sub_list, sub_val_list


def is_same_model(model1, model2):
    flag = 0
    count = 0
    for k, v in model1.state_dict().items():
        model1_val = v
        model2_val = model2.state_dict()[k]
        if (model1_val == model2_val).all():
            pass
        else:
            flag += 1
            print('value of key <{}> mismatch'.format(k))
        count += 1

    return True if flag == 0 else False


def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


def sum_list(list):
    sum = 0
    for item in list:
        sum += item
    return sum
