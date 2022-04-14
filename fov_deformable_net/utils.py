import os
import random
import time
import math
import torch
import torchvision.transforms as transforms
import numpy as np


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def RandomRot(img, angle=90, p=0.5):
    if random.random() > p:
        return transforms.functional.rotate(img, angle)
    return img

def step_lr_adjust(optimizer, epoch, init_lr=1e-4, step_size=20, gamma=0.1):
    lr = init_lr * gamma ** (epoch // step_size)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def cycle_lr_adjust(optimizer, epoch, base_lr=1e-5, max_lr=1e-4, step_size=10, gamma=1):
    cycle = np.floor(1 + epoch/(2  * step_size))
    x = np.abs(epoch/step_size - 2 * cycle + 1)
    scale =  gamma ** (epoch // (2 * step_size))
    lr = base_lr + (max_lr - base_lr) * np.maximum(0, (1-x)) * scale
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def compare_psnr(img1, img2):
    img1 = np.float64(img1)
    img2 = np.float64(img2)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100

    PIXEL_MAX = 1.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

# using ImageNet values
def normalize_tensor_transform(output, label):
    output_norm = torch.zeros_like(output)
    output_norm[:, 0, ...] = (output[:, 0, ...] - 0.485) / 0.229
    output_norm[:, 1, ...] = (output[:, 1, ...] - 0.456) / 0.224
    output_norm[:, 2, ...] = (output[:, 2, ...] - 0.406) / 0.225

    label_norm = torch.zeros_like(label)
    label_norm[:, 0, ...] = (label[:, 0, ...] - 0.485) / 0.229
    label_norm[:, 1, ...] = (label[:, 1, ...] - 0.456) / 0.224
    label_norm[:, 2, ...] = (label[:, 2, ...] - 0.406) / 0.225

    return output_norm, label_norm

def process(img, ccm):
    # apply gamma
    img_out = torch.pow((img+1e-8), 0.454)
    # apply ccm
    img_out = torch.einsum('ikjl, mk -> imjl', [img_out, ccm])
    return img_out

def apply_cmatrix(img, ccm):
    if not img.shape[1] == 3:
        raise ValueError('Incorrect channel dimension!')
    
    img_out = torch.zeros_like(img)
    img_out[:, 0, :, :] = ccm[0, 0] * img[:, 0, :, :] + ccm[0, 1] * img_out[:, 1, :, :] + ccm[0, 2] * img_out[:, 2, :, :]
    img_out[:, 1, :, :] = ccm[1, 0] * img[:, 0, :, :] + ccm[1, 1] * img_out[:, 1, :, :] + ccm[1, 2] * img_out[:, 2, :, :]
    img_out[:, 2, :, :] = ccm[2, 0] * img[:, 0, :, :] + ccm[2, 1] * img_out[:, 1, :, :] + ccm[2, 2] * img_out[:, 2, :, :]
    return img_out