import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn


def ceil_modulo(x, mod):
    if x % mod == 0:
        return x
    return (x // mod + 1) * mod


def pad_img_to_modulo(img, mod):
    _, height, width = img.shape
    out_height = ceil_modulo(height, mod)
    out_width = ceil_modulo(width, mod)
    return np.pad(img, ((0, 0), (0, out_height - height), (0, out_width - width)), mode='symmetric')


def preprocess_image(img, mod=8):
    if img.ndim == 3:
        img = np.transpose(img, (2, 0, 1))
    if img.ndim == 2: # mask (h, w)
        img = img[None, :, :] # (1, h, w)
    out_img = img.astype('float32') / 255
    out_img = pad_img_to_modulo(out_img, mod)
    return out_img


def to_tensor(img):
    # img: (c, h, w) 0-1
    img_tensor = torch.from_numpy(img).float()
    img_tensor = img_tensor.unsqueeze(0) # (1, c, h, w)
    return img_tensor


def pad_tensor_to_modulo(img, mod):
    batch_size, channels, height, width = img.shape
    out_height = ceil_modulo(height, mod)
    out_width = ceil_modulo(width, mod)
    return F.pad(img, pad=(0, out_width - width, 0, out_height - height), mode='reflect')

def move_to_device(obj, device):
    if isinstance(obj, nn.Module):
        return obj.to(device)
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, (tuple, list)):
        return [move_to_device(el, device) for el in obj]
    if isinstance(obj, dict):
        return {name: move_to_device(val, device) for name, val in obj.items()}
    raise ValueError(f'Unexpected type {type(obj)}')