import torch
import torch.nn as nn
import torch.utils as utils
import torch.nn.init as init
import torch.utils.data as data
import torchvision.utils as v_utils
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable


def conv_block(in_dim, out_dim, act_fn):
    model = nn.Sequential(
        nn.Conv2d(in_dim,out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_dim),
        act_fn,
    )
    return model


def conv_trans_block(in_dim,out_dim,act_fn):
    model = nn.Sequential(
        nn.ConvTranspose2d(in_dim,out_dim, kernel_size=3, stride=2, padding=1,output_padding=1),
        nn.BatchNorm2d(out_dim),
        act_fn,
    )
    return model


def maxpool():
    pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    return pool


def conv_block_3(in_dim,out_dim,act_fn):
    model = nn.Sequential(
        conv_block(in_dim, out_dim, act_fn),
        conv_block(out_dim, out_dim, act_fn),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_dim),
    )
    return model

def quaternion_to_matrix(q):
        B = q.shape[0]
        qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        R = torch.zeros((B, 3, 3), device=q.device)
        R[:, 0, 0] = 1 - 2 * (qy ** 2 + qz ** 2)
        R[:, 0, 1] = 2 * (qx * qy - qz * qw)
        R[:, 0, 2] = 2 * (qx * qz + qy * qw)
        R[:, 1, 0] = 2 * (qx * qy + qz * qw)
        R[:, 1, 1] = 1 - 2 * (qx ** 2 + qz ** 2)
        R[:, 1, 2] = 2 * (qy * qz - qx * qw)
        R[:, 2, 0] = 2 * (qx * qz - qy * qw)
        R[:, 2, 1] = 2 * (qy * qz + qx * qw)
        R[:, 2, 2] = 1 - 2 * (qx ** 2 + qy ** 2)
        return R