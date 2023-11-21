# -*- coding: utf-8 -*-
# @Time   : 2020/7/17
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

# UPDATE
# @Time   : 2021/3/8
# @Author : Jiawei Guan
# @Email  : guanjw@ruc.edu.cn

"""
recbole.utils.utils
################################
"""

import datetime
import importlib
import os
import random
from pytz import timezone

import numpy as np
import torch
import scipy.sparse as sp


def normalize_sp(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten() # .flatten convert the array into 1-dim
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def normalize_dense(mx):
    """Row-normalize dense matrix"""
    # adj 行正则 x/1e-12=0 因为1e-12对应的那些x都是0
    mx = mx / np.clip(np.sum(mx, axis=-1, keepdims=True), a_min=1e-12, a_max=None) # VERY_SMALL_NUMBER =1e-12
    return mx

def random_neq(l, r, pos_set):
    # 随机选择一个不在交互序列中的作为负样本
    t = np.random.randint(l, r)
    while t in pos_set:
        t = np.random.randint(l, r)
    return t

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total parameter': total_num, 'Trainable parameter': trainable_num}

def get_local_time():
    r"""Get current time

    Returns:
        str: current time
    """
    cur = datetime.datetime.now(timezone('Asia/Shanghai'))
    cur = cur.strftime('%b-%d-%Hh%Mm%Ss')

    return cur


def mkdir_ifnotexist(dir_path):
    r"""Make sure the directory exists, if it does not exist, create it

    Args:
        dir_path (str): directory path

    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def early_stopping(value, best, cur_step, max_step, bigger=False):
    r""" validation-based early stopping

    Args:
        value (float): current result
        best (float): best result
        cur_step (int): the number of consecutive steps that did not exceed the best result
        max_step (int): threshold steps for stopping
        bigger (bool, optional): whether the bigger the better

    Returns:
        tuple:
        - float,
          best result after this step
        - int,
          the number of consecutive steps that did not exceed the best result after this step
        - bool,
          whether to stop
        - bool,
          whether to update
    """
    stop_flag = False
    update_flag = False
    if bigger:
        if value > best:  # 如果想让指标越大越好 则>  并且best初始化为很小的值     如果想让指标越小越好的话 则<    并且best初始化为很大
            # 的值
            cur_step = 0
            best = value
            update_flag = True
        else:
            cur_step += 1
            if cur_step > max_step:
                stop_flag = True
    else:
        if value < best:
            cur_step = 0
            best = value
            update_flag = True
        else:
            cur_step += 1
            if cur_step > max_step:
                stop_flag = True
    return best, cur_step, stop_flag, update_flag


def calculate_valid_score(valid_result, valid_metric=None):
    r""" return valid score from valid result

    Args:
        valid_result (dict): valid result
        valid_metric (str, optional): the selected metric in valid result for valid score

    Returns:
        float: valid score
    """
    if valid_metric:
        return valid_result[valid_metric]
    else:
        return valid_result['Recall@10']


def dict2str(result_dict):
    r""" convert result dict to str

    Args:
        result_dict (dict): result dict

    Returns:
        str: result str
    """

    result_str = ''
    for metric, value in result_dict.items():
        result_str += str(metric) + ' : ' + str(value) + '    '
    return result_str


def init_seed(seed, reproducibility):
    r""" init random seed for random functions in numpy, torch, cuda and cudnn

    Args:
        seed (int): random seed
        reproducibility (bool): Whether to require reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if reproducibility:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False




def get_gpu_usage(device=None):
    r""" Return the reserved memory and total memory of given device in a string.
    Args:
        device: cuda.device. It is the device that the model run on.

    Returns:
        str: it contains the info about reserved memory and total memory of given device.
    """

    reserved = torch.cuda.max_memory_reserved(device) / 1024 ** 3
    total = torch.cuda.get_device_properties(device).total_memory / 1024 ** 3

    return '{:.2f} G/{:.2f} G'.format(reserved, total)
