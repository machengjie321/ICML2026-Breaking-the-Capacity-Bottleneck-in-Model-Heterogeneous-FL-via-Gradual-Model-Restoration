from typing import Union, Generator
from copy import deepcopy
import random
import pynvml
import numpy as np
import torch
import time
def disp_num_params(model):
    total_param_in_use = 0
    total_all_param = 0
    for layer, layer_prefx in zip(model.prunable_layers, model.prunable_layer_prefixes):
        layer_param_in_use = layer.num_weight
        layer_all_param = layer.mask.nelement()
        total_param_in_use += layer_param_in_use
        total_all_param += layer_all_param
        # print("{} remaining: {}/{} = {}".format(layer_prefx, layer_param_in_use, layer_all_param,
        #                                         layer_param_in_use / layer_all_param))
    print("Total: {}/{} = {}".format(total_param_in_use, total_all_param, total_param_in_use / total_all_param))

    return total_param_in_use / total_all_param, total_param_in_use

import pynvml
import subprocess

import subprocess


import subprocess
import re

import subprocess

import subprocess
import re

def get_gpu_to_cifar_map():
    """
    1. 识别 nvidia-smi 中正在运行的 python 进程及其 GPU ID。
    2. 通过 ps 检查启动参数是否包含 'CIFAR'。
    3. 返回一个字典：{gpu_id: 进程数量}
    """
    # 存储格式：{gpu_index: 进程计数}
    gpu_with_cifar_counts = {}
    
    try:
        # 获取 nvidia-smi 原始输出
        smi_output = subprocess.check_output(["nvidia-smi"], encoding='utf-8')
        
        # 适配你提供的带有 N/A 的表格格式
        pattern = re.compile(r"\|\s+(\d+)\s+N/A\s+N/A\s+(\d+)\s+[CG]\s+python")
        
        for line in smi_output.split('\n'):
            match = pattern.search(line)
            if match:
                gpu_idx = int(match.group(1))
                pid = match.group(2)
                
                try:
                    # 使用 ps 查看完整命令行
                    ps_cmd = f"ps -p {pid} -f --no-headers"
                    cmd_detail = subprocess.check_output(ps_cmd, shell=True, encoding='utf-8').strip()
                    
                    if "CIFAR" in cmd_detail.upper():
                        # 如果匹配成功，计数加 1
                        gpu_with_cifar_counts[gpu_idx] = gpu_with_cifar_counts.get(gpu_idx, 0) + 1
                        print(f"Verified: GPU {gpu_idx} 发现 CIFAR 进程 (PID: {pid})")
                except:
                    continue
                    
    except Exception as e:
        print(f"GPU 进程检测出错: {e}")
    
    return gpu_with_cifar_counts


    
def get_gpu_samples(duration=25, interval=0.5):
    """
    返回 duration 时间内的 GPU 采样
    每 interval 秒采一次
    """
    pynvml.nvmlInit()
    num_gpus = pynvml.nvmlDeviceGetCount()

    sample_count = int(duration / interval)

    # 初始化采样结构
    gpu_samples = {i: {"mem": [], "util": []} for i in range(num_gpus)}

    for _ in range(sample_count):
        for i in range(num_gpus):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)

            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            free_mem_mb = mem.free // 1024 ** 2

            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_util = util.gpu  # 0–100%

            gpu_samples[i]["mem"].append(free_mem_mb)
            gpu_samples[i]["util"].append(gpu_util)

        time.sleep(interval)

    pynvml.nvmlShutdown()
    return gpu_samples

def select_best_gpu(min_memory=11 * 1024, duration=25, interval=0.5,
                    alpha=0.99, beta=0.1):
    
    # 获取运行了 cifar 的 GPU 映射
    cifar_counts = get_gpu_to_cifar_map()
    
    print(f"\nSampling GPU status for {duration}s ...")
    samples = get_gpu_samples(duration, interval)

    gpu_stats = []
    for gpu_id, data in samples.items():
        avg_mem = sum(data["mem"]) / len(data["mem"])
        avg_util = sum(data["util"]) / len(data["util"])
        
        # 2. 根据进程数量扣除显存
        num_processes = cifar_counts.get(gpu_id, 0)
        if num_processes > 0:
            penalty = num_processes * 2 * 1024  # 每个进程扣 8GB
            print(f"--- GPU {gpu_id}: 检测到 {num_processes} 个 CIFAR 进程，扣除 {num_processes * 2}GB 显存 ---")
            avg_mem = max(0, avg_mem - penalty)
        
        gpu_stats.append((gpu_id, avg_mem, avg_util))


    # 归一化计算
    max_free_mem = max(m for _, m, _ in gpu_stats) + 1e-6
    max_util = max(u for _, _, u in gpu_stats) + 1e-6

    print("\n---- Adjusted GPU Averages ----")
    for i, mem, util in gpu_stats:
        print(f"GPU {i}: Free Mem: {mem:.1f} MB  Util: {util:.1f}%")
    
    best_gpu = None
    best_score = -1

    for i, avg_mem, avg_util in gpu_stats:
        if avg_mem < min_memory:
            continue

        mem_norm = avg_mem / max_free_mem
        util_norm = avg_util / max_util
        score = alpha * mem_norm + beta * (1 - util_norm)

        if score > best_score:
            best_score = score
            best_gpu = i

    if best_gpu is None:
        raise RuntimeError("No GPU meets the criteria after precise penalty.")

    print(f"\nFinal Selected GPU: {best_gpu} (Score: {best_score:.4f})")
    return best_gpu

def copy_dict(ori_dict: Union[dict, Generator]):
    generator = ori_dict.items() if isinstance(ori_dict, dict) else ori_dict
    copied_dict = dict()
    for key, param in generator:
        copied_dict[key] = param
    return copied_dict


def deepcopy_dict(ori_dict: Union[dict, Generator]):
    generator = ori_dict.items() if isinstance(ori_dict, dict) else ori_dict
    deepcopied_dict = dict()
    for key, param in generator:
        deepcopied_dict[key] = param.clone()
    return deepcopied_dict


def copy_shuffle_list(inp_list):
    copy_list = deepcopy(inp_list)
    random.shuffle(copy_list)
    return copy_list


def dirichlet_split_noniid(train_labels, alpha, n_clients):
    '''
    按照参数为alpha的Dirichlet分布将样本索引集合划分为n_clients个子集
    '''

    n_classes = train_labels.max()+1
    # (K, N) 类别标签分布矩阵X，记录每个类别划分到每个client去的比例
    label_distribution = np.random.dirichlet([alpha]*n_clients, n_classes)
    # (K, ...) 记录K个类别对应的样本索引集合
    class_idcs = [np.argwhere(train_labels == y).flatten()
                  for y in range(n_classes)]

    # 记录N个client分别对应的样本索引集合
    client_idcs = [[] for _ in range(n_clients)]
    for k_idcs, fracs in zip(class_idcs, label_distribution):
        # np.split按照比例fracs将类别为k的样本索引k_idcs划分为了N个子集
        # i表示第i个client，idcs表示其对应的样本索引集合idcs
        for i, idcs in enumerate(np.split(k_idcs,
                                          (np.cumsum(fracs)[:-1]*len(k_idcs)).
                                          astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]
    client_idcs = sorted(client_idcs, key=lambda x: (len(x)), reverse=False)

    return client_idcs

def compute_same_params_ratio(model1, model2, atol=1e-5):
    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()

    same_elements = 0
    total_elements = 0

    for key in state_dict1:
        param1 = state_dict1[key]
        param2 = state_dict2[key]

        if param1.shape != param2.shape:
            raise ValueError(f"参数 {key} 的形状不匹配")

        # 将参数拉平后进行比较
        total_elements += param1.numel()
        # 对于浮点数，建议使用 isclose 来比较（考虑数值精度）
        same = torch.isclose(param1.view(-1), param2.view(-1), atol=atol)
        same_elements += same.sum().item()

    ratio = same_elements / total_elements
    return ratio