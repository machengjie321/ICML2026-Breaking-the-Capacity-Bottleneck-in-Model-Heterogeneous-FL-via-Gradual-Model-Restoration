import argparse
import os
from copy import deepcopy
import torch
from utils.save_load import mkdir_save
from utils.functional import disp_num_params
from timeit import default_timer as timer
from utils.functional import deepcopy_dict
from collections import OrderedDict
from abc import ABC, abstractmethod
import copy
import numpy as np
import logging
import os
import math
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-ex',
                        help='experiment name',
                        action='store',
                        default='ex',
                        type=str)

    parser.add_argument('-ic', '--increment',
                        help="increase",
                        action='store',
                        dest='increase',
                        type=float,
                        default=0.2,
                        required=False)

    parser.add_argument('-r', '--resume',
                        help="Resume previous prototype",
                        action='store_true',
                        dest='resume',
                        default=False,
                        required=False)

    parser.add_argument('-re', '--recover',
                        help="use recover",
                        action='store_true',
                        dest='recover',
                        default=False,
                        required=False)

    parser.add_argument('-niid',
                        help="non--iid",
                        action='store_true',
                        dest='sample',

                        )

    parser.add_argument('-Res',
                        help="Residual",
                        action='store_true',
                        dest='Residual',

                        )

    parser.add_argument('-i', '--interval',
                        help="interval_round",
                        action='store',
                        dest='interval',
                        type=int,
                        default=50,
                        required=False)

    parser.add_argument('-g', '--gpu',
                        help="gpu_device",
                        action='store',
                        dest='device',
                        type=str,
                        default=0,
                        required=False)

    parser.add_argument('-m', '--merge',
                        help="merge_model",
                        action='store',
                        dest='merge',
                        type=str,
                        default='buffmaskfedavg',
                        required=False)

    parser.add_argument('-md',
                        help="min density",
                        action='store',
                        dest='min_density',
                        type=float,
                        default=0.1,
                        required=False
                        )

    parser.add_argument('-ac',
                        help="accumulate weight",
                        action='store',
                        dest='accumulate',
                        type=str,
                        default='wg',
                        required=False
                        )

    parser.add_argument('-ch', '--chronous',
                        help="Asyn or Syn",
                        action='store',
                        dest='chronous',
                        type=str,
                        default='asyn',
                        required=False)

    parser.add_argument('-num_clients',
                        help="num_client",
                        action='store',
                        dest='num_clients',
                        type=int,
                        default=10,
                        required=False)

    parser.add_argument('-sample_data_degree',
                        help="sample_data_degree",
                        action='store',
                        dest='sample_data_degree',
                        type=float,
                        default=0.6,
                        required=False)

    parser.add_argument('-sample_client',
                        help="sample_client, high, medium, low",
                        action='store',
                        dest='sample_client',
                        type=str,
                        default='medium',
                        required=False)
    parser.add_argument('-patience',
                        help="patience",
                        action='store',
                        dest='patience',
                        type=int,
                        default=10,
                        required=False
                        )
    parser.add_argument('--wait_stable_mult',
                        help="server wait-for-stable multiplier (patience * multiplier)",
                        action='store',
                        dest='wait_stable_mult',
                        type=int,
                        default=1,
                        required=False
                        )
    parser.add_argument('-server_up',
                        help="server upload speed",
                        action='store',
                        dest='server_up_speed',
                        type=float,
                        default=10,
                        required=False
                        )

    parser.add_argument('-bern',
                        help="fiarse bern ",
                        dest='bern',
                        action='store_true',
                        default=False,
                        required=False
                        )


    return parser.parse_args()



class FjordServer(ABC):

    def __init__(self,  config, args, model, save_interval=50):
                # === Basic settings ===
        self.config = config
        self.args = args
        self.seed = args.seed
        self.experiment_name = args.experiment_name
        self.recover = args.recover
        self.interval = args.interval
        self.save_path = os.path.join("results", config.EXP_NAME, args.experiment_name)
        self.save_interval = 50
        self.use_adaptive = True
        self.client_selection = args.client_selection
        self.client_list = None
        
        self.client_density = self.args.client_density
        self.actual_client_density = deepcopy(self.client_density)
        self.first_layer_channel_name = None
        self.first_layer_channel_bounds = []
        self.list_mask = None


        # === Device configuration ===
        self.device = torch.device("cuda:" + str(args.device) if torch.cuda.is_available() else "cpu")

        # === Model & client count ===
        self.model = model.to(self.device)
        self.number_clients = args.number_clients
        
        self.config = config
        self.experiment_name = args.experiment_name
        self.save_path = os.path.join("results", config.EXP_NAME, args.experiment_name)
        self.save_interval = save_interval
        self.client_selection = args.client_selection

        self.model = model.to(self.device)
        self.model.train()
        mkdir_save(self.model, os.path.join(self.save_path, "init_model.pt"))

        self.indices = None

        self.ip_train_loader = None
        self.ip_test_loader = None
        self.ip_optimizer_wrapper = None
        self.ip_control = None

        self.test_loader = None
        self.control = None



        self.init_test_loader()
        # self.init_clients()
        self.init_control()
        self.init_ip_config()
        self.save_exp_config()
        self.list_mask = self.roll_split_universal(0,self.model, self.client_density)
        log_file_path = os.path.join(self.save_path, 'app.log')
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file_path,mode='w'),  # Log to file
                # logging.StreamHandler()  # (Optional) Log to console
            ]
        )

        # Create logger instance
        self.logger = logging.getLogger(__name__)
        self.log_first_layer_channel_bounds()
        


    @abstractmethod
    def init_clients(self):
        pass

    @abstractmethod
    def init_control(self, *args, **kwargs):
        pass



    @abstractmethod
    def init_ip_config(self, *args, **kwargs):
        pass

    def log_first_layer_channel_bounds(self):
        if self.logger is None or not self.first_layer_channel_bounds:
            return

        bounds_text = []
        for client_id, bounds in enumerate(self.first_layer_channel_bounds):
            if bounds is None:
                continue
            start_idx, end_idx = bounds
            bounds_text.append(f"c{client_id}:{start_idx}->{end_idx}")

        if not bounds_text:
            return

        layer_name = self.first_layer_channel_name or "unknown"
        message = f"first layer channels ({layer_name}) " + ", ".join(bounds_text)
        self.logger.info(message)
        print(message)
    
    @torch.no_grad()
    def roll_split_universal(self, round_n, model, density, Roll=False):
        state_dict = model.state_dict()
        output_prefix = model.output_layer_prefix
        num_clients = len(density)
        idx = [OrderedDict() for _ in range(num_clients)]
        idx_i = [None for _ in range(num_clients)]
        list_mask = []
        actual_client_density = [0 for _ in range(num_clients)]
        first_layer_name = None
        first_layer_channel_bounds = [None for _ in range(num_clients)]
        
        block_roll_idx = dict()  # block-wise roll index 缓存
        if self.config.EXP_NAME == "stackoverflow":
            BASE_EMB = self.config.BASE_EMB      # 例如 256
            client_scaler_rate  = {}
            client_local_dim    = {}
            idx_i = {}  # 当前通道索引

            for client_id in range(num_clients):
                density_i = density[client_id]

                # 1) sqrt(density) + 量化
                base_scaler  = density_i ** 0.5
                scaler_steps = int(round(base_scaler * 32))
                scaler_steps = max(1, min(32, scaler_steps))
                scaler_rate  = scaler_steps / 32.0

                # 2) 直接用 BASE_EMB * scaler_rate 得到 local_dim
                raw_dim   = math.ceil(BASE_EMB * scaler_rate)
                local_dim = max(1, raw_dim)

                client_scaler_rate[client_id] = scaler_rate
                client_local_dim[client_id]   = local_dim




        for client_id in range(num_clients):
            masks = OrderedDict()
            
            for name, param in state_dict.items():
                if self.config.EXP_NAME == "ImageNet100":
                    if 'weight' in name:
                        if param.dim() > 1:
                            out_channels = param.size(0)
                            in_channels = param.size(1)


                            # conv 层主路径
                            if 'conv1' in name or 'conv2' in name:
                                if idx_i[client_id] is None:
                                    idx_i[client_id] = torch.arange(in_channels, device=param.device)
                                    
                                input_idx_i_m = idx_i[client_id]

                                local_output_size = int(int(np.ceil(density[client_id]**0.5 * out_channels))/4)*4


                                if Roll:
                                    base_idx = torch.arange(out_channels, device=param.device)
                                    roll = round_n % out_channels
                                    model_idx = torch.roll(base_idx, shifts=roll, dims=-1)
                                else:
                                    model_idx = torch.arange(out_channels, device=param.device)

                                output_idx_i_m = model_idx[:local_output_size]
                                if first_layer_channel_bounds[client_id] is None and output_idx_i_m.numel() > 0:
                                    if first_layer_name is None:
                                        first_layer_name = name
                                    first_layer_channel_bounds[client_id] = (
                                        int(output_idx_i_m[0].item()),
                                        int(output_idx_i_m[-1].item()),
                                    )
                                idx_i[client_id] = output_idx_i_m

                            # shortcut / downsample.0
                            elif '.downsample.0' in name:
                                conv_key = name.replace('.downsample.0', '.conv1')
                                input_idx_i_m = idx[client_id][conv_key][1]
                                output_idx_i_m = idx_i[client_id]  # 与主路径 conv2 相同

                            # 最后 linear 层
                            elif name.startswith(output_prefix):
                                input_idx_i_m = idx_i[client_id]
                                output_idx_i_m = torch.arange(out_channels, device=param.device)

                            else:
                                print(name)
                                raise ValueError('Not valid conv-like layer')

                            idx[client_id][name] = (output_idx_i_m, input_idx_i_m)

                        else:
                            # 1D 参数（如 norm 的 weight）
                            input_idx_i_m = idx_i[client_id]
                            idx[client_id][name] = input_idx_i_m

                    elif 'bias' in name:
                        in_channels = param.size(0)
                        if name.startswith(output_prefix):
                            input_idx_i_m = torch.arange(in_channels, device=param.device)
                        else:
                            input_idx_i_m = idx_i[client_id]
                        idx[client_id][name] = input_idx_i_m

                    else:
                        pass
                elif self.config.EXP_NAME == "stackoverflow":
                    if 'weight' in name or 'bias' in name:

                        # =====================
                        # 1) 处理 weight
                        # =====================
                        if 'weight' in name:
                            if param.dim() > 1:
                                input_size  = param.size(1)
                                output_size = param.size(0)

                                for client_id in range(num_clients):
                                    scaler_rate = client_scaler_rate[client_id]
                                    local_dim   = client_local_dim[client_id]

                                    # ---- 1) Embedding：不裁剪 vocab 维，只裁剪 embedding 维 ----
                                    if 'embedding' in name.split('.')[-2]:
                                        # 全局: [num_tokens, BASE_EMB]
                                        input_size  = param.size(1)  # BASE_EMB
                                        output_size = param.size(0)  # num_tokens

                                        # vocab 全保留
                                        output_idx_i_m = torch.arange(output_size, device=param.device)

                                        # embedding 维只保留 local_dim
                                        assert input_size >= local_dim, \
                                            f"embedding: input_size={input_size} < local_dim={local_dim}"

                                        ridx = torch.arange(input_size, device=param.device)
                                        if Roll:
                                            roll = round_n % input_size
                                            ridx = torch.roll(ridx, roll, -1)
                                        input_idx_i_m = ridx[:local_dim]

                                        # 当前通道 = embedding 输出维
                                        idx_i[client_id] = input_idx_i_m
                                        idx[client_id][name] = (output_idx_i_m, input_idx_i_m)

                                    # ---- 2) decoder.linear2：固定输出类别，裁剪输入（embedding 维） ----
                                    elif 'decoder' in name and 'linear2' in name:
                                        # weight: [vocab_size, d_model]
                                        input_idx_i_m  = idx_i[client_id]  # 从上一层继承
                                        output_idx_i_m = torch.arange(output_size, device=param.device)
                                        idx[client_id][name] = (output_idx_i_m, input_idx_i_m)

                                    # ---- 3) Multi-head Attention 的 q/k/v ----
                                    elif ('linear_q' in name) or ('linear_k' in name) or ('linear_v' in name):
                                        input_idx_i_m = idx_i[client_id]
                                        # 这里不再用 local_embed_dim 强行对齐，只要求「当前通道数」就是 local_dim
                                        if input_idx_i_m.numel() != local_dim:
                                            # 以 embedding 阶段为准，直接用当前 idx_i 长度重定义 local_dim
                                            # 或者你也可以选择 assert 这里强制检查
                                            local_dim = input_idx_i_m.numel()
                                            client_local_dim[client_id] = local_dim

                                        # 输出我们也截到 local_dim
                                        assert output_size >= local_dim
                                        ridx = torch.arange(output_size, device=param.device)
                                        if Roll:
                                            roll = round_n % output_size
                                            ridx = torch.roll(ridx, roll, -1)
                                        output_idx_i_m = ridx[:local_dim]
                                        if first_layer_channel_bounds[client_id] is None and output_idx_i_m.numel() > 0:
                                            if first_layer_name is None:
                                                first_layer_name = name
                                            first_layer_channel_bounds[client_id] = (
                                                int(output_idx_i_m[0].item()),
                                                int(output_idx_i_m[-1].item()),
                                            )

                                        # 更新当前通道
                                        idx_i[client_id] = output_idx_i_m
                                        idx[client_id][name] = (output_idx_i_m, input_idx_i_m)

                                    # ---- 4) MHA 的 linear_o ----
                                    elif 'linear_o' in name:
                                        input_idx_i_m = idx_i[client_id]
                                        if input_idx_i_m.numel() != local_dim:
                                            local_dim = input_idx_i_m.numel()
                                            client_local_dim[client_id] = local_dim

                                        assert output_size >= local_dim
                                        ridx = torch.arange(output_size, device=param.device)
                                        if Roll:
                                            roll = round_n % output_size
                                            ridx = torch.roll(ridx, roll, -1)
                                        output_idx_i_m = ridx[:local_dim]
                                        if first_layer_channel_bounds[client_id] is None and output_idx_i_m.numel() > 0:
                                            if first_layer_name is None:
                                                first_layer_name = name
                                            first_layer_channel_bounds[client_id] = (
                                                int(output_idx_i_m[0].item()),
                                                int(output_idx_i_m[-1].item()),
                                            )

                                        idx_i[client_id] = output_idx_i_m
                                        idx[client_id][name] = (output_idx_i_m, input_idx_i_m)

                                    # ---- 5) 其它 DenseLinear（FFN 的 linear1 / linear2 等） ----
                                    else:
                                        input_idx_i_m = idx_i[client_id]
                                        curr_dim = input_idx_i_m.numel()
                                        # 输出大小按 scaler_rate 缩，或者同样截到 curr_dim / local_dim 都行
                                        local_out = max(1, int(math.ceil(output_size * scaler_rate)))
                                        local_out = min(local_out, output_size)

                                        ridx = torch.arange(output_size, device=param.device)
                                        if Roll:
                                            roll = round_n % output_size
                                            ridx = torch.roll(ridx, roll, -1)
                                        output_idx_i_m = ridx[:local_out]
                                        if first_layer_channel_bounds[client_id] is None and output_idx_i_m.numel() > 0:
                                            if first_layer_name is None:
                                                first_layer_name = name
                                            first_layer_channel_bounds[client_id] = (
                                                int(output_idx_i_m[0].item()),
                                                int(output_idx_i_m[-1].item()),
                                            )

                                        idx_i[client_id] = output_idx_i_m
                                        idx[client_id][name] = (output_idx_i_m, input_idx_i_m)

                            else:
                                # 标量权重（极少见），直接沿用当前通道索引
                                for client_id in range(num_clients):
                                    input_idx_i_m = idx_i[client_id]
                                    idx[client_id][name] = input_idx_i_m

                        # =====================
                        # 2) 处理 bias
                        # =====================
                        else:
                            output_size = param.size(0)

                            for client_id in range(num_clients):
                                # decoder.linear2 的 bias：所有类别保留
                                if 'decoder' in name and 'linear2' in name:
                                    input_idx_i_m = torch.arange(output_size, device=param.device)

                                # q/k/v 的 bias：沿用当前通道
                                elif ('linear_q' in name) or ('linear_k' in name) or ('linear_v' in name):
                                    input_idx_i_m = idx_i[client_id]

                                    # 如果你之前有「q/k 回写到 weight 输入索引」的逻辑，可以保留：
                                    if 'linear_v' not in name:
                                        w_out_idx, w_in_idx = idx[client_id][name.replace('bias', 'weight')]
                                        idx_i[client_id] = w_in_idx

                                # 其它 bias：对应输出通道
                                else:
                                    input_idx_i_m = idx_i[client_id]

                                idx[client_id][name] = input_idx_i_m

                    else:
                        # 非 weight/bias，比如 LayerNorm 的 running stats 等，先不裁剪
                        pass

                    
                else:
                    if 'weight' in name or 'bias' in name:
                        if 'weight' in name:
                            if param.dim() > 1:
                                in_channels = param.size(1)
                                out_channels = param.size(0)
                                if idx_i[client_id] is None:
                                    idx_i[client_id] = torch.arange(in_channels, device=param.device)
                                input_idx_i_m = idx_i[client_id]
                                if name.startswith(output_prefix):
                                    output_idx_i_m = torch.arange(out_channels, device=param.device)
                                else:
                                    
                                    local_output_size = int(np.ceil(density[client_id]**0.5 * out_channels))
                                    if Roll:
                                        base_idx = torch.arange(out_channels, device=param.device)
                                        roll = round_n % out_channels
                                        model_idx = torch.roll(base_idx, shifts=roll, dims=-1)
                                    else:
                                        model_idx = torch.arange(out_channels, device=param.device)
                                    
                                    output_idx_i_m = model_idx[:local_output_size]
                                    if first_layer_channel_bounds[client_id] is None and output_idx_i_m.numel() > 0:
                                        if first_layer_name is None:
                                            first_layer_name = name
                                        first_layer_channel_bounds[client_id] = (
                                            int(output_idx_i_m[0].item()),
                                            int(output_idx_i_m[-1].item()),
                                        )
                                if self.config.EXP_NAME == "FEMNIST" and 'classifier.0.' in name:
                                    spatial = 7 * 7
                                    expanded = []
                                    for c in input_idx_i_m:
                                        base = c.item() * spatial
                                        expanded.extend(range(base, base + spatial))
                                    input_idx_i_m = torch.tensor(expanded, device=input_idx_i_m.device)
                                    
                                idx[client_id][name] = output_idx_i_m, input_idx_i_m
                                idx_i[client_id] = output_idx_i_m
                            else:
                                input_idx_i_m = idx_i[client_id]
                                idx[client_id][name] = input_idx_i_m
                        else:
                            in_channels = param.size(0)
                            if name.startswith(output_prefix):
                                input_idx_i_m = idx_i[client_id]
                                idx[client_id][name] = input_idx_i_m
                            else:
                                input_idx_i_m = idx_i[client_id]
                                idx[client_id][name] = input_idx_i_m
                    else:
                        pass
            
            density_total = 0
            density_kept = 0

            for name, param in self.model.named_parameters():
                if name in idx[client_id]:
                    idx_k = idx[client_id][name]
                    if isinstance(idx_k, tuple):
                        out_idx, in_idx = idx_k
                        if param.dim() == 2:
                            total = param.numel()
                            kept = len(out_idx) * len(in_idx)
                        elif param.dim() == 4:
                            total = param.numel()
                            kept = len(out_idx) * len(in_idx) * param.shape[2] * param.shape[3]
                        else:
                            raise NotImplementedError(f"Unsupported dim {param.dim()} for {name}")
                    elif param.dim() == 1:
                        in_idx = idx_k
                        total = param.numel()
                        kept = len(in_idx)
                    else:
                        raise NotImplementedError(f"Unsupported dim {param.dim()} for {name}")
                    
                    density_total += total
                    density_kept += kept

            actual_client_density[client_id] = round(density_kept / density_total, 4)
               
        self.first_layer_channel_name = first_layer_name
        self.first_layer_channel_bounds = first_layer_channel_bounds
        self.actual_client_density = actual_client_density
        return idx


    @torch.no_grad()
    def get_client_dict(self, list_state_dict):
        new_list_state_dict = []

        for state_dict, mask in zip(list_state_dict, self.list_mask):
            new_state = OrderedDict()

            for k, v in state_dict.items():
                if 'weight' in k or 'bias' in k:
                    if 'weight' in k:
                        if v.dim() == 4:  # Conv2d: [out_channels, in_channels, kH, kW]
                            out_idx, in_idx = mask[k]
                            mask_tensor = torch.zeros_like(v)
                            mask_tensor[out_idx[:, None, None, None],
                                        in_idx[None, :, None, None],
                                        :, :] = 1
                            new_state[k] = v * mask_tensor

                        elif v.dim() == 2:  # Linear: [out_features, in_features]
                            out_idx, in_idx = mask[k]
                            mask_tensor = torch.zeros_like(v)
                            mask_tensor[out_idx[:, None], in_idx[None, :]] = 1
                            new_state[k] = v * mask_tensor

                        elif v.dim() == 1:  # BN.weight
                            idx = mask[k]
                            mask_tensor = torch.zeros_like(v)
                            mask_tensor[idx] = 1
                            new_state[k] = v * mask_tensor

                        else:
                            raise NotImplementedError(f"{k} has unexpected dim {v.dim()}")

                    elif 'bias' in k:
                        idx = mask[k]
                        mask_tensor = torch.zeros_like(v)
                        mask_tensor[idx] = 1
                        new_state[k] = v * mask_tensor

                else:
                    # 非参数部分直接复制
                    new_state[k] = v

            new_list_state_dict.append(new_state)

        return new_list_state_dict

    
    def save(self,list_acc, list_fed_avg_acc):
            mkdir_save(list_acc, os.path.join(self.save_path, 'self.list_acc.pt'))
            mkdir_save(list_fed_avg_acc, os.path.join(self.save_path, 'fed_avg_acc.pt'))

    def main(self, idx, list_sd, list_num_proc, start, list_loss, list_acc,list_fed_avg_loss, list_fed_avg_acc, list_est_time,
             list_model_size, ):
        total_num_proc = sum(list_num_proc)
        list_sd = self.get_client_dict(list_sd)
        self_sd = self.model.state_dict()
        sd = deepcopy(self_sd)
        fed_avg_sd = deepcopy(self_sd)
        
        with torch.no_grad():
            keys = [k for k in self.model.state_dict().keys() if not k.endswith("num_batches_tracked")]
            
            # 初始化聚合结果
            sum_weight_dict = {k: torch.zeros_like(self_sd[k], device=self.device) for k in keys}
            sum_mask_dict = {k: torch.zeros_like(self_sd[k], device=self.device) for k in keys}
            sd = OrderedDict()
            fed_avg_sd = OrderedDict()

            # 记录每个客户端的 density
            self.client_density = []
    
            for state in list_sd:
                if state is None:
                    self.client_density.append(0.0)
                    continue

                total_elem = 0
                nonzero_elem = 0

                for key in keys:
                    v = state[key]

                    # 累加用于 fedavg
                    sum_weight_dict[key] += v

                    # 累加非零mask
                    mask = (v != 0)
                    sum_mask_dict[key] += mask

                    # 计算density
                #     total_elem += v.numel()
                #     nonzero_elem += mask.sum().item()

                # density = round(nonzero_elem / total_elem, 4)
                # print('client'+str(density))

            # 聚合
            for key in keys:
                divisor = torch.where(sum_mask_dict[key] == 0, torch.full_like(sum_mask_dict[key], 1e-10), sum_mask_dict[key])
                merged = torch.div(sum_weight_dict[key], divisor)
                sd[key] = merged
                fed_avg_sd[key] = sum_weight_dict[key] / len(list_sd)

            self.model.load_state_dict(sd)


        if idx % self.args.interval == 0:
            self.model.load_state_dict(fed_avg_sd)   
            fed_avg_loss, fed_avg_acc = self.model.evaluate(self.test_loader)
            self.model.load_state_dict(sd) 
            loss, acc = self.model.evaluate(self.test_loader)
            list_loss.append(loss)
            list_acc.append(acc)
            list_fed_avg_acc.append(fed_avg_acc)
            list_fed_avg_loss.append(fed_avg_loss)

            print("Round #{} (Experiment = {}).".format(idx, self.experiment_name))
            print(f"fed_avg Loss/acc (at round #{idx}) = {fed_avg_loss}/{fed_avg_acc}   Loss/acc={loss}/{acc}")
            print("Elapsed time = {}".format(timer() - start))

            self.logger.info("Round #{} (Experiment = {}).".format(idx, self.experiment_name))
            self.logger.info(f"fed_avg Loss/acc (at round #{idx}) = {fed_avg_loss}/{fed_avg_acc}   Loss/acc={loss}/{acc}")
            self.logger.info("Elapsed time = {}".format(timer() - start))
            self.save(list_acc, list_fed_avg_acc)






        return [self.model.state_dict() for _ in range(self.args.number_clients)]


class FjordClient:
    def __init__(self, model, config,args,  model_rate):
        self.config = config
        self.device = torch.device("cuda:" + str(args.device) if torch.cuda.is_available() else "cpu")
        self.args = args
        self.model = deepcopy(model).to(self.device)
        self.optimizer = None
        self.optimizer_scheduler = None
        self.optimizer_wrapper = None
        self.train_loader = None
        self.model_rate = model_rate


    @abstractmethod
    def init_optimizer(self, *args, **kwargs):
        pass

    @abstractmethod
    def init_train_loader(self, *args, **kwargs):
        pass
    
    def load_lr(self,lr):
        self.optimizer_wrapper.set_lr(lr)
        
    def main(self, ):
        model_rate = self.model_rate
        self.model.train()
        num_proc_data = 0



        density = sorted(set([0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00,model_rate]))
        idx = density.index(model_rate)
        # print(model_rate)
        for i in range(self.config.NUM_LOCAL_UPDATES):
            rate = density[i%(idx+1)]
            # rate = model_rate
            inputs, labels = self.train_loader.get_next_batch()

            self.optimizer_wrapper.step3(inputs.to(self.device), labels.to(self.device),rate)
            num_proc_data += len(inputs)

        return self.model.state_dict(), num_proc_data

    def load_mask(self, masks):
        self.list_mask = masks

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)


class FjordFL(ABC):
    def __init__(self, args, config, server, client_list):
        self.config = config
        self.max_round = config.MAX_ROUND
        self.server = server
        self.client_list = client_list

        self.list_loss, self.list_acc, self.list_est_time, self.list_model_size = [], [], [], []
        self.list_fed_avg_loss, self.list_fed_avg_acc = [],[]
        self.start_adj_round = None

    def main(self):

        start = timer()
        for idx in range(self.max_round):
            list_state_dict, list_num = [], []



            for client in self.client_list:
                sd, npc= client.main()
                list_state_dict.append(sd)
                list_num.append(npc)

            

                


            new_list_sd = self.server.main(idx, list_state_dict, list_num,   start,
                                                      self.list_loss, self.list_acc,self.list_fed_avg_loss, self.list_fed_avg_acc, self.list_est_time,
                                                      self.list_model_size,  )
            global_lr = None
            if self.server.ip_optimizer_wrapper is not None:
                self.server.ip_optimizer_wrapper.lr_scheduler_step()
                global_lr = self.server.ip_optimizer_wrapper.get_last_lr()

            # Step 2: Push model (and possibly LR) to each active client
            for i in range(len(self.client_list)):
                    client = self.client_list[i]
                    if global_lr is not None:
                        client.load_lr(global_lr)
                        
            for client, new_sd in zip(self.client_list, new_list_sd):
                client.load_state_dict(new_sd)
                
        self.server.save(self.list_acc, self.list_fed_avg_acc)
        return 3
