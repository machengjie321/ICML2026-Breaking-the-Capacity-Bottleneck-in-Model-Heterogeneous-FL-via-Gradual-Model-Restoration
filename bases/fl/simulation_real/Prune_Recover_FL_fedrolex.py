import argparse
import pickle
import copy
from collections import deque
from xml.dom import NoDataAllowedErr
from bases.optim.optimizer_wrapper import OptimizerWrapper
import torch.optim.lr_scheduler as lr_scheduler
from sympy import print_tree
import configs.InternetSpeed as internet_speed
import os
from copy import deepcopy
from typing import Union, Type, List
import torch
import numpy as np
from utils.save_load import mkdir_save, load
from abc import ABC, abstractmethod
from timeit import default_timer as timer
from bases.optim.optimizer_wrapper import OptimizerWrapper
from bases.nn.linear import DenseLinear, SparseLinear
import logging
import math
from collections import OrderedDict

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
    parser.add_argument('--recover_step_mode',
                        help="recover density mode: legacy | ladder | fixed",
                        action='store',
                        dest='recover_step_mode',
                        type=str,
                        default='legacy',
                        required=False)
    parser.add_argument('--recover_step',
                        help="fixed recover density step size",
                        action='store',
                        dest='recover_step',
                        type=float,
                        default=0.1,
                        required=False)
    parser.add_argument('--local_topk_mode',
                        help="sub-model ranking mode: global | client_replace",
                        action='store',
                        dest='local_topk_mode',
                        type=str,
                        default='global',
                        required=False)
    parser.add_argument('--measure_topk_overlap',
                        help="measure overlap between global top-k and client-replaced top-k",
                        dest='measure_topk_overlap',
                        action='store_true',
                        default=False,
                        required=False)
    parser.add_argument('--overlap_topk_ratio',
                        help="fallback top-k ratio when density is unavailable",
                        action='store',
                        dest='overlap_topk_ratio',
                        type=float,
                        default=0.1,
                        required=False)
    parser.add_argument('--prune_warmup_rounds',
                        help="number of initial rounds to train without pruning before switching to target densities",
                        action='store',
                        dest='prune_warmup_rounds',
                        type=int,
                        default=0,
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
    
    parser.add_argument('-clip',
                        help="clip ",
                        dest='clip',
                        action='store_true',
                        default=False,
                        required=False
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
    
    parser.add_argument('-server_up',
                        help="server upload speed",
                        action='store',
                        dest='server_up_speed',
                        type=float,
                        default=10,
                        required=False
                        )

    parser.add_argument('-bp',
                        help="bias_prune",
                        dest='bias_prune',
                        action='store_true',
                        default=False,
                        required=False)
    parser.add_argument('-mu',
                        help="FedProx mu",
                        action='store',
                        dest='mu',
                        type=float,
                        default=0.01,
                        required=False
                        )

    return parser.parse_args()


def is_heterofl_method(method_name):
    method_name = method_name.lower()
    return method_name.startswith('heterofl') or method_name.endswith('_heterofl')


def is_fedrolex_method(method_name):
    method_name = method_name.lower()
    return method_name.startswith('fedrolex') or method_name.endswith('_fedrolex')


def is_fjord_method(method_name):
    method_name = method_name.lower()
    return method_name.startswith('fjord') or method_name.endswith('_fjord')


def is_hetero_family_method(method_name):
    return (
        is_heterofl_method(method_name)
        or is_fedrolex_method(method_name)
        or is_fjord_method(method_name)
    )


def unpack_client_state_payload(payload):
    if payload is None:
        return None, None, None
    if len(payload) == 2:
        idx, state_dict = payload
        return idx, state_dict, None
    if len(payload) == 3:
        idx, state_dict, mask = payload
        return idx, state_dict, mask
    raise ValueError(f"Unexpected client payload length: {len(payload)}")


class ExpConfig:  # setup the config
    def __init__(self, exp_name: str, save_dir_name: str, seed: int, batch_size: int, num_local_updates: int,
                 optimizer_class: Type, optimizer_params: dict, lr_scheduler_class: Union[Type, None],
                 lr_scheduler_params: Union[dict, None], use_adaptive: bool, device=None):
        self.exp_name = exp_name
        self.save_dir_name = save_dir_name
        self.seed = seed
        self.batch_size = batch_size
        self.num_local_updates = num_local_updates
        self.optimizer_class = optimizer_class
        self.optimizer_params = optimizer_params
        self.lr_scheduler_class = lr_scheduler_class
        self.lr_scheduler_params = lr_scheduler_params
        self.use_adaptive = use_adaptive
        self.device = device


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=10, verbose=False, delta=0, client_num=10):

        self.num = 1 + client_num

        self.patience = [patience] * self.num
        self.verbose = verbose
        self.counter = [0] * self.num

        self.best_score = [None] * self.num
        from collections import deque

        self.average_score = [None] * self.num
        self.early_stop = [False] * self.num
        self.val_loss_min = np.inf
        self.delta = delta
        self.state = [None] * self.num
        self.last_score = [None] * self.num

    def __call__(self, acc, logger):

        score = acc
        self.state = [None] * self.num
        for i in range(self.num):
            if self.best_score[i] is None:
                self.best_score[i] = score[i]
                self.state[i] = True
                self.last_score[i] = score[i]
                continue

            elif i == 0 and self.last_score[i] == score[i]:
                self.state[i] = True
                continue

            elif score[i] <= self.best_score[i] + self.delta:
                self.last_score[i] = score[i]
                self.counter[i] += 1

                if self.counter[i] >= self.patience[i]:
                    if i == 0:
                        logger.info(
                            f'server out of patience, best score is {self.best_score[i]}, current score is {score[i]}')
                        print(
                            f' server out of patience, best score is {self.best_score[i]}, current score is {score[i]}')
                    else:
                        logger.info(
                            f'client {i} out of patience, best score is {self.best_score[i]}, current score is {score[i]}')
                        print(
                            f' client {i} out of patience, best score is {self.best_score[i]}, current score is {score[i]}')
                    self.early_stop[i] = True
                    self.state[i] = False
                    continue
            else:
                self.last_score[i] = score[i]
                self.best_score[i] = score[i]
                self.counter[i] = 0
                self.state[i] = True
        return self.state


# experiments/FEMNIST/PMT.py -a -i -s 0 -e
class FedMapServer(ABC):
    def __init__(self, config, args, model, seed, optimizer_class: Type, optimizer_params: dict,
                 use_adaptive, use_evaluate=True, lr_scheduler_class=None, lr_scheduler_params=None, control=None,
                 control_scheduler=None, resume=False, init_time_offset=0, device=None):
        self.config = config
        self.experiment_name = args.experiment_name
        self.recover = args.recover
        self.save_path = os.path.join("results", config.EXP_NAME, args.experiment_name)
        self.save_interval = 50
        self.use_adaptive = True
        self.client_selection = args.client_selection
        self.download_speed = internet_speed.high_download_speed
        self.upload_speed = internet_speed.high_upload_speed
        self.client_list = None
        
        self.increase = args.increase
        
        self.exp_config = ExpConfig(self.config.EXP_NAME, self.save_path, seed, self.config.CLIENT_BATCH_SIZE,
                                    self.config.NUM_LOCAL_UPDATES, optimizer_class, optimizer_params,
                                    lr_scheduler_class,
                                    lr_scheduler_params, use_adaptive)

        self.device = torch.device("cuda:" + str(args.device) if torch.cuda.is_available() else "cpu")

        self.model = model.to(self.device)
        self.number_clients = args.number_clients

        self.need_client_acc = args.need_client_acc

        self.indices = None

        self.client_is_sparse = False
        self.test_data,self.test_label = None, None
        self.ex = args.ex
        self.method = args.ex

        self.ip_train_loader = None
        self.ip_test_loader = None
        self.ip_optimizer_wrapper = None
        self.ip_control = None
        self.round = None

        self.test_loader = None
        self.control = None
        self.init_test_loader()

        self.init_control()
        self.init_ip_config()
        self.save_exp_config()
        self.start_time = timer()
        self.min_density = args.min_density
        self.target_min_density = float(args.min_density)
        self.list_mask = None
        self.model_idx = None
        self.interval = args.interval
        self.merge = args.merge
        self.sever_to_client_sum = []
        self.fed_avg_acc = []
        self.model_G = []
        self.fed_avg_loss = []
        self.user_ids = None
        self.accumulate = args.accumulate
        self.increse = float(args.increase)
        self.recover_step_mode = getattr(args, 'recover_step_mode', 'legacy')
        if self.recover_step_mode is None:
            self.recover_step_mode = 'legacy'
        self.recover_step_mode = str(self.recover_step_mode).lower()
        if self.recover_step_mode not in {'legacy', 'ladder', 'fixed'}:
            self.recover_step_mode = 'legacy'
        self.recover_step = float(getattr(args, 'recover_step', 0.1))
        self.use_coeff = args.use_coeff if hasattr(args,'use_coeff') else True
        self.display = False
        self.logger = None
        self.old_list_mask = None
        self.old_accumulate_weight_dict = None
        self.list_loss = []
        self.list_acc = [0]
        self.fed_avg_acc = []
        self.fed_avg_loss = []
        self.list_est_time = []
        self.list_model_size = []
        self.list_client_size = []
        self.args = args

        self.sever_to_client_sum = []

        self.model_size = []
        self.list_optimizer = None

        self.early_stop = False
        self.target_client_density = copy.deepcopy(args.client_density)
        self.prune_warmup_rounds = max(0, int(getattr(args, 'prune_warmup_rounds', 0)))
        self.prune_warmup_active = self.prune_warmup_rounds > 0
        self.prune_warmup_transition_done = self.prune_warmup_rounds <= 0
        self.client_density = (
            [1.0 for _ in self.target_client_density]
            if self.prune_warmup_active
            else copy.deepcopy(self.target_client_density)
        )
        if self.prune_warmup_active:
            self.min_density = 1.0
        self.list_client_density = [[] for _ in range(len(self.client_density))]

        self.list_client_loss = [[] for _ in range(len(self.client_density))]
        self.list_client_acc = [[] for _ in range(len(self.client_density))]
        self.list_threshold = [0] * len(self.client_density)
        self.early_stoping = EarlyStopping(patience=args.patience,
                                           client_num=len(self.client_density) if self.need_client_acc else 0)
        self.patience = args.patience
        self.new_mask = None

        self.sum_server_upload = [0]
        self.sum_server_download = [[0] for _ in range(len(self.client_density))]
        self.optimizer_scheduler = None
        self.num = 0
        self.increment_model_size = None
        self.time = []
        self.train_number = None
        self.list_train_number = [[] for _ in range(len(self.client_density))]
        self.list_client_dict = [None for _ in range(len(self.client_density))]
        self.sub_density = [0 for _ in range(len(self.client_density))]
        self.client_model_size = None
        self.list_state_dict = None
        self.first_stage = True
        self.list_stalness = [0 for _ in range(len(self.client_density))]
        self.list_coeff = [0 for _ in range(len(self.client_density))]
        self.stal = args.stal
        self.stal_a = args.stal_a

        self.list_client_sd_buffer = [None for _ in range(len(self.client_density))]
        self.client_arrive_num = [0 for _ in range(len(self.client_density))]
        self.old_model = copy.deepcopy(self.model.state_dict())
        self.average_round_time = None
        self.interval_signal = False
        self.client_train_time = [[0.0] for i in range(len(self.client_density))]
        self.sum_client_train_time = [[0.0] for i in range(len(self.client_density))]
        self.server_merge_time = [0.0]
        self.sum_server_merge_time = [0.0]
        self.client_upload_time = [[0.0] for i in range(len(self.client_density))]
        self.sum_client_upload_time = [[0.0] for i in range(len(self.client_density))]
        self.client_download_time = [[0.0] for i in range(len(self.client_density))]
        self.sum_client_download_time = [[0.0] for i in range(len(self.client_density))]
        self.density_size = {}
        self.begin_save = False
        self.server_accumulate_number = []
        self.wast_time = []
        self.test_split_statu = True
        self.test_split_acc = 0
        self.save_for_split = deque(config.save_for_split)
        self.sp = self.save_for_split.popleft()
        self.restore_density = sorted(set(self.target_client_density))
        self.actual_client_density = copy.deepcopy(self.client_density)
        self.first_layer_channel_name = None
        self.first_layer_channel_bounds = []
        self.wait_for_stable = 0
        self.client_wait_for_stable = [0 for _ in range(len(self.client_density))]
        self.seed = args.seed
        
        self.model_idxs = {}
        self.roll_idx = {}

        self.reshuffle_params()
        
        # fjord
        self.increment_state_dict = None
        self.fjord_masks = []
        if is_hetero_family_method(self.method):
            self.control.accumulate_w(self.model)
                # get the client model based on client_density
            list_state_dict, self.model_idx, sub_model_time, self.list_mask, _, self.increment_state_dict, _, _ = self.control.sub_adjust_fast(
                            client_density=self.client_density, use_coff=self.use_coeff,min_density=self.config.min_density, )
            if is_fjord_method(self.method):
                fjord_client_density = sorted(set([0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00] + self.client_density))

                fjord_mask_all = self.roll_split_universal(0, self.model,fjord_client_density,Roll=False)
                for cd in self.client_density:
                    idx = fjord_client_density.index(cd)  # 找到密度的位置
                    self.fjord_masks.append(fjord_mask_all[ :idx+1])  # 引用对应 mask
            else:
                self.list_mask = self.roll_split_universal(0, self.model, self.client_density, Roll=False)

    def get_client_wait_for_stable_steps(self):
        return max(1, int(self.patience * 2))

    def tick_client_wait_for_stable(self):
        for i in range(len(self.client_wait_for_stable)):
            if self.client_wait_for_stable[i] > 0:
                self.client_wait_for_stable[i] -= 1

    def get_recover_mode(self):
        return self.recover_step_mode

    def get_ladder_next_density(self, current_density):
        restore_density = sorted(set(self.restore_density + [current_density]))
        next_density_candidates = [d for d in restore_density if d > current_density + 0.01]
        if next_density_candidates:
            return min(next_density_candidates)
        return 1.0

    def get_fixed_next_density(self, current_density):
        step = max(0.0, float(self.recover_step))
        return min(1.0, current_density + step)

    def in_prune_warmup(self, round_idx):
        return self.prune_warmup_rounds > 0 and int(round_idx) < self.prune_warmup_rounds

    def reset_early_stopping_state(self):
        self.wait_for_stable = 0
        self.client_wait_for_stable = [0 for _ in range(len(self.client_density))]
        self.begin_save = False
        for j in range(len(self.early_stoping.counter)):
            self.early_stoping.counter[j] = 0
            self.early_stoping.best_score[j] = None
            self.early_stoping.early_stop[j] = False
            self.early_stoping.state[j] = None
            self.early_stoping.last_score[j] = None

    def update_prune_warmup_state(self, round_idx):
        if self.prune_warmup_rounds <= 0:
            return False

        round_idx = int(round_idx)
        if round_idx < self.prune_warmup_rounds:
            self.prune_warmup_active = True
            if any(abs(float(d) - 1.0) > 1e-12 for d in self.client_density) or self.min_density < 0.999999:
                self.client_density = [1.0 for _ in self.target_client_density]
                self.min_density = 1.0
                self.actual_client_density = copy.deepcopy(self.client_density)
            return False

        if not self.prune_warmup_transition_done:
            self.prune_warmup_active = False
            self.prune_warmup_transition_done = True
            self.client_density = copy.deepcopy(self.target_client_density)
            self.min_density = self.target_min_density
            self.actual_client_density = copy.deepcopy(self.client_density)
            self.reset_early_stopping_state()
            message = (
                f"Finish prune warmup at round {round_idx}: switch to target density list "
                f"{self.client_density}"
            )
            print(message)
            if self.logger is not None:
                self.logger.info(message)
            return True

        return False



    def save_checkpoint(self):
        '''
            功能：当验证损失减少时保存模型
            input:
                val_loss         验证损失
                model            模型
                model_path       模型保存地址
        '''
        self.list_est_time.append(timer() - self.start_time)

        checkpoint = {'self.model': self.model,
                      'self.list_client_sd': self.list_client_sd_buffer,
                      'self.control.accumulate_weight_dict': self.control.accumulate_weight_dict,
                      'self.list_mask': self.list_mask,
                      'self.list_stalness': self.list_stalness,
                      'self.list_coeff': self.list_coeff,
                      }

        checkpoint_path = os.path.join(self.save_path, 'checkpoint.pth')
        mkdir_save(checkpoint, checkpoint_path)

    def delete_checkpoint(self):
        checkpoint_path = os.path.join(self.save_path, 'checkpoint.pth')
        try:
            os.remove(checkpoint_path)
        except FileNotFoundError:
            pass

    def save_final_model(self):
        checkpoint = {'self.model': self.model, }
        checkpoint_path = os.path.join(self.save_path, 'final_model.pth')
        mkdir_save(checkpoint, checkpoint_path)

    def save_checkpoint_for_split(self, acc, fedavg_model):
        '''
            功能：当验证损失减少时保存模型
            input:
                val_loss         验证损失
                model            模型
                model_path       模型保存地址
        '''
        if self.ex in ['fed_avg','fed_asyn','heterofl','pr_fl']:
            self.list_est_time.append(timer() - self.start_time)
            checkpoint = {'self.model': fedavg_model,
                          'self.control.accumulate_weight_dict':self.control.accumulate_weight_dict
                          }
            checkpoint_path = os.path.join(self.save_path, 'split', str(acc), 'checkpoint.pth')

            mkdir_save(checkpoint, checkpoint_path)
            # exit(0)

    def save_display_data(self):
        mkdir_save(self.list_client_acc, os.path.join(self.save_path, 'list_client_acc.pt'))
        mkdir_save(self.list_loss, os.path.join(self.save_path, 'self.list_loss.pt'))
        mkdir_save(self.list_acc, os.path.join(self.save_path, 'self.list_acc.pt'))

        mkdir_save(self.fed_avg_acc, os.path.join(self.save_path, 'fed_avg_acc.pt'))
        mkdir_save(self.list_client_loss, os.path.join(self.save_path, 'list_client_loss.pt'))
        mkdir_save(self.fed_avg_loss, os.path.join(self.save_path, 'fed_avg_loss.pt'))
        mkdir_save(self.list_client_density, os.path.join(self.save_path, 'self.list_client_density'))
        mkdir_save(self.time, os.path.join(self.save_path, 'self.time'))
        mkdir_save(self.train_number, os.path.join(self.save_path, 'self.train_number'))
        mkdir_save(self.list_train_number, os.path.join(self.save_path, 'self.list_train_number'))
        mkdir_save(self.client_train_time, os.path.join(self.save_path, 'self.client_train_time'))
        mkdir_save(self.sum_client_train_time, os.path.join(self.save_path, 'self.sum_client_train_time'))
        mkdir_save(self.server_merge_time, os.path.join(self.save_path, 'self.server_merge_time'))
        mkdir_save(self.sum_server_merge_time, os.path.join(self.save_path, 'self.sum_server_merge_time'))
        mkdir_save(self.client_upload_time, os.path.join(self.save_path, 'self.client_upload_time'))
        mkdir_save(self.sum_client_upload_time, os.path.join(self.save_path, 'self.sum_client_upload_time'))
        mkdir_save(self.client_download_time, os.path.join(self.save_path, 'self.client_download_time'))
        mkdir_save(self.sum_client_download_time, os.path.join(self.save_path, 'self.sum_client_download_time'))
        mkdir_save(self.server_accumulate_number, os.path.join(self.save_path, 'self.server_accumulate_number'))
        mkdir_save(self.sum_server_download, os.path.join(self.save_path, 'self.sum_server_download'))
        mkdir_save(self.sum_server_upload, os.path.join(self.save_path, 'self.sum_server_upload'))

    def get_save_dir_name(self):
        if not self.use_adaptive:
            return "conventional"
        else:
            mdd_100, chl = 100 * self.config.MAX_DEC_DIFF, self.config.ADJ_HALF_LIFE
            lrhl = self.config.LR_HALF_LIFE if hasattr(self.config, "LR_HALF_LIFE") else None
            assert mdd_100 - int(mdd_100) == 0
            return "mdd{}_chl{}_lrhl{}".format(int(mdd_100), lrhl, chl)



    @abstractmethod
    def init_test_loader(self):
        pass

    @abstractmethod
    def init_clients(self):
        pass

    @abstractmethod
    def init_control(self, *args, **kwargs):
        pass

    @abstractmethod
    def save_exp_config(self, *args, **kwargs):
        pass

    @abstractmethod
    def init_ip_config(self, *args, **kwargs):
        pass

    def check_client_to_sparse(self):  # if model.density() <= config.TO_SPARSE_THR ,set the model to sparse
        if not self.client_is_sparse and self.model.density() <= self.config.TO_SPARSE_THR:
            self.client_is_sparse = True

    def get_real_size(self, list_state_dict, exp, density):
        list_model_size = []
        for i in range(len(list_state_dict)):
            if list_state_dict[i] == None or density[i] == 0:
                list_model_size.append(0)
                continue

            file_size = self.config.model_size * density[i]
            list_model_size.append(file_size)

        return list_model_size

    def clean_dict_to_client(self, state_dict) -> dict:
        """
        Clean up state dict before processing, e.g. remove entries, transpose.
        To be overridden by subclasses.
        """
        with torch.no_grad():
            clean_state_dict = state_dict  # not deepcopy

            for layer, prefix in zip(self.model.param_layers, self.model.param_layer_prefixes):
                key = prefix + ".bias"
                if isinstance(layer, DenseLinear) and key in clean_state_dict.keys():

                    if clean_state_dict[key].sum() != 0:
                        clean_state_dict[key] = clean_state_dict[key].view((-1, 1))
                    else:
                        clean_state_dict[key] = None

        return clean_state_dict

    @torch.no_grad()
    def process_state_dict_to_client(self, list_state_dict):
        """
        Process list_state_dict before sending to client, e.g. to cpu, to sparse, keep values only.
        if not self.client_is_sparse: send dense
        elif self.is_adj_round(): send full sparse state_dict
        else: send sparse values only
        To be overridden by subclasses.
        """
        list_state_dict = copy.deepcopy(list_state_dict)

        with torch.no_grad():
            for i in range(len(list_state_dict)):
                list_state_dict[i] = self.clean_dict_to_client(list_state_dict[i])

            for clean_state_dict in list_state_dict:
                for _, prefix in zip(self.model.prunable_layers, self.model.prunable_layer_prefixes):
                    # works for both layers
                    key_w = prefix + ".weight"
                    if key_w in clean_state_dict.keys():
                        weight = clean_state_dict[key_w]
                        sparse_weight = weight.view(weight.size(0), -1).to_sparse()
                        if sparse_weight._nnz() == 0:
                            sparse_weight = None
                        clean_state_dict[key_w] = sparse_weight

        return list_state_dict

    def process_state_dict_to_client_fast(self, list_state_dict):

        return list_state_dict

    def calculate_staleness(self, list_state_dict):
        list_stalness = [0 for _ in range(len(self.client_density))]

        for i in range(len(list_state_dict)):
            idx_state_dict = list_state_dict[i]

            if idx_state_dict == None:
                continue
            else:
                idx, _, _ = unpack_client_state_payload(idx_state_dict)
                time = self.round - idx + 1
                if time <= 0: time = 1.0
                if self.stal.lower().startswith('con'):
                    list_stalness[i] = 1.0
                elif self.stal.lower().startswith('poly'):
                    a = self.stal_a
                    list_stalness[i] = math.pow(time, -a)
                elif self.stal.lower().startswith('hinge'):
                    a = self.stal_a
                    if self.round - idx <= 2:
                        list_stalness[i] = 1.0
                    else:
                        list_stalness[i] = 1.0 / (a * (self.round - idx - 2) + 1)

        for i in range(len(list_stalness)):
            if list_stalness[i] != 0:
                self.list_stalness[i] = list_stalness[i]

        sum_client = sum(self.list_stalness)
        if sum_client != 0:
            assert sum_client != 0
            for i in range(len(list_stalness)):
                self.list_coeff[i] = self.list_stalness[i] / sum_client


                    

    @torch.no_grad()
    def fed_avg_model(self, list_num_proc, list_state_dict):
        '''
        fed_avg_model = fed_avg(list_num_proc, list_state_dict)
        model =  fed_avg_model
        :param list_num_proc:
        :param list_state_dict:
        :param idx:
        :return:
        '''
        total_num_proc = sum(list_num_proc)
        if total_num_proc == 0:
            return self.model
        fed_avg_model = copy.deepcopy(self.model)
        list_state_dict = self.list_client_sd_buffer

        coeff = [list_num_proc[i] * self.list_coeff[i] for i in range(len(list_state_dict))]

        total_coeff = sum(coeff)

        with torch.no_grad():
            for key, param in fed_avg_model.state_dict().items():
                avg_inc_val = None
                i = 0
                for num_proc, idx_state_dict in zip(list_num_proc, list_state_dict):
                    if idx_state_dict == None:
                        continue
                    idx, state_dict, _ = unpack_client_state_payload(idx_state_dict)
                    if key in state_dict.keys():
                        if state_dict[key].is_sparse: state_dict[key] = state_dict[key].to_dense()
                        state_dict[key] = state_dict[key].view(param.size())
                        mask = state_dict[key] != 0
                        if mask is None:
                            inc_val = state_dict[key].to(self.device) - param
                        else:
                            inc_val = state_dict[key].to(self.device) - param
                        inc_val.view(param.size())

                        if avg_inc_val is None:
                            avg_inc_val = coeff[i] / total_coeff * inc_val
                        else:
                            avg_inc_val += coeff[i] / total_coeff * inc_val
                    i = i + 1

                if avg_inc_val is None or key.endswith("num_batches_tracked"):
                    continue
                else:
                    param.add_(avg_inc_val)

        return fed_avg_model

    @torch.no_grad()
    def fed_avg(self, list_num_proc, list_state_dict, idx, sgrd_to_upload):  # to complete the merge model ps: fedavg

        self_sd = self.model.state_dict()
        dict_keys = self_sd.keys()

        client_partion = 0
        for i in range(len(list_state_dict)):
            if list_state_dict[i] != None:
                client_partion += self.list_coeff[i]
        if client_partion == 0:
            return

        sd = copy.deepcopy(self_sd)
        for key in dict_keys:
            sum_weight = torch.zeros(size=self_sd[key].size(), device=self.device)
            sum_mask = torch.zeros(size=sum_weight.size(), device=self.device)
            if key.endswith("num_batches_tracked"):
                continue
            i = 0
            for num_proc, idx_state_dict in zip(list_num_proc, list_state_dict):
                if idx_state_dict == None: continue

                idx, state_dict, _ = unpack_client_state_payload(idx_state_dict)
                if state_dict[key].is_sparse: state_dict[key] = state_dict[key].to_dense()
                sum_weight = sum_weight + self.list_coeff[i] * num_proc * state_dict[key].to(self.device)
                mask = (state_dict[key] != 0).to(self.device)
                sum_mask = sum_mask + self.list_coeff[i] * mask * num_proc
                i = i + 1

            divisor = torch.where(sum_mask == 0, torch.tensor([1e-10], device=self.device), sum_mask)

            sum_weight = torch.div(sum_weight, divisor)
            mk = sum_weight == 0
            sum_weight = sum_weight.view(sd[key].size())
            sum_weight[mk] = sd[key][mk]

            if sum_weight.sum() == 0:
                continue
            sd[key] = sum_weight * client_partion + (1 - client_partion) * sd[key]

        self.model.load_state_dict(sd)
        if self.interval_signal:
            if self.accumulate == 'w':
                self.control.accumulate(self.old_model, idx, max(self.interval, 10))
                self.old_model = copy.deepcopy(self_sd)
            elif self.accumulate == 'wg':
                self.control.accumulate_wg_square(self.old_model)
                self.old_model = copy.deepcopy(self_sd)
            elif self.accumulate == 'g':
                self.control.accumulate_g(sgrd_to_upload)
            else:
                if self.display:
                    self.logger.info('wrong accumulate')

    @torch.no_grad()
    def buff_fed_avg(self, list_num_proc, list_state_dict, idx,
                     sgrd_to_upload):  # to complete the merge model ps: fedavg

        coeff = [list_num_proc[i] * self.list_coeff[i] for i in range(len(list_state_dict))]
        list_state_dict = self.list_client_sd_buffer
        total_coeff = sum(coeff)
        self_sd = self.model.state_dict()

        with torch.no_grad():
            for key, param in self.model.state_dict().items():
                avg_inc_val = None
                i = 0
                for num_proc, idx_state_dict in zip(list_num_proc, list_state_dict):
                    if idx_state_dict == None:
                        continue
                    idx, state_dict, _ = unpack_client_state_payload(idx_state_dict)
                    if key in state_dict.keys():
                        if state_dict[key].is_sparse: state_dict[key] = state_dict[key].to_dense()
                        state_dict[key] = state_dict[key].view(param.size())
                        mask = state_dict[key] != 0
                        if mask is None:
                            inc_val = state_dict[key].to(self.device) - param
                        else:
                            inc_val = state_dict[key].to(self.device) - param
                        inc_val.view(param.size())

                        if avg_inc_val is None:
                            avg_inc_val = coeff[i] / total_coeff * inc_val
                        else:
                            avg_inc_val += coeff[i] / total_coeff * inc_val
                    i = i + 1

                if avg_inc_val is None or key.endswith("num_batches_tracked"):
                    continue
                else:
                    param.add_(avg_inc_val)

        if self.interval_signal:
            if self.accumulate == 'w':
                self.control.accumulate(self.old_model, idx, max(self.interval, 10))
                self.old_model = copy.deepcopy(self_sd)
            elif self.accumulate == 'wg':
                self.control.accumulate_wg_square(self.old_model)
                self.old_model = copy.deepcopy(self_sd)
            elif self.accumulate == 'g':
                self.control.accumulate_g(sgrd_to_upload)
            else:
                if self.display:
                    self.logger.info('wrong accumulate')

    @torch.no_grad()
    def fed_avg_client_model(self, list_num_proc, list_state_dict):  # to complete the merge model ps: fedavg

        client_partion = 0
        for i in range(len(list_state_dict)):
            if list_state_dict[i] != None:
                client_partion += self.list_coeff[i]

        model = copy.deepcopy(self.model)
        coeff = [list_num_proc[i] * self.list_coeff[i] / client_partion for i in range(len(list_state_dict))]

        for i in range(len(list_state_dict)):
            if list_state_dict[i] == None:
                coeff[i] = 0
        print(coeff)

        total_coeff = sum(coeff)
        with torch.no_grad():
            for key, param in model.state_dict().items():
                avg_inc_val = None
                i = 0
                for num_proc, idx_state_dict in zip(list_num_proc, list_state_dict):
                    if idx_state_dict == None:
                        continue
                    idx, state_dict, _ = unpack_client_state_payload(idx_state_dict)
                    if key in state_dict.keys():
                        if state_dict[key].is_sparse: state_dict[key] = state_dict[key].to_dense()
                        state_dict[key] = state_dict[key].view(param.size())

                        inc_val = state_dict[key].to(self.device) - param

                        inc_val.view(param.size())

                        if avg_inc_val is None:
                            avg_inc_val = coeff[i] / total_coeff * inc_val
                        else:
                            avg_inc_val += coeff[i] / total_coeff * inc_val
                    i = i + 1

                if avg_inc_val is None or key.endswith("num_batches_tracked"):
                    continue
                else:
                    param.add_(avg_inc_val)

        return model


    @torch.no_grad()
    def split_model(self, start=0):
        list_state_dict = []
        sub_model_time = []
        
        

                
        for mask in self.list_mask:
            if is_hetero_family_method(self.method):
                clean_model = copy.deepcopy(self.model)
                clean_state_dict = clean_model.state_dict()
                for k, v in self.model.state_dict().items():
                    if 'weight' in k or 'bias' in k:
                        if 'weight' in k:
                            if v.dim() > 1:
                                clean_state_dict[k] = copy.deepcopy(v[torch.meshgrid(mask[k])])
                            else:
                                clean_state_dict[k] = copy.deepcopy(v[mask[k]])
                        else:
                            clean_state_dict[k] = copy.deepcopy(v[mask[k]])
                    else:
                        clean_state_dict[k] = copy.deepcopy(v)
        

                        

                        
            else:
                clean_model = copy.deepcopy(self.model)
                clean_state_dict = clean_model.state_dict()
                # for name, param in self.model.state_dict().items():
                #     if name 
                    
                for layer, prefix in zip(self.model.prunable_layers, self.model.prunable_layer_prefixes):
                    # works for both layers
                    key_w = prefix + ".weight"
                    if key_w in self.model.state_dict().keys():
                        weight = self.model.state_dict()[key_w]
                        w_mask = mask[key_w]
                        real_weight = (weight * w_mask)
                        clean_state_dict[key_w] = real_weight
                    key_b = prefix + ".bias"
                    if key_b in self.model.state_dict():
                        bias = self.model.state_dict()[key_b]
                        # 从 weight mask 推导出需要保留哪些 output 通道（第一维）
                        # 对于 Linear：mask shape = [out_features, in_features]
                        # 对于 Conv2d：mask shape = [out_channels, in_channels, k, k]
                        
                        dim = w_mask.dim()
                        if dim == 2:  # Linear
                            out_mask = (w_mask.abs().sum(dim=1) > 0).float()
                        elif dim == 4:  # Conv2d
                            out_mask = (w_mask.abs().sum(dim=(1, 2, 3)) > 0).float()
                        else:
                            raise ValueError(f"Unsupported weight dim: {dim}")
                        clean_state_dict[key_b] = bias * out_mask
                
                # prevent use more GPU
                clean_model.load_state_dict(clean_state_dict)
                # clean_model.to('cpu')
                for layer in clean_model.prunable_layers:
                    layer.mask.copy_(layer.state_dict()['weight'] != 0)

                clean_state_dict = clean_model.state_dict()
            list_state_dict.append(clean_state_dict)
        sub_model_time.append(timer() - start)
        

            

        return list_state_dict, sub_model_time


    @torch.no_grad()
    def test_split(self, start, model):
        list_state_dict = []
        sub_model_time = []
        acc_list, loss_list, density = [], [], []

        for mask in self.list_mask:
            clean_model = copy.deepcopy(model)
            clean_state_dict = clean_model.state_dict()
            for layer, prefix in zip(self.model.prunable_layers, self.model.prunable_layer_prefixes):
                # works for both layers
                key_w = prefix + ".weight"
                if key_w in self.model.state_dict().keys():
                    weight = self.model.state_dict()[key_w]
                    w_mask = mask[key_w]
                    real_weight = (weight * w_mask)
                    clean_state_dict[key_w] = real_weight
                # prevent use more GPU
            clean_model.load_state_dict(clean_state_dict)
            # clean_model.to('cpu')
            for layer in clean_model.prunable_layers:
                mask = layer.state_dict()['weight'] != 0
                layer.mask.copy_(mask)

            density.append(clean_model.density())
            if not torch.is_tensor(self.test_data):
                loss, acc = clean_model.evaluate(self.test_loader)
            else:
                # evaluate2 is put all test data on the GPU to accelerate the test process
                loss, acc = clean_model.evaluate2(self.test_data,self.test_label)
            acc_list.append(acc)
            loss_list.append(loss)
            clean_state_dict = clean_model.state_dict()
            list_state_dict.append(clean_state_dict)
            sub_model_time.append(timer() - start)

        return acc_list, loss_list, density, list_state_dict, sub_model_time

    def small_test_loader(self, subset_data_loader):
        return subset_data_loader

    def collate_to_device(self, test_loader):
        with torch.no_grad():
            # 初始化空列表来存储数据和标签
            all_data = []
            all_labels = []

            # 遍历测试加载器以收集所有数据和标签
            for data, labels in test_loader:
                all_data.append(data)
                all_labels.append(labels)

            # 使用torch.cat在0维度（批次维度）上合并所有数据和标签
            all_data = torch.cat(all_data)
            all_labels = torch.cat(all_labels)
            all_data = all_data.to(self.device)
            all_labels = all_labels.to(self.device)

        return all_data, all_labels
    def test_None_list_client_dict(self, list_sd):
        None_number = 0
        for i in range(len(list_sd)):
            sd = list_sd[i]
            if sd == None:
                None_number += 1
            else:
                if is_hetero_family_method(self.method):
                    idx, state_dict, payload_mask = unpack_client_state_payload(sd)
                    mask = payload_mask if payload_mask is not None else self.list_mask[i]
                    clean_state_dict = OrderedDict()
                    
                    for k, v in self.model.state_dict().items():
                        if 'weight' in k or 'bias' in k:
                            if 'weight' in k:
                                if v.dim() > 1:
                                    if tuple(state_dict[k].shape) == tuple(v.shape):
                                        clean_state_dict[k] = state_dict[k]
                                    else:
                                        mesh = torch.meshgrid(mask[k], indexing='ij')
                                        temp = torch.zeros_like(v)
                                        temp[mesh] = state_dict[k]
                                        clean_state_dict[k] = temp
                                else:
                                    if tuple(state_dict[k].shape) == tuple(v.shape):
                                        clean_state_dict[k] = state_dict[k]
                                    else:
                                        temp = torch.zeros_like(v)
                                        temp[mask[k]] = state_dict[k]
                                        clean_state_dict[k] = temp

                            else:
                                if tuple(state_dict[k].shape) == tuple(v.shape):
                                    clean_state_dict[k] = state_dict[k]
                                else:
                                    temp = torch.zeros_like(v)
                                    temp[mask[k]] = state_dict[k]
                                    clean_state_dict[k] = temp
                        else:
                            clean_state_dict[k] = state_dict[k]
                            
                                
                    sd[1] = clean_state_dict
                self.list_client_sd_buffer[i] = sd
                self.client_arrive_num[i] = self.client_arrive_num[i]

        if None_number == 10:
            return True
        else:
            return False

    def calc_model_params(self, model, display=False):
        sum_param_in_use = 0  # the sum of all used (model layers+bias)
        sum_all_param = 0
        for layer, layer_prefix in zip(model.prunable_layers, model.prunable_layer_prefixes):
            num_bias = layer.bias.numel() if hasattr(layer, "bias") and layer.bias is not None else 0
            layer_param_in_use = layer.mask.sum().int().item() + num_bias
            layer_all_param = layer.mask.nelement() + num_bias
            sum_param_in_use += layer_param_in_use
            sum_all_param += layer_all_param
            if display:
                print("\t{} remaining: {}/{} = {}".format(layer_prefix, layer_param_in_use, layer_all_param,
                                                          layer_param_in_use / layer_all_param))
        if display:
            print("\tTotal: {}/{} = {}".format(sum_param_in_use, sum_all_param, sum_param_in_use / sum_all_param))

        return sum_param_in_use / sum_all_param


    def Aggregate_model(self, list_num_proc, list_sd, idx, sgrad_to_upload):
        merge_map = {
            'buff_mask_fed_avg': 'buff_mask_fed_avg',
            'buff_fed_avg': 'buff_fed_avg',
            'fed_avg': 'fed_avg',
            'mask_fed_avg': 'mask_fed_avg',
            'heterofl': 'heterofl',
            'fed_asyn': 'fedasyn',
            'gradient_avg':'gradient_avg'
        }
        if self.merge in merge_map:
            self.aggregate_model(list_num_proc, list_sd, idx, sgrad_to_upload, mode=merge_map[self.merge])
        else:
            raise ValueError(f"Unknown merge method: {self.merge}")


    @torch.no_grad()
    def aggregate_model(self, list_num_proc, list_state_dict, idx, sgrd_to_upload, mode='fed_avg'):
        """
        通用模型聚合方法。
        支持：
        - 'fed_avg'        : Standard FedAvg
        - 'buff_fed_avg'   : FedAvg based on Buff
        - 'mask_fed_avg'      : 异步权重调整
        - 'buff_mask_fed_avg' : 使用缓冲区且掩码更新
        """
        self_sd = self.model.state_dict()
        total_coeff = 0
        asyn_coeff = 0
        coeffs = []

        list_state_dict_used = list_state_dict if (mode != 'buff_mask_fed_avg' or mode != 'buff_fed_avg') else self.list_client_sd_buffer

        client_partion = 0
        for i in range(len(list_state_dict_used)):
            if list_state_dict_used[i] != None:
                client_partion += self.list_coeff[i]


        sd = copy.deepcopy(self_sd)
        dict_keys = self_sd.keys()

        for key in dict_keys:
            if key.endswith("num_batches_tracked") or 'threshold' in key:
                continue
            sum_weight = torch.zeros_like(self_sd[key], device=self.device)
            sum_mask = torch.zeros_like(self_sd[key], device=self.device)
            sum_gradient = torch.zeros(size=self_sd[key].size(), device=self.device)
            sum_num_proc = 0
            i = 0
            for num_proc, state in zip(list_num_proc,list_state_dict_used):

                if state == None: continue
                idx, state_dict, _ = unpack_client_state_payload(state)

                if state_dict[key].is_sparse: state_dict[key] = state_dict[key].to_dense()
                
                sum_weight = sum_weight + self.list_coeff[i] * num_proc * state_dict[key].to(self.device)
                mask = (state_dict[key] != 0).to(self.device)
                sum_gradient[mask] = sum_gradient[mask] + self.list_coeff[i] * num_proc * (state_dict[key] - sd[key])[mask]
                sum_mask = sum_mask + self.list_coeff[i] * mask * num_proc
                sum_num_proc = sum_num_proc + self.list_coeff[i] * num_proc
                i = i + 1


            divisor = torch.where(sum_mask == 0, torch.tensor([1e-10], device=self.device), sum_mask)
            merged = torch.div(sum_weight, divisor)
            sum_weight = sum_weight.view(sd[key].size())
            mk = merged == 0
            merged[mk] = self_sd[key][mk]
            
            
            if mode == 'fedasyn':
                sd[key] = merged * client_partion + (1 - client_partion) * self_sd[key]
            elif mode == 'mask_fed_avg':
                sd[key] = merged * client_partion + (1 - client_partion) * self_sd[key]
            elif mode ==  'buff_mask_fed_avg' or mode == 'heterofl':
                sd[key][~mk] = merged[~mk]
            elif mode == 'gradient_avg':
                sd[key] = sd[key]+sum_gradient/sum_num_proc
            else:
                sd[key] = sum_weight/sum_num_proc
                

        self.model.load_state_dict(sd)
        if self.interval_signal:

            if self.accumulate == 'w':
                self.control.accumulate_w(self.model)
                
            elif self.accumulate == 'wg':
                self.control.accumulate_wg_square( self.old_model)
                self.old_model = copy.deepcopy(self.model.state_dict())
            elif self.accumulate == 'g':
                self.control.accumulate_g(sgrd_to_upload)
            else:
                if self.display:
                    self.logger.info('wrong accumulate')
                    
    def test_pruning_performance(self,avg_acc,fed_avg_model):
        if len(self.save_for_split) > 0 and avg_acc > self.save_for_split[0]:
            self.sp = self.save_for_split.popleft()
            self.save_display_data()
            self.test_split_statu = True
        if self.test_split_statu and avg_acc >= self.sp:
            self.save_checkpoint_for_split(self.sp, fed_avg_model)
            self.test_split_acc = avg_acc
            self.test_split_statu = False
            self.save_display_data()
        elif not self.test_split_statu and avg_acc >= self.sp and avg_acc < self.test_split_acc:
            self.save_checkpoint_for_split(self.sp, fed_avg_model)
            self.test_split_acc = avg_acc
            self.test_split_statu = False
            self.save_display_data()



    def load_the_run_data(self, list_time, average_round_time, interval_signal, train_number, idx, client_density):
        self.client_train_time, self.sum_client_train_time, self.server_merge_time, self.sum_server_merge_time, \
            self.client_upload_time, self.sum_client_upload_time, self.client_download_time, \
            self.sum_client_download_time, self.wast_time, = list_time

        self.average_round_time = average_round_time
        self.interval_signal = interval_signal
        self.train_number = train_number
        self.round = idx
        self.client_density = copy.deepcopy(client_density)

    def test_client_model(self,list_sd):
        if self.min_density < 0.99:
            test_client_interval = self.interval
        else:
            test_client_interval = self.interval * 5



        for i in range(len(self.client_density)):
            if self.need_client_acc and self.train_number[i] % test_client_interval == 1:

                if list_sd[i] != None:
                    if not torch.is_tensor(self.test_data):
                        client_loss, client_acc = self.client_list[i].model.evaluate(self.test_loader)
                    else:

                        client_loss, client_acc = self.client_list[i].model.evaluate2(self.test_data,self.test_label)
                else:
                    client_loss, client_acc = self.list_client_loss[i][-1], self.list_client_acc[i][-1]
                self.list_client_acc[i].append(client_acc)
                self.list_client_loss[i].append(client_loss)

                self.list_client_density[i].append(self.client_density[i])
                self.list_train_number[i].append(self.train_number[i])

    def Evaluate_Global_Model(self,fed_avg_model):
        if not torch.is_tensor(self.test_data):
            loss, acc = self.model.evaluate(self.test_loader)
        else:
            loss, acc = self.model.evaluate2(self.test_data, self.test_label)
        if self.min_density < 0.99:
            if not torch.is_tensor(self.test_data):
                avg_loss, avg_acc = fed_avg_model.evaluate(self.test_loader)
            else:
                avg_loss, avg_acc = fed_avg_model.evaluate2(self.test_data, self.test_label)
        else:
            avg_loss = loss
            avg_acc = acc

        self.list_loss.append(loss)
        self.list_acc.append(acc)
        self.fed_avg_acc.append(avg_acc)
        self.fed_avg_loss.append(avg_loss)

    def GMR(self,list_optimizer):
        # if early_stoping.num == 1, we only use the global acc as the input to ESM
        # else every client and the server all has the ESM
        if self.early_stoping.num == 1:
            acc = [self.fed_avg_acc[-1]]
        else:
            # acc = [self.list_acc[-1]] + [client_acc[-1] for client_acc in self.list_client_acc]
            acc = [self.fed_avg_acc[-1]] + [client_acc[-1] for client_acc in self.list_client_acc]

        # increase the patience for final convergence
        if (not self.recover) or (self.min_density == 1):
            if self.early_stoping.patience[0] != min(self.patience * 5, 50):
                self.logger.info('Enter the end stage, increase the patience to obtain the best acc')
                self.early_stoping.patience[0] = min(self.patience * 5, 50)
                
        # Early stopping mechanism
        state = self.early_stoping(acc, self.logger)

        # To prevent frequent I/O operations from slowing down the process,
        # saving is only allowed after the code has reached a certain stage.
        if not self.begin_save:
            if self.early_stoping.counter[0] >= self.patience:
                self.begin_save = True
                self.early_stoping.counter[0] = 0
                state[0] = None

        # if state[0] == True, which means the global model improved during the training
        # I store the best model for restoration
        if state[0] == True and self.begin_save:
            self.list_optimizer = list_optimizer
            # self.save_checkpoint()
            self.save_display_data()
        # if state[0] == False, which means the global model improved during the training
        elif state[0] == False and self.begin_save:
            # if the global model cannot improve during the training, restore all clients model
            # try:
            #     self.logger.info('it is not the best model')

            # except FileNotFoundError:
            #     print('The checkpoint file was not found')
            self.save_display_data()
            if self.recover:
                if self.min_density == 1.0:
                    self.early_stop = True
                else:
                    recover_mode = self.get_recover_mode()
                    if recover_mode == 'ladder':
                        self.min_density = self.get_ladder_next_density(self.min_density)
                    elif recover_mode == 'fixed':
                        self.min_density = self.get_fixed_next_density(self.min_density)
                    elif self.increase >= 1.0:
                        self.min_density = self.get_ladder_next_density(self.min_density)
                    elif self.min_density + self.increse < 1.0:
                        self.min_density = self.min_density + self.increse
                    else:
                        self.early_stoping.patience[0] = self.early_stoping.patience[0] * 5
                        self.min_density = 1.0
                    assert self.min_density <= 1.0
                    
                    self.wait_for_stable = self.early_stoping.patience[0]
                    for j in range(len( self.early_stoping.counter)):
                        self.early_stoping.counter[j] = 0
                        self.early_stoping.best_score[j] = -1
                        self.early_stoping.early_stop[j] = False
                        

            else:
                self.early_stop = True





        # Change the client density based on client accuracy
        if self.recover and self.begin_save:
            # because the index 0 is the server, so begin from the 1
            if len(self.early_stoping.counter) > 1:
                for i in range(1, self.early_stoping.num):
                    client_id = i - 1

                    if self.client_wait_for_stable[client_id] > 0:
                        continue

                    if state[i] == False:
                        client_recovered = False
                        self.logger.info(f'client{client_id} is out of patience')
                        recover_mode = self.get_recover_mode()
                        if recover_mode == 'ladder':
                            current_density = self.client_density[client_id]
                            self.client_density[client_id] = self.get_ladder_next_density(current_density)
                            client_recovered = True
                            assert self.client_density[client_id] <= 1.0
                        elif recover_mode == 'fixed':
                            current_density = self.client_density[client_id]
                            next_density = self.get_fixed_next_density(current_density)
                            if next_density > current_density:
                                self.client_density[client_id] = next_density
                                client_recovered = True
                        elif self.increase >= 1.0:
                            current_density = self.client_density[client_id]
                            self.client_density[client_id] = self.get_ladder_next_density(current_density)
                            client_recovered = True
                        
                            assert self.client_density[client_id] <= 1.0
                            
                        elif self.client_density[client_id] + min(self.client_density[client_id]*0.7, 0.20) < 1.0:
                            
                            self.client_density[client_id] = self.client_density[client_id] + min(self.client_density[client_id], 0.20)
                            client_recovered = True
                        if client_recovered:
                            self.logger.info(
                                f'for client{client_id}, increase model client, client density {self.client_density[client_id]}')
                            self.client_wait_for_stable[client_id] = self.get_client_wait_for_stable_steps()
                            self.early_stoping.best_score[0] = None
                            self.early_stoping.early_stop[0] = False
                            self.early_stoping.best_score[i] = None
                            self.early_stoping.early_stop[i] = False
                            self.save_display_data()
            # update the min_density
        


        for i in range(len(self.client_density)):
            if self.client_density[i] < self.min_density:
                self.client_density[i] = self.min_density
                
                if len(self.early_stoping.counter) > 1:
                    self.early_stoping.counter[0] = 0
                    self.early_stoping.best_score[0] = -1
                    self.early_stoping.early_stop[0] = False
                    self.early_stoping.counter[i+1] = 0
                    self.early_stoping.best_score[i+1] = -1
                    self.early_stoping.early_stop[i+1] = False
                    



    def get_server_to_client_metadata(self, list_state_dict):
        sorted_clientdensity, sorted_clientdensity_indics = torch.sort(
            torch.tensor(self.client_density), descending=False)
        sorted_clientdensity = sorted_clientdensity.tolist()
        for i in range(len(sorted_clientdensity)):
            if i == 0:
                self.sub_density[i] = sorted_clientdensity[i]
            else:
                self.sub_density[i] = sorted_clientdensity[i] - sorted_clientdensity[i - 1]
                
        self.client_model_size = self.get_real_size(self.process_state_dict_to_client(list_state_dict),
                                                    self.experiment_name, self.client_density)

        return self.sub_density,self.client_model_size

    def Print_FL_Message(self,idx, time_download):
        print("Round #{} (Experiment = {}， LR = {}).".format(idx, self.experiment_name,
                                                             self.ip_optimizer_wrapper.get_last_lr()))
        print("Elapsed time = {}".format(self.time[-1]))
        print("fed_avg Loss/acc (at round #{}) = {}/{}   Loss/acc={}/{}".format(self.round, self.fed_avg_loss[-1],
                                                                                self.fed_avg_acc[-1],
                                                                                self.list_loss[-1],
                                                                                self.list_acc[-1]))
        print('the density is ' + ",".join([f"{client_density:.2f}" for client_density in self.client_density]))
        print('self.train_number :' + str(self.train_number))
        print('sum_server_upload :' + str(self.sum_server_upload[-1]) + '_' + str(time_download))
        self.logger.info("Round #{} (Experiment = {}).".format(idx, self.experiment_name))
        self.logger.info("Elapsed time = {}".format(self.time[-1]))
        self.logger.info(
            "fed_avg Loss/acc (at round #{}) = {}/{}   Loss/acc={}/{}".format(self.round, self.fed_avg_loss[-1],
                                                                              self.fed_avg_acc[-1], self.list_loss[-1],
                                                                              self.list_acc[-1]))
        self.logger.info(
            'the density is ' + ",".join([f"{client_density:.2f}" for client_density in self.client_density]))
        self.logger.info('self.train_number :' + str(self.train_number))
        self.logger.info('sum_server_upload :' + str(self.sum_server_upload[-1]) + '_' + str(time_download))

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

    def reshuffle_params(self):
        for k, v in self.model.state_dict().items():
            if 'weight' in k and v.dim() > 1:
                output_size = v.size(0)
                self.model_idxs[k] = torch.randperm(output_size, device=v.device)
                self.roll_idx[k] = 0
        return self.model_idxs
    
    

    
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
            D = self.config.BASE_EMB // self.config.NUM_HEADS 
            BASE_EMB = self.config.BASE_EMB      # 例如 256
            client_scaler_rate  = {}
            client_local_dim    = {}
            idx_i = {}  # 当前通道索引

            for client_id in range(num_clients):
                density_i = density[client_id]

                # 1) sqrt(density) + 量化
                base_scaler  = density_i ** 0.5
                scaler_steps = int(round(base_scaler * D))
                scaler_steps = max(1, min(D, scaler_steps))
                scaler_rate  = scaler_steps / D

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
                                    legacy_static_fedrolex = Roll and self.config.EXP_NAME in ("FEMNIST", "CIFAR10")
                                    if legacy_static_fedrolex:
                                        output_idx_i_m = torch.arange(out_channels, device=param.device)[:local_output_size]
                                    else:
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



    def main(self, idx, list_sd, list_num_proc, current_time,client_density, list_optimizer, train_number,
             interval_signal, average_round_time, sgrad_to_upload, list_time, sum_server_upload, sum_server_download,
             time_download):
        self.load_the_run_data(list_time, average_round_time, interval_signal, train_number, idx, client_density)
        warmup_state_changed = False

        # if no client arrived, return to client train directly
        sum_time = 0
        if self.test_None_list_client_dict(list_sd) and not interval_signal:
            return self.list_state_dict, self.model_idx, sum_time, self.list_client_loss, self.sub_density, self.increment_model_size, self.client_model_size, self.client_density
        #calculate the staleness of different clients
        self.calculate_staleness(list_sd)
        server_start = timer()


        #  to complete the fed avg
        self.Aggregate_model(list_num_proc, list_sd, idx, sgrad_to_upload)
        if self.need_client_acc:
            self.test_client_model(list_sd)

        start = timer()
        
        if interval_signal:
            old_client_desity = copy.deepcopy(self.client_density)
            warmup_state_changed = self.update_prune_warmup_state(idx)
            # record the metadata
            self.sum_server_upload.append(sum_server_upload)
            for i in range(len(self.client_density)):
                self.sum_server_download[i].append(sum_server_download[i])
            self.time.append(current_time)

            # test the global model performance
            if self.min_density > 0.99:
                fed_avg_model = self.model
            else:
                fed_avg_model = self.fed_avg_model(list_num_proc, list_sd)

            self.Evaluate_Global_Model(fed_avg_model)
            self.test_pruning_performance(self.fed_avg_loss[-1], fed_avg_model)
            self.Print_FL_Message(idx, time_download)

            # GMR
            # Only update the client mask at fixed intervals
            # other rounds use the old client mask directly
            self.tick_client_wait_for_stable()

            # Try to alter the client density
            if self.wait_for_stable <= 0 and not self.prune_warmup_active:
                self.GMR(list_optimizer)
                self.wait_for_stable = self.wait_for_stable - 1

            # if client_density has been changed, use adjust function to adjust the client model
            if (self.recover or warmup_state_changed) and old_client_desity != self.client_density:
                print(old_client_desity)
                print(self.client_density)
                self.logger.info('Adjust model'+': New density list is ' + str(self.client_density))
                print('Adjust model' + ': New density list is ' + str(self.client_density))
   
                
        if interval_signal and is_hetero_family_method(self.method):
            use_roll = is_fedrolex_method(self.method)
            self.list_mask = self.roll_split_universal(idx, self.model, self.client_density, Roll=use_roll)
            self.log_first_layer_channel_bounds()
            
            
        
        # if interval_signal:
        #     if not (self.method.lower().startswith('heterofl') or self.method.lower().startswith('fjord') or 
        #             self.method.lower().startswith('fedrolex')):

        #         list_state_dict, self.model_idx, sub_model_time, self.list_mask, _, self.increment_state_dict, _, self.list_threshold = self.control.sub_adjust_fast(
        #                         client_density=self.client_density, use_coff=self.use_coeff,
        #                         min_density=self.config.min_density, )
                
        
        # based on self.list_mask to get the corresponding model
        list_state_dict, sub_model_time = self.split_model(start)        

        if interval_signal:
            self.increment_model_size = self.get_real_size(
                self.process_state_dict_to_client(self.increment_state_dict), self.experiment_name, self.sub_density)
            
            self.sub_density, self.client_model_size = self.get_server_to_client_metadata(list_state_dict)


        # add extra idx to list_state_dict for Asynchronous Federated Learning
        for i in range(len(list_state_dict)):
            list_state_dict[i] = [idx, list_state_dict[i]]
        # print('list_threshold'+str(self.list_threshold))

        sum_time += timer() - server_start
        
        
                
        return list_state_dict, self.model_idx, sum_time, self.list_client_loss, self.sub_density, self.increment_model_size, self.client_model_size, self.client_density




class FedMapClient:
    def __init__(self, model, config, use_adaptive, exp_config, args, device):
        self.args = args
        self.method = args.ex
        self.config = config
        self.use_adaptive = use_adaptive
        self.model = deepcopy(model)
        self.model.train()
        self.optimizer = None
        self.optimizer_scheduler = None
        self.optimizer_wrapper = None
        self.train_loader = None
        self.client_is_sparse = False
        self.exp_config = exp_config
        self.lr_scheduler = None
        if self.exp_config.lr_scheduler_class is not None:
            self.lr_scheduler = self.exp_config.lr_scheduler_class(optimizer=self.optimizer,
                                                                   **self.exp_config.lr_scheduler_params)
        self.list_mask = [None for _ in range(len(self.model.prunable_layers))]
        self.is_sparse = False
        self.terminate = False
        self.device = device
        self.mask_dict = {}
        self.test_loader = None
        self.scheduler = None
        self.model.to(self.device)
        self.num = 0
        self.use_lr_mask = False
        self.weight_decay = None
        self.client_idx = -1
        self.accumulated_sgrad = dict()

    @abstractmethod
    def init_optimizer(self, *args, **kwargs):
        pass

    @abstractmethod
    def init_train_loader(self, *args, **kwargs):
        pass

    # @abstractmethod
    # def init_test_loader(self, tl):
    #     pass







    def cleanup_state_dict_to_server(self, sparse_model) -> dict:
        """
        Clean up state dict before process, e.g. remove entries, transpose.
        To be overridden by subclasses.
        """
        clean_state_dict = sparse_model.state_dict()  # not deepcopy

        for layer, prefix in zip(sparse_model.param_layers, sparse_model.param_layer_prefixes):
            key = prefix + ".bias"
            if isinstance(layer, SparseLinear) and key in clean_state_dict.keys():
                clean_state_dict[key] = clean_state_dict[key].view(-1)

        del_list = []
        del_suffix = "placeholder"
        for key in clean_state_dict.keys():
            # clean_state_dict[key] = clean_state_dict[key].cpu()
            if key.endswith(del_suffix):
                del_list.append(key)

        for del_key in del_list:
            del clean_state_dict[del_key]

        return clean_state_dict


    
    
    
    @torch.no_grad()
    def process_state_dict_to_server(self, state_dict):

        clean_state_dict = copy.deepcopy(state_dict)

        with torch.no_grad():
            for _, prefix in zip(self.model.prunable_layers, self.model.prunable_layer_prefixes):
                # works for both layers
                key_w = prefix + ".weight"
                if key_w in state_dict.keys():
                    weight = clean_state_dict[key_w]
                    sparse_weight = weight.view(weight.size(0), -1).to_sparse()
                if sparse_weight._nnz() == 0:
                    sparse_weight = None
                clean_state_dict[key_w] = sparse_weight

        return clean_state_dict


    def load_mask(self, masks):
        self.list_mask = masks

    def check_client_to_sparse(self):  # if model.density() <= config.TO_SPARSE_THR ,set the model to sparse
        if not self.client_is_sparse and self.model.density() <= self.config.TO_SPARSE_THR:
            self.client_is_sparse = True

    def calc_model_params(self, display=False):
        sum_param_in_use = 0  # the sum of all used (model layers+bias)
        sum_all_param = 0
        for layer, layer_prefix in zip(self.model.prunable_layers, self.model.prunable_layer_prefixes):
            num_bias = layer.bias.numel() if hasattr(layer, "bias") and layer.bias is not None else 0
            layer_param_in_use = layer.mask.sum().int().item() + num_bias
            layer_all_param = layer.mask.nelement() + num_bias
            sum_param_in_use += layer_param_in_use
            sum_all_param += layer_all_param
            if display:
                print("\t{} remaining: {}/{} = {}".format(layer_prefix, layer_param_in_use, layer_all_param,
                                                          layer_param_in_use / layer_all_param))
        if display:
            print("\tTotal: {}/{} = {}".format(sum_param_in_use, sum_all_param, sum_param_in_use / sum_all_param))
        if sum_all_param == 0: sum_all_param = 1
        return sum_param_in_use / sum_all_param

    @torch.no_grad()
    def load_state_dict(self, idx_state_dict):
        idx, state_dict, _ = unpack_client_state_payload(idx_state_dict)
        self.client_idx = idx
        self.model.load_state_dict(state_dict)
        param_dict = dict(self.model.named_parameters())
        buffer_dict = dict(self.model.named_buffers())

        for key, param in {**param_dict, **buffer_dict}.items():
            if key in state_dict.keys():

                if state_dict[key].size() != param.size():
                    param.copy_(state_dict[key].view(param.size()))
                else:
                    param.copy_(state_dict[key])
        for key, sd in self.model.state_dict().items():
            mask = sd != 0
            self.mask_dict[key] = mask

        for layer in self.model.prunable_layers:
            mask = layer.state_dict()['weight'] != 0
            layer.mask.copy_(mask)


    def load_lr(self,lr):
        self.optimizer_wrapper.set_lr(lr)


    def train_model(self):
        num_proc_data = 0
        list_grad, loss = None, None

        if self.args.client_model_norm:
            old_model = copy.deepcopy(self.model)
        for i in range(self.config.NUM_LOCAL_UPDATES):
            inputs, labels = self.train_loader.get_next_batch()
            if self.args.client_model_norm:
                list_grad, loss = self.optimizer_wrapper.step2(inputs.to(self.device), labels.to(self.device),
                                                               old_model, self.args.mu)
            else:
                list_grad, loss = self.optimizer_wrapper.step(inputs.to(self.device), labels.to(self.device))
            
            if self.config.EXP_NAME == "ImageNet100":
            # if True:
                param_dict = dict(self.model.named_parameters())
                buffer_dict = dict(self.model.named_buffers())
                
                with torch.no_grad():  # ✅ 安全更新参数和缓冲区
                    for key, param in {**param_dict, **buffer_dict}.items():
                        if key in self.mask_dict:
                            mask = self.mask_dict[key]
                            if mask.size() != param.size():
                                param.copy_((param * mask).view_as(param))
                            else:
                                param.copy_(param * mask)

            num_proc_data += len(inputs)



        return num_proc_data, loss, list_grad

    def main(self, idx, logger):
        self.model.train()
        model_density = round(self.calc_model_params(),4)
        num_proc_data, loss, list_grad = self.train_model()
        accumulated_grad = dict()
        lr = self.optimizer_wrapper.optimizer.param_groups[0]['lr']
        state_dict = self.model.state_dict() 
        state_dict_to_server = self.process_state_dict_to_server(state_dict)
        return state_dict, num_proc_data, lr, model_density, state_dict_to_server, self.client_idx + 1, accumulated_grad


class FedMapFL(ABC):
    def __init__(self, args, config, server, client_list):
        self.config = config

        # self.use_adaptive = True

        self.max_round = config.MAX_ROUND
        self.server = server
        self.client_list = client_list
        self.server.client_list = client_list
        if getattr(self.server, "list_mask", None) is not None:
            self.client_masks_in_use = [copy.deepcopy(mask) for mask in self.server.list_mask]
        else:
            self.client_masks_in_use = [None] * len(client_list)
        self.list_store_mask_client = [None] * len(client_list)

        self.increase = args.increase
        self.interval = args.interval
        self.min_density = args.min_density
        self.resume = args.resume
        self.args = args
        self.threshold = 10000
        self.pre_threshold = 0
        self.chronous = args.chronous

        self.list_loss, self.list_acc, self.list_est_time, self.list_model_size = [0], [0], [0], [0]

        self.average_download_speed = args.average_download_speed
        self.average_upload_speed = args.average_upload_speed
        self.variance_download = 0.3

        self.variance_upload = 0.3
        self.average_server_down = 50
        self.variance_server_down = 0.1
        self.average_server_up = args.server_up_speed
        self.variance_server_up = 0.1
        self.comm_rng = np.random.RandomState(args.seed)
        # self.computerlr = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        self.client_density = args.client_density
        self.client_state = [True] * len(client_list)
        self.size_client_need_upload = [0] * len(client_list)
        self.client_need_download = [0] * len(client_list)
        self.list_num = [0] * len(client_list)
        # from sklearn.cluster import DBSCAN
        self.train_number = [0] * len(client_list)
        
        self.client_start_work_time = [0] * len(client_list)
        self.list_client_sd = [None for _ in range(len(self.client_density))]
        self.last_round_time = [0] * len(client_list)
        self.per_round_time = [[] for i in range(len(self.client_density))]
        self.client_train_time = [[0.0] for i in range(len(self.client_density))]
        self.sum_client_train_time = [[0.0] for i in range(len(self.client_density))]
        self.server_merge_time = [0.0]
        self.sum_server_merge_time = [0.0]
        self.client_upload_time = [[0.0] for i in range(len(self.client_density))]
        self.sum_client_upload_time = [[0.0] for i in range(len(self.client_density))]
        self.client_download_time = [[0.0] for i in range(len(self.client_density))]
        self.sum_client_download_time = [[0.0] for i in range(len(self.client_density))]
        self.waste_time = [0] * len(self.client_list)
        self.list_time = [self.client_train_time, self.sum_client_train_time, self.server_merge_time,
                          self.sum_server_merge_time,
                          self.client_upload_time, self.sum_client_upload_time, self.client_download_time,
                          self.sum_client_download_time,
                          self.waste_time
                          ]

        self.Residual = args.Res
        self.start_client_idx = len(self.client_list)

        self.interval_signal = False
        self.previous_idx = -1

        self.communicate_time_from_server_to_client = [0] * len(self.client_list)

        self.standard_time = 0
        self.client_model_size = None
        self.last_train_num = [0] * len(self.client_list)
        self.average_round_time = [deque(maxlen=self.interval // 2) for _ in range(len(self.client_list))]

        self.sum_server_upload = 0
        self.sum_server_download = [0] * len(self.client_list)
        self.sum_time_download = 0
        self.time_server_up_complete = 0
        self.server_upload = [0] * len(self.client_list)
        self.train_num = 0

        self.standard_time = 0
        self.one_round_time = 0


        ds_path = os.path.join("results", 'density_size.pt')
        if os.path.exists(ds_path):
            self.density_size = load(ds_path)
        else:
            self.density_size = {}
            mkdir_save(self.density_size, ds_path)
            
        self.server_to_client = [0] * len(self.client_list) 
        # ========== Federated learning control ==========
        self.max_round = config.MAX_ROUND  # Maximum training rounds
        self.chronous = args.chronous      # Synchronous or asynchronous FL
        self.resume = args.resume          # Resume training from checkpoint
        self.increase = args.increase      # Increase density strategy
        self.interval = args.interval      # Interval for scheduling
        self.min_density = args.min_density  # Minimum client model density
        self.Residual = args.Res           # Residual parameter for pruning
        self.start_client_idx = len(self.client_list)
        self.train_num = 0
        self.standard_time = 0
        self.one_round_time = 0

        # ========== Client status and density tracking ==========
        self.client_density = args.client_density  # Client-side model density list
        self.client_state = [True] * len(self.client_list)  # Whether the client is active
        self.list_client_sd = [None] * len(self.client_list)  # State dict for each client
        self.client_model_size = None  # Size of client model
        self.last_train_num = [0] * len(self.client_list)  # Samples trained in the previous round

        # ========== Training statistics ==========
        self.list_loss = [0]         # Global loss per round
        self.list_acc = [0]          # Global accuracy per round
        self.list_est_time = [0]     # Estimated time per round
        self.list_model_size = [0]   # Model size per round
        self.list_num = [0] * len(self.client_list)  # Training data per client
        self.train_number = [0] * len(self.client_list)  # Cumulative training samples
        self.last_round_time = [0] * len(self.client_list)  # Last communication time per client
        self.per_round_time = [[] for _ in range(len(self.client_list))]  # Round-wise time tracking
        self.average_round_time = [deque(maxlen=self.interval // 2) for _ in range(len(self.client_list))]  # Moving average

        # ========== Time statistics ==========
        self.client_start_work_time = [0] * len(self.client_list)  # Start time per client
        self.client_train_time = [[0.0] for _ in range(len(self.client_list))]  # Local training time
        self.server_merge_time = [0.0]  # Server aggregation time
        self.client_upload_time = [[0.0] for _ in range(len(self.client_list))]  # Upload time
        self.client_download_time = [[0.0] for _ in range(len(self.client_list))]  # Download time
        self.waste_time = [0] * len(self.client_list)  # Idle/waiting time
        self.list_statu_time = [  # For easy access by component
            self.client_train_time,
            self.server_merge_time,
            self.client_upload_time,
            self.client_download_time,
            self.waste_time
        ]

        # ========== Communication statistics ==========
        self.size_client_need_upload = [0] * len(self.client_list)  # Upload size for each client
        self.server_to_client = [0] * len(self.client_list)  # Model size downloaded from server
        self.communicate_time_from_server_to_client = [0] * len(self.client_list)  # Server-to-client communication time
        self.sum_server_download = [0] * len(self.client_list)  # Total download size per client
        self.server_upload = [0] * len(self.client_list)  # Total upload size per client
        self.sum_time_download = 0
        self.sum_server_upload = 0
        self.server_up_complete = 0  # How many clients completed upload
        

        # ========== Network speed and bandwidth simulation ==========
        self.average_download_speed = args.average_download_speed  # Client download speed
        self.average_upload_speed = args.average_upload_speed      # Client upload speed
        self.variance_download = 0.3                               # Download speed variance
        self.variance_upload = 0.3                                 # Upload speed variance
        self.average_server_down = 50                              # Server download speed
        self.variance_server_down = 0.1
        self.average_server_up = args.server_up_speed              # Server upload speed
        self.variance_server_up = 0.1

        # ========== Adaptive control and signal flags ==========
        self.threshold = 10000          # Global update threshold
        self.pre_threshold = 0          # Threshold of previous round
        self.interval_signal = False    # Whether to trigger interval update
        self.previous_idx = -1          # Last triggered client index

        # ========== Density-to-size mapping ==========
        ds_path = os.path.join("results", "density_size.pt")
        if os.path.exists(ds_path):
            self.density_size = load(ds_path)
        else:
            self.density_size = {}
            mkdir_save(self.density_size, ds_path)



    def accumulate_dict(self, accumulate_dict, sgrd, sum_mask):

        for key in sgrd.keys():
            if key in accumulate_dict.keys():
                accumulate_dict[key] += sgrd[key]
                sum_mask[key] += sgrd[key] != 0
            else:
                accumulate_dict[key] = sgrd[key]
                assert key not in sum_mask.keys()
                sum_mask[key] = sgrd[key] != 0

    def zeroed_state_dict(self, state_dict):
        zeroed_dict = {}
        for key, value in state_dict.items():
            if isinstance(value, torch.Tensor):
                zeroed_dict[key] = torch.zeros_like(value)
            else:
                zeroed_dict[key] = self.zeroed_state_dict(value)

        return zeroed_dict

    def get_internet_speed(self):

        download_speed, upload_speed = [], []

        server_up = 0
        server_down = 0
        n = 20
        for i in range(n):
            server_up += self.comm_rng.lognormal(mean=0, sigma=self.variance_server_up) * self.average_server_up
            server_down += self.comm_rng.lognormal(mean=0, sigma=self.variance_server_down) * self.average_server_down
        server_up = server_up / n
        server_down = server_down / n

        for i in range(len(self.average_upload_speed)):
            dp = 0
            up = 0
            for j in range(n):
                dp += self.comm_rng.lognormal(mean=0, sigma=self.variance_upload) * self.average_download_speed[i]
                up += self.comm_rng.lognormal(mean=0, sigma=self.variance_download) * self.average_upload_speed[i]

            dp = dp / n
            up = up / n

            download_speed.append(dp)
            upload_speed.append(up)

        if self.config.EXP_NAME == "CelebA":
            server_up = server_up * 100
            server_down = server_down * 100

        return server_up, server_down, download_speed, upload_speed

    def process_list(self, input_list):
        # create a dict to record all position of every value.
        # 创建一个字典来记录每个值出现的所有位置
        index_dict = {}
        for index, value in enumerate(input_list):
            if value not in index_dict:
                index_dict[value] = []
            index_dict[value].append(index)
        # 获取没有重复的值并排序
        unique_sorted_values = sorted(index_dict.keys())
        # 根据排序后的值获取原始下标
        indices_lists = [index_dict[value] for value in unique_sorted_values]
        return unique_sorted_values, indices_lists

    def get_real_size(self, list_state_dict, exp, density):
        list_model_size = []
        for i in range(len(list_state_dict)):
            if list_state_dict[i] == None or density[i] == 0:
                list_model_size.append(0)
                continue

            file_size = self.config.model_size * density[i]
            list_model_size.append(file_size)

        return list_model_size

    def train_client(self,idx,logger):
        # test the which clients are free
        for i in range(len(self.client_list)):
            if self.client_state[i] == False:
                assert self.server_to_client[i] > 0 or self.size_client_need_upload[i] > 0
                assert self.server_to_client[i] == 0 or self.size_client_need_upload[i] == 0
                # it means the old model is not sent to the server or the client do not get the new model.
            else:
                self.client_state[i] = True

        list_state_dict, list_last_lr, list_sparse_model, density = [], [], [], []
        use_legacy_imagenet100_sync_payload = (
            self.config.EXP_NAME == "ImageNet100"
            and self.chronous == "syn"
            and (is_heterofl_method(self.server.method) or is_fedrolex_method(self.server.method))
        )

        # train the free clients
        for i in range(len(self.client_list)):
            if self.client_state[i]:

                client = self.client_list[i]
                sd, npc, last_lr, ds, sparse_model, client_idx, sgrd = client.main(idx, logger)
                self.client_train_time[i].append(self.config.asyn_interval)
                self.sum_client_train_time[i].append(
                    self.sum_client_train_time[i][-1] + self.client_train_time[i][-1])
                if is_hetero_family_method(self.server.method) and not use_legacy_imagenet100_sync_payload:
                    payload_mask = copy.deepcopy(self.client_masks_in_use[i])
                    self.list_store_model_server[i] = [client_idx, sd, payload_mask]
                    list_state_dict.append([client_idx, sd, payload_mask])
                else:
                    self.list_store_model_server[i] = [client_idx, sd]
                    list_state_dict.append([client_idx, sd])
                self.train_number[i] += 1
                self.list_num[i] = npc  # Amount of data for client-side training models
            else:
                sd, npc, last_lr, ds, sparse_model, client_time, sgrd = None, 0, None, None, None, 0, None
                list_state_dict.append(None)
            # when the old client model arrives at the server, it need npc to calculate the coeff,
            list_last_lr.append(last_lr)
            density.append(ds)
            list_sparse_model.append(sparse_model)
            
        return list_state_dict, list_last_lr, list_sparse_model, density




    
    def simulate_client_to_server(self, list_sparse_model, density, begin_time_client_upload, list_state_dict):
        '''
        Simulates the client-to-server communication process, including upload bandwidth,
        client transmission scheduling, and server-side aggregation timing.
        
        Note: 
            - The server has higher bandwidth than clients.
            - For Async-FL (FedAsync), only one client is aggregated at a time.
        '''

        # Step 1: Compute each client's model size
        # print('client_density'+str(density))
        model_size = self.get_real_size(list_sparse_model, self.server.experiment_name, density)
        # Accumulate total server download traffic
        for i in range(len(self.client_list)):
            self.sum_server_download[i] += model_size[i]
        self.server.list_model_size.append(model_size)

        # Step 2: Network speed simulation
        server_up, server_down, download_speed, upload_speed = self.get_internet_speed()
 
        self.size_client_need_upload = [cs + ms for cs, ms in zip(self.size_client_need_upload, model_size)]

        # Step 3: Calculate transmission start time
        time_client_begin_transmission = []
        self.communicate_time_from_server_to_client
        for i in range(len(self.client_state)):
            assert self.server_to_client[i] > 0 or self.size_client_need_upload[i] > 0
            assert self.server_to_client[i] == 0 or self.size_client_need_upload[i] == 0

            if self.client_state[i]:
                    time_client_begin_transmission.append(
                        self.communicate_time_from_server_to_client[i] + self.client_train_time[i][-1])
                    begin_time_client_upload[i] = time_client_begin_transmission[i]
            else:
                # Approximate start time for inactive clients
                time_client_begin_transmission.append(self.communicate_time_from_server_to_client[i])

        # Step 4: Simulate when server receives client models
        from control.sub_algorithm import simulate_client_to_server as cts
        server_receive_time = np.array(
            cts(time_client_begin_transmission,
                                    copy.copy(self.size_client_need_upload),
                                    upload_speed,
                                    server_down)
        )
        
        server_close_time = copy.copy(server_receive_time)

        # Step 5: Decide server aggregation order
        if self.server.merge == 'fedasyn':
            max_true_client = 1  # Only the first valid client is used
            if self.start_client_idx == len(server_close_time):
                server_download_sequence = list(range(len(server_close_time)))
            else:
                server_download_sequence = list(range(self.start_client_idx, len(server_close_time))) + \
                                        list(range(0, self.start_client_idx))
        else:
            max_true_client = len(server_close_time)
            copy_time_client_begin_transmission = [
                0 if not self.client_state[i] else time_client_begin_transmission[i]
                for i in range(len(time_client_begin_transmission))
            ]
            server_download_sequence = [
                idx for idx, _ in sorted(enumerate(copy_time_client_begin_transmission), key=lambda x: x[1])
            ]

        # Step 6: Simulate server receiving client models
        for i in server_download_sequence:
            active = self.client_state[i]
            time_begin = time_client_begin_transmission[i]
            size_left = self.size_client_need_upload[i]

            if active:
                if server_close_time[i] > self.threshold:
                    # Upload exceeds time threshold → interrupt
                    self.client_state[i] = False
                    self.size_client_need_upload[i] = size_left - upload_speed[i] * (self.threshold - time_begin)
                    self.size_client_need_upload[i] = max(self.size_client_need_upload[i], 0.001)
                    server_close_time[i] = self.threshold
                else:
                    if max_true_client > 0:
                        self.size_client_need_upload[i] = 0
                        self.client_upload_time[i].append(server_close_time[i] - begin_time_client_upload[i])
                        max_true_client -= 1
                    else:
                        self.size_client_need_upload[i] = size_left - upload_speed[i] * (self.threshold - time_begin)
                        self.size_client_need_upload[i] = max(self.size_client_need_upload[i], 0.001)
                        self.client_state[i] = False
            else:
                assert self.server_to_client[i] > 0 or self.size_client_need_upload[i] > 0
                assert self.server_to_client[i] == 0 or self.size_client_need_upload[i] == 0

                if size_left > 0:
                    if server_close_time[i] > self.threshold:
                        self.size_client_need_upload[i] = size_left - upload_speed[i] * (self.threshold - time_begin)
                        self.size_client_need_upload[i] = max(self.size_client_need_upload[i], 0.001)
                        server_close_time[i] = self.threshold
                    else:
                        assert self.server_to_client[i] == 0
                        assert list_state_dict[i] is None
                        if max_true_client > 0:
                            self.size_client_need_upload[i] = 0
                            list_state_dict[i] = self.list_store_model_server[i]
                            self.client_state[i] = True
                            self.client_upload_time[i].append(server_close_time[i] - begin_time_client_upload[i])
                            self.start_client_idx = i + 1
                            max_true_client -= 1
                        else:
                            self.size_client_need_upload[i] = 0.001
                            self.client_state[i] = False
                            server_close_time[i] = self.threshold

                if self.server_to_client[i] > 0:
                    server_receive_time[i] = self.threshold + 1000  # Defer future use

        return list_state_dict,server_up,download_speed,upload_speed,server_close_time,server_receive_time,begin_time_client_upload
    
    def update_time_tracking(self, idx, list_state_dict, begin_time_client_upload, standard_client, server_close_time, server_receive_time, download_speed, upload_speed):
        """
        Update time-related statistics after one FL round.

        Args:
            idx: current FL round index
            list_state_dict: list of client model state_dicts
            begin_time_client_upload: list of upload start times
            standard_client_number: expected #clients per round (for async FL)
            server_close_time: list of times when server finishes receiving from each client
        """

        if idx < 2:
            self.standard_time = max(server_close_time)
            return

        if self.chronous.lower().startswith('syn'):
            # Synchronous FL
            for i in range(len(self.client_density)):
                self.average_round_time[i].append(server_close_time[i] - self.standard_time)
            self.standard_time = max(server_close_time)
        else:
            true_sum = 0
            for i in range(len(self.client_density)):
                if self.client_state[i]:
                    true_sum += 1

            # if just one client arrived to the server, server wait more time_client_begin_transmission
            if self.server.merge != 'fedasyn':
                new_threshold = max(server_receive_time[:standard_client]) + self.config.asyn_interval
                sum_time = new_threshold - self.threshold
                for i in range(len(self.client_list)):
                    if not self.client_state[i]:
                        assert self.server_to_client[i] > 0 or self.size_client_need_upload[i] > 0
                        assert self.server_to_client[i] == 0 or self.size_client_need_upload[i] == 0
                        if self.server_to_client[i] > 0:
                            self.server_to_client[i] = 0.001 if self.server_to_client[i] <= download_speed[
                                        i] * sum_time else \
                                        self.server_to_client[i] - download_speed[i] * sum_time
                            server_close_time[i] = new_threshold
                        if self.size_client_need_upload[i] > 0:
                            if server_receive_time[i] <= new_threshold:
                                assert self.size_client_need_upload[i] <= upload_speed[i] * sum_time
                                server_close_time[i] = server_receive_time[i]
                                self.size_client_need_upload[i] = 0
                                list_state_dict[i] = self.list_store_model_server[i]
                                self.client_state[i] = True
                                self.client_upload_time[i].append(
                                            server_close_time[i] - begin_time_client_upload[i])
                                # self.sum_client_upload_time[i] += self.client_upload_time[i][-1]
                            else:
                                self.size_client_need_upload[i] -= upload_speed[i] * sum_time
                                server_close_time[i] = new_threshold
                                if self.size_client_need_upload[i] < 0: self.size_client_need_upload[i] = 0.001
                self.threshold = new_threshold

            
            for i in range(len(self.client_density)):
                if self.client_state[i]:
                    self.average_round_time[i].append(server_close_time[i] - self.last_round_time[i])
                    self.last_round_time[i] = server_close_time[i]
            
            # print('test for enter')
            # print(self.client_state)
            # print(self.average_round_time)
            self.standard_time = self.threshold

            for i in range(len(self.client_density)):
                if self.client_state[i]:
                    self.waste_time[i] += (self.threshold - server_close_time[i])
                            
            
    def adjust_client_density(self, idx):
        """
        Adjust client model densities based on average round time difference.
        Called periodically when `self.interval_signal` is True.
        """
        if True:
            return
        if not self.interval_signal:
            return

        # Save optimizer states if needed (currently unused)
        list_optimizer = [client.optimizer_wrapper.optimizer.state_dict() for client in self.client_list]

        # Only adjust if minimum density is not full and not the first round

        if self.server.min_density == 1.0 or idx == 0:
            return

        # Adjustment condition based on experiment type
        exp_name = self.args.ex.lower()


        # 如果是 first_stage 并且不是 ims 开头的，也跳过
        if not self.server.first_stage:
            return

        # 其他特殊前缀可以另外加
        if not getattr(self.args, 'density_gradient', False):
            return


        

        # Use first client's average time as standard
        standard_T = sum(self.average_round_time[0]) / len(self.average_round_time[0])
        # Adjust each client's density
        for i in range(len(self.average_round_time)):
            avg_t = sum(self.average_round_time[i]) / len(self.average_round_time[i])
            rate = (standard_T - avg_t) / avg_t
            new_td = min(round(self.client_density[i] * (1 + 0.5 * rate), 4), 1.0)
            self.client_density[i] = max(new_td, self.config.min_density)

        # Round all densities to 3 decimal places
        self.client_density = [round(cd, 3) for cd in self.client_density]
        # print("new client density:"+ str(self.client_density ))

        return list_optimizer

    def prepare_server_to_client_transmission(
        self,
        model_idx,
        Increment_model_size,
        download_speed,
        upload_speed,
        begin_time_server_merge,
        begin_time_client_download,
        debug=False
    ):
        """
        Prepare model transmission from server to clients.
        - Updates self.server_to_client based on client readiness and bandwidth.
        - Calculates list_upload_size based on Residual setting.
        - Updates client download start time, start_work_time, and state.
        """

        sum_time = 0

        # === Stage check: end first stage if client density changed ===
        if self.server.first_stage and self.min_density != 1.0:
            for i in range(len(self.client_density)):
                if self.client_density[i] != self.server.client_density[i]:
                    self.server.first_stage = False
                    print('enter to the second stage')
                    break

        # === Sync updated density from server ===
        self.client_density = self.server.client_density

        # === If at least one client is active, increase global time (async case) ===
        if any(self.client_state):
            self.standard_time += self.config.asyn_interval

        # === Record server merge time if client 0 participated ===
        if self.client_state[0]:
            self.server_merge_time.append(self.standard_time - begin_time_server_merge)

        # === Mark when each active client begins downloading model ===
        for i in range(len(self.client_density)):
            if self.client_state[i]:
                begin_time_client_download[i] = self.standard_time

        # === Step: decrease unfinished download/upload volume for inactive clients ===
        false_count = sum([not st for st in self.client_state])
        if debug:
            print('First server_to_client', self.server_to_client)

        if false_count != len(self.client_density):
            for i in range(len(self.client_density)):
                if not self.client_state[i]:
                    assert self.server_to_client[i] > 0 or self.size_client_need_upload[i] > 0
                    assert self.server_to_client[i] == 0 or self.size_client_need_upload[i] == 0

                    if self.server_to_client[i] > 0:
                        self.server_to_client[i] = max(
                            0.001,
                            self.server_to_client[i] - download_speed[i] * sum_time
                        )
                    if self.size_client_need_upload[i] > 0:
                        self.size_client_need_upload[i] = max(
                            0.001,
                            self.size_client_need_upload[i] - upload_speed[i] * sum_time
                        )

        # === Determine how much each client should receive ===
        list_upload_size = [0] * len(self.client_density)
        list_client_size = self.client_model_size

        if self.Residual:
            # Only send increment updates up to the largest index
            Biggest = None
            for i in range(len(self.client_density)):
                if self.client_state[i]:
                    last_index = model_idx[i][-1]
                    if Biggest is None or last_index > Biggest:
                        Biggest = last_index

            if Biggest is not None:
                list_upload_size = Increment_model_size[:Biggest + 1] + \
                                [0] * (len(Increment_model_size) - Biggest - 1)
        else:
            # Send full model size per client
            for i in range(len(self.client_density)):
                if self.client_state[i]:
                    list_upload_size[i] = self.client_model_size[i]
            list_upload_size, sort_perm = self.process_list(list_upload_size)

        self.sum_server_upload += sum(list_upload_size)

        # if debug:
        #     print('Total upload size:', sum(list_upload_size))

        # === Update download state ===
        for i in range(len(self.client_list)):
            self.client_start_work_time[i] = self.server_to_client[i] / download_speed[i]
            if self.client_state[i]:
                assert self.server_to_client[i] == 0 and self.size_client_need_upload[i] == 0
                self.server_to_client[i] += list_client_size[i]

        self.sum_time_download += self.standard_time - self.server_up_complete
        self.server_up_complete = 0  # Will be updated in next stage

        # Return upload size list and sort_perm if needed
        return list_upload_size, list_client_size, sort_perm if not self.Residual else None

    def simulate_server_to_client_transmission(
        self,
        list_upload_size,
        list_client_size,
        sort_perm,
        download_speed,
        server_up,
        standard_client_number
    ):
        """
        Simulate server-to-client transmission time, update time_from_server_to_client,
        server_up_complete, and next round threshold.
        """

        # === Step 1: Compute time_from_server_to_client ===
        if self.Residual:
            time_from_server_to_client = [
                self.client_model_size[i] / min(server_up, download_speed[i])
                for i in range(len(self.client_model_size))
            ]
            self.server_up_complete = sum(list_upload_size) / server_up + self.standard_time
        else:
            time_from_server_to_client = [0] * len(self.client_list)
            for i in range(len(sort_perm)):
                for j in sort_perm[i]:
                    time_from_server_to_client[j] = (
                        self.server_up_complete + list_upload_size[i] / min(server_up, download_speed[j])
                    )
                self.server_up_complete += list_upload_size[i] / server_up

            self.server_up_complete += self.standard_time

        # === Step 2: For inactive clients, use remaining server_to_client time ===
        for i in range(len(self.client_list)):
            if not self.client_state[i]:
                time_from_server_to_client[i] = self.server_to_client[i] / download_speed[i]

        # === Step 3: Update global threshold (round end time) ===
        if self.chronous.lower().startswith('syn'):
            self.threshold = self.standard_time + 1000
        else:
            # Estimate max time across top-K clients or server's own completion
            topK_time = max(time_from_server_to_client[:standard_client_number])
            delay_time = self.server_up_complete - self.standard_time
            self.threshold = self.standard_time + max(topK_time, delay_time) + self.config.asyn_interval

        return time_from_server_to_client

    def finalize_download_states(self, time_from_server_to_client, begin_time_client_download, list_state_dict, download_speed):
        """
        Update client states after simulating server-to-client transmission.
        Determines whether each client successfully received the model.
        """

        for i in range(len(self.client_list)):
            if self.client_state[i]:
                # Active client fails to finish download
                if time_from_server_to_client[i] + self.standard_time > self.threshold:
                    if is_hetero_family_method(self.server.method):
                        payload_idx, payload_state_dict, _ = unpack_client_state_payload(list_state_dict[i])
                        self.list_store_model_client[i] = [
                            payload_idx,
                            payload_state_dict,
                            copy.deepcopy(self.server.list_mask[i]),
                        ]
                    else:
                        self.list_store_model_client[i] = list_state_dict[i]
                    if is_hetero_family_method(self.server.method):
                        self.list_store_mask_client[i] = copy.deepcopy(self.server.list_mask[i])
                    self.server_to_client[i] = (time_from_server_to_client[i] + self.standard_time - self.threshold) * download_speed[i]
                    self.server_to_client[i] = max(0.001, self.server_to_client[i])
                    self.client_state[i] = False
                    time_from_server_to_client[i] = self.threshold - self.standard_time

                    assert self.server_to_client[i] > 0 or self.size_client_need_upload[i] > 0
                    assert self.server_to_client[i] == 0 or self.size_client_need_upload[i] == 0
                else:
                    # Successful download
                    self.server_to_client[i] = 0
                    self.client_download_time[i].append(
                        time_from_server_to_client[i] + self.standard_time - begin_time_client_download[i]
                    )
            else:
                # Client was inactive: resume if transmission finishes
                assert self.server_to_client[i] > 0 or self.size_client_need_upload[i] > 0
                assert self.server_to_client[i] == 0 or self.size_client_need_upload[i] == 0

                if self.server_to_client[i] > 0:
                    if time_from_server_to_client[i] + self.standard_time > self.threshold:
                        self.server_to_client[i] -= download_speed[i] * (self.threshold - self.standard_time)
                        self.server_to_client[i] = max(0.001, self.server_to_client[i])
                        time_from_server_to_client[i] = self.threshold - self.standard_time
                    else:
                        self.server_to_client[i] = 0
                        time_from_server_to_client[i] = self.client_start_work_time[i]
                        assert self.server_to_client[i] == 0 and self.size_client_need_upload[i] == 0
                        self.client_state[i] = True
                        self.client_download_time[i].append(
                            time_from_server_to_client[i] + self.standard_time - begin_time_client_download[i]
                        )
                        list_state_dict[i] = self.list_store_model_client[i]
                        if is_hetero_family_method(self.server.method) and self.list_store_mask_client[i] is not None:
                            self.client_masks_in_use[i] = copy.deepcopy(self.list_store_mask_client[i])

                if self.size_client_need_upload[i] > 0:
                    assert self.server_to_client[i] == 0
                    time_from_server_to_client[i] = 0

        return list_state_dict, time_from_server_to_client  # updated
        
    def apply_model_and_schedule_to_clients(self, list_state_dict):
        """
        Apply the aggregated model (and possibly global learning rate) to clients.
        """

        # Step 1: Update global learning rate if optimizer wrapper is used
        global_lr = None
        if self.server.ip_optimizer_wrapper is not None and self.client_state[0]:
            self.server.ip_optimizer_wrapper.lr_scheduler_step()
            global_lr = self.server.ip_optimizer_wrapper.get_last_lr()

        # Step 2: Push model (and possibly LR) to each active client
        for i in range(len(self.client_state)):
            if self.client_state[i]:
                client = self.client_list[i]
                client.load_state_dict(list_state_dict[i])
                if is_hetero_family_method(self.server.method):
                    _, _, payload_mask = unpack_client_state_payload(list_state_dict[i])
                    effective_mask = payload_mask if payload_mask is not None else self.server.list_mask[i]
                    self.client_masks_in_use[i] = copy.deepcopy(effective_mask)
                    client.load_mask(copy.deepcopy(effective_mask))
                if global_lr is not None:
                    client.load_lr(global_lr)               

    def main(self):
        # model initialization completed

        log_file_path = os.path.join(self.server.save_path, 'app.log')
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s - %(message)s',
                            handlers=[
                                logging.FileHandler(log_file_path),  # 写入文件
                                # logging.StreamHandler()  # 输出到控制台
                            ])
        # 创建一个日志记录器
        logger = logging.getLogger(__name__)
        start = timer()

        self.server.start_time = start
        self.server.logger = logger
        self.server.log_first_layer_channel_bounds()

        # the time_client_begin_transmission from server to client
        self.list_store_model_server = [None for _ in range(len(self.client_density))]
        self.list_store_model_client = [None for _ in range(len(self.client_density))]
        self.list_store_mask_client = [None for _ in range(len(self.client_density))]
        idx = self.train_number[0]
        begin_time_client_upload = [0] * len(self.client_state)
        begin_time_server_merge = 0
        begin_time_client_download = [0] * len(self.client_state)
        debug = False
        # begin FL training
        standard_client_number = int(self.args.part[0] * self.args.number_clients)

        while True:
            # === Step 1: Check whether to trigger interval-based logic (e.g., density adjustment)
            if idx % self.interval == 0 and idx != self.previous_idx:
                self.interval_signal = True
            else:
                self.interval_signal = False
            self.previous_idx = idx


            # # === Step 2: Early stopping condition
            # if (self.standard_time > self.config.MAX_TIME and idx > self.config.MAX_ROUND and self.server.early_stop) or \
            #     self.server.list_acc[-1] > 1:
            if (self.standard_time > self.config.MAX_TIME ) or \
                self.server.list_acc[-1] > 1 or \
                idx > self.config.MAX_ROUND:
                logger.info(f'early stop at round {idx}')
                print(f'early stop at round {idx}')
                self.server.save_display_data()
                self.server.delete_checkpoint()
                self.server.save_final_model()
                return 3
                break

            # === Step 3: Local training on all selected clients
            list_state_dict, list_last_lr, list_sparse_model, density = self.train_client(idx, logger)
            density = self.server.client_density

            # === Step 4: Simulate client-to-server communication and upload delays
            begin_time_client_upload = [0] * len(self.client_state)
            list_state_dict, server_up, download_speed, upload_speed, server_close_time, server_receive_time, \
                begin_time_client_upload = self.simulate_client_to_server(
                    list_sparse_model, density, begin_time_client_upload, list_state_dict
                )

            # === Step 5: Update time-related statistics
            self.update_time_tracking(
                idx, list_state_dict, begin_time_client_upload,
                standard_client_number, server_close_time, server_receive_time, download_speed, upload_speed
            )

            # === Step 6: Adjust client density if interval signal is triggered
            list_optimizer = self.adjust_client_density(idx)

            # === Step 7: Placeholder for gradient (not used in current setup)
            sgrad_to_upload = None

            # === Step 8: Record the time when server starts model aggregation
            if self.client_state[0]:
                begin_time_server_merge = self.standard_time

            # === Step 9: Server aggregates models from clients and computes updates
            list_state_dict, model_idx, sum_time, self.client_loss, sub_density, Increment_model_size, \
                self.client_model_size, _ = self.server.main(
                    idx, list_state_dict, self.list_num,
                    self.standard_time, self.client_density, list_optimizer,
                    self.train_number, self.interval_signal, self.average_round_time,
                    sgrad_to_upload, self.list_time,
                    self.sum_server_upload, self.sum_server_download, self.sum_time_download
                )

            # === Step 10: Determine transmission size per client and update server_to_client states
            list_upload_size, list_client_size, sort_perm = self.prepare_server_to_client_transmission(
                model_idx, Increment_model_size,
                download_speed, upload_speed,
                begin_time_server_merge, begin_time_client_download,
                debug=debug
            )

            # === Step 11: Simulate transmission delays and compute round deadline
            time_from_server_to_client = self.simulate_server_to_client_transmission(
                list_upload_size,
                list_client_size,
                sort_perm,
                download_speed,
                server_up,
                standard_client_number
            )

            list_state_dict, time_from_server_to_client = self.finalize_download_states(
                time_from_server_to_client, begin_time_client_download, list_state_dict, download_speed
            )

            self.apply_model_and_schedule_to_clients(list_state_dict)

            # === Step 12: Record when server finishes sending to each client
            self.communicate_time_from_server_to_client = [
                self.standard_time + sc for sc in time_from_server_to_client
            ]
            idx = self.train_number[0]
