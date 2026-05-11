import argparse
import pickle
import copy
from collections import deque
from xml.dom import NoDataAllowedErr

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
                        default='g',
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
                        default=1.0,
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

    parser.add_argument('-mu',
                        help="FedProx mu",
                        action='store',
                        dest='mu',
                        type=float,
                        default=0.01,
                        required=False
                        )

    return parser.parse_args()


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

            elif self.last_score[i] == score[i]:
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

        self.ip_train_loader = None
        self.ip_test_loader = None
        self.ip_optimizer_wrapper = None
        self.ip_control = None
        self.round = None

        self.test_loader = None
        self.control = None
        self.init_test_loader()
        self.init_clients()

        self.init_control()
        self.init_ip_config()
        self.save_exp_config()
        self.start_time = timer()
        self.min_density = args.min_density
        self.list_extra_params = self.get_init_extra_params()
        self.list_mask = None
        self.model_idx = None
        self.interval = args.interval
        self.merge = args.merge
        self.sever_to_client_sum = []
        self.fed_avg_acc = []
        self.model_G = []
        self.fed_avg_loss = []
        self.accumulate = args.accumulate
        self.increse = float(args.increase)
        self.use_coeff = True
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

        self.sever_to_client_sum = []

        self.model_size = []
        self.list_optimizer = None

        self.early_stop = False
        self.client_density = copy.deepcopy(args.client_density)
        self.list_client_density = [[] for _ in range(len(self.client_density))]

        self.list_client_loss = [[] for _ in range(len(self.client_density))]
        self.list_client_acc = [[] for _ in range(len(self.client_density))]

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
        os.remove(checkpoint_path)

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
    def get_init_extra_params(self) -> List[tuple]:
        pass

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
        if self.config.EXP_NAME == "ImageNet100":
            list_model_size = []

            for i in range(len(list_state_dict)):


                if list_state_dict[i] == None or density[i] == 0:
                    list_model_size.append(0)
                    continue

                file_size = self.config.model_size * density[i]
                list_model_size.append(file_size)
        else:
            list_model_size = []

            for i in range(len(list_state_dict)):

                ds = self.config.EXP_NAME + str(density[i])
                if ds in self.density_size.keys():
                    list_model_size.append(self.density_size[ds])
                else:
                    client_model = list_state_dict[i]

                    if client_model == None or density[i] == 0:
                        with open((exp + '.pkl'), 'wb') as f:
                            pickle.dump(client_model, f)
                        file_size = os.path.getsize(exp + '.pkl')
                        file_size = file_size / (1024 * 1024)
                        os.remove(exp + '.pkl')
                        self.density_size['zero'] = file_size

                        list_model_size.append(0)
                        self.density_size[ds] = 0
                        continue

                    with open((exp + '.pkl'), 'wb') as f:
                        pickle.dump(client_model, f)
                    file_size = os.path.getsize(exp + '.pkl')
                    file_size = file_size / (1024 * 1024)
                    if self.config.EXP_NAME == "MNIST":
                        file_size = (file_size) * 100
                    file_size = round(file_size, 4)
                    list_model_size.append(file_size)
                    self.density_size[ds] = file_size
                    os.remove(exp + '.pkl')

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
                idx, _ = idx_state_dict
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
    def Buff_mask_fed_avg(self, list_num_proc, list_sd,
                          idx, sgrd_to_upload):  # to complete the merge model ps: fedavg
        '''
        sub_fed_avg_model = sub_fed_avg(list_num_proc, list_state_dict)
        mask = sub_fed_avg_model==0
        model[mask] =  sub_fed_avg_model[mask]
        :param list_num_proc:
        :param list_state_dict:
        :param idx:
        :return:
        '''

        self_sd = self.model.state_dict()
        dict_keys = self_sd.keys()
        list_state_dict = self.list_client_sd_buffer
        sd = copy.deepcopy(self_sd)

        for key in dict_keys:
            sum_weight = torch.zeros(size=self_sd[key].size(), device=self.device)
            sum_mask = torch.zeros(size=sum_weight.size(), device=self.device)
            if key.endswith("num_batches_tracked"):
                continue
            for coeff, npc, idx_state_dict in zip(self.list_coeff, list_num_proc, list_state_dict):
                idx, state_dict = idx_state_dict
                if state_dict[key].is_sparse: state_dict[key] = state_dict[key].to_dense()
                sum_weight = sum_weight + coeff * npc * state_dict[key].to(self.device)
                mask = (state_dict[key] != 0).to(self.device)
                sum_mask = sum_mask + coeff * npc * mask
            divisor = torch.where(sum_mask == 0, torch.tensor([1e-10], device=self.device), sum_mask)
            sum_weight = torch.div(sum_weight, divisor)
            mask2 = sum_weight != 0
            sd[key][mask2] = sum_weight[mask2]

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
    def fedasyn(self, list_num_proc, list_state_dict,
                idx, sgrd_to_upload):  # Do not merge the terrible model
        # if self.display:
        #     self.logger.info('use sub_fedavg_and_fair to merge client')
        # merged_state_dict = dict()
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

                idx, state_dict = idx_state_dict
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
            # for num_proc, state_dict in zip(list_num_proc, list_state_dict):
            #     sum_weight = sum_weight + num_proc / total_num_proc * state_dict[key].to_dense()
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

    def fed_avg_mask_model(self, list_num_proc, list_state_dict, idx):
        '''
        we want use fed_avg to accumulate model then use mask ,
        fed_avg_model = fed_avg(list_num_proc, list_sd)
        mask = fed_avg_model==0
        model[mask] =  fed_avg_model[mask]
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

        list_coeff = [list_num_proc[i] * self.list_coeff[i] for i in range(len(list_state_dict))]

        self_sd = self.model.state_dict()
        dict_keys = self_sd.keys()
        sd = copy.deepcopy(self_sd)
        for key in dict_keys:
            sum_weight = torch.zeros(size=self_sd[key].size(), device=self.device)
            sum_client = 0
            if key.endswith("num_batches_tracked"):
                continue
            i = 0
            for coeff, idx_state_dict in zip(list_coeff, list_state_dict):

                idx, state_dict = idx_state_dict
                if state_dict[key].is_sparse: state_dict[key] = state_dict[key].to_dense()
                sum_weight = sum_weight + coeff * state_dict[key].to(self.device)
                sum_client = sum_client + coeff
                i = i + 1

            sum_weight = torch.div(sum_weight, sum_client)
            mk = sum_weight == 0
            sum_weight = sum_weight.view(sd[key].size())
            sum_weight[mk] = sd[key][mk]

            if sum_weight.sum() == 0:
                continue

            sd[key] = sum_weight

        fed_avg_model.load_state_dict(sd)

        return fed_avg_model

    @torch.no_grad()
    def fed_avg_model(self, list_num_proc, list_state_dict, idx):
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
                    idx, state_dict = idx_state_dict
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

                idx, state_dict = idx_state_dict
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
                    idx, state_dict = idx_state_dict
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
                    idx, state_dict = idx_state_dict
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
    def split_model(self, start):
        list_state_dict = []
        sub_model_time = []

        for mask in self.list_mask:
            clean_model = copy.deepcopy(self.model)
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
            num_bias = 0 if layer.bias is None else layer.bias.nelement()
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

        if self.merge == 'buff_mask_fed_avg':
            self.Buff_mask_fed_avg(list_num_proc, list_sd, idx, sgrad_to_upload)
        elif self.merge == 'buff_fed_avg':
            self.buff_fed_avg(list_num_proc, list_sd, idx, sgrad_to_upload)
        elif self.merge == 'fed_avg':
            self.fed_avg(list_num_proc, list_sd, idx, sgrad_to_upload)
        elif self.merge == 'mask_fed_avg' or self.merge == 'heterofl':
            self.fedasyn(list_num_proc, list_sd, idx, sgrad_to_upload)
        elif self.merge == 'fed_asyn':
            self.fedasyn(list_num_proc, list_sd, idx, sgrad_to_upload)


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


    def load_the_run_data(self,list_time, average_round_time, interval_signal, train_number, idx, client_density  ):
        self.client_train_time, self.sum_client_train_time, self.server_merge_time, self.sum_server_merge_time, \
            self.client_upload_time, self.sum_client_upload_time, self.client_download_time, self.sum_client_download_time, self.wast_time, = list_time

        self.average_round_time = average_round_time
        self.interval_signal = interval_signal

        self.train_number = train_number

        self.round = idx
        # if the first stage, client_density may be changed by FedMP_FL

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
            self.save_checkpoint()
            self.save_display_data()
        # if state[0] == False, which means the global model improved during the training
        elif state[0] == False and self.begin_save:
            # if the global model cannot improve during the training, restore all clients model
            try:
                self.logger.info('it is not the best model')
                checkpoint = load(os.path.join(self.save_path, 'checkpoint.pth'))
                self.model = checkpoint['self.model']
                self.control.accumulate_weight_dict = checkpoint['self.control.accumulate_weight_dict']
                self.list_client_sd_buffer = checkpoint['self.list_client_sd']
                self.list_mask = checkpoint['self.list_mask']
                self.list_stalness = checkpoint['self.list_stalness']
                self.list_coeff = checkpoint['self.list_coeff']
                for key, value in self.control.accumulate_weight_dict.items():
                    self.control.accumulate_weight_dict[key] = self.control.accumulate_weight_dict[key].to(
                        self.device)
                self.model = self.model.to(self.device)
                self.control.model = self.model
                # 删除 checkpoint 变量
                del checkpoint
                # 强制释放显存
                torch.cuda.empty_cache()
            except FileNotFoundError:
                print('The checkpoint file was not found')
            self.save_display_data()
            if self.recover:
                if self.min_density == 1.0:
                    self.early_stop = True
                else:
                    if self.min_density + self.increse < 1.0:
                        self.min_density = self.min_density + self.increse
                    else:
                        self.early_stoping.patience[0] = self.early_stoping.patience[0] * 5
                        self.min_density = 1.0
                    self.early_stoping.counter[0] = 0
                    self.early_stoping.early_stop[0] = False
            else:
                self.early_stop = True



        for i in range(len(self.client_density)):
            if self.client_density[i] < self.min_density:
                self.client_density[i] = self.min_density
                if len(self.early_stoping.counter) > 1:
                    self.early_stoping.counter[i] = 0
                    self.early_stoping.early_stop[i] = False


        # Change the client density based on client accuracy
        if self.recover:
            # because the index 0 is the server, so begin from the 1
            for i in range(1, self.early_stoping.num):

                if state[i] == False:
                    self.logger.info(f'client{i - 1} is out of patience')
                    if self.client_density[i - 1] + min(self.client_density[i - 1], 0.20) < 1.0:
                        self.client_density[i - 1] = self.client_density[i - 1] + min(self.client_density[i - 1],
                                                                                      0.20)
                        self.early_stoping.counter[i] = 0
                        self.early_stoping.early_stop[i] = False
                    else:
                        self.client_density[i - 1] = 1.0
                        self.early_stoping.counter[i] = 0
                        self.early_stoping.early_stop[i] = False
                    self.logger.info(
                        f'for client{i - 1}, increase model client, client density {self.client_density[i - 1]}')

            # update the min_density
            min_density = min(self.client_density)
            if self.min_density + 0.01 < min_density:
                self.min_density = min_density
                if min_density == 1.0:
                    self.early_stoping.patience[0] = self.early_stoping.patience[0] * 2
                self.early_stoping.counter[0] = 0
                self.early_stoping.early_stop[0] = False

    def get_server_to_client_metadata(self):
        sorted_clientdensity, sorted_clientdensity_indics = torch.sort(
            torch.tensor(self.client_density), descending=False)
        sorted_clientdensity = sorted_clientdensity.tolist()
        for i in range(len(sorted_clientdensity)):
            if i == 0:
                self.sub_density[i] = sorted_clientdensity[i]
            else:
                self.sub_density[i] = sorted_clientdensity[i] - sorted_clientdensity[i - 1]


        self.client_model_size = self.get_real_size(self.process_state_dict_to_client(self.list_state_dict),
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


    def main(self, idx, list_sd, list_num_proc, current_time,client_density, list_optimizer, train_number,
             interval_signal, average_round_time, sgrad_to_upload, list_time, sum_server_upload, sum_server_download,
             time_download):

        self.load_the_run_data(list_time, average_round_time, interval_signal, train_number, idx, client_density)

        # if no client arrived, return to client train directly
        sum_time = 0
        if self.test_None_list_client_dict(list_sd) and not interval_signal:
            return self.list_state_dict, self.model_idx, sum_time, self.list_client_loss, self.sub_density, self.increment_model_size, self.client_model_size
        #calculate the staleness of different clients
        self.calculate_staleness(list_sd)
        server_start = timer()



        #  to complete the fed avg
        self.Aggregate_model(list_num_proc, list_sd, idx, sgrad_to_upload)
        self.test_client_model(list_sd)





        if interval_signal:
            # record the metadata
            self.sum_server_upload.append(sum_server_upload)
            for i in range(len(self.client_density)):
                self.sum_server_download[i].append(sum_server_download[i])
            self.time.append(current_time)

            # test the global model performance
            if self.min_density > 0.99:
                fed_avg_model = self.model
            else:
                fed_avg_model = self.fed_avg_model(list_num_proc, list_sd, idx)
            self.Evaluate_Global_Model(fed_avg_model)
            self.test_pruning_performance(self.fed_avg_loss[-1], fed_avg_model)


            self.Print_FL_Message(idx, time_download)





        # GMR
        start = timer()
        # Only update the client mask at fixed intervals
        # other rounds use the old client mask directly
        if interval_signal:
            old_client_desity = copy.deepcopy(self.client_density)

            # Try to alter the client density
            self.GMR(list_optimizer)

            # if client_density has been changed, use adjust function to adjust the client model
            if self.recover and old_client_desity != self.client_density:
                self.logger.info('Adjust model'+': the new density list is ' + str(self.client_density))
                print('Adjust model' + ': the new density list is ' + str(self.client_density))

                # get the client model based on client_density
                self.list_state_dict, self.model_idx, sub_model_time, self.list_mask, _, increment_state_dict, client_density = self.control.sub_adjust_fast(
                        client_density=self.client_density, use_coff=self.use_coeff,
                        min_density=self.config.min_density, )

                self.increment_model_size = self.get_real_size(
                    self.process_state_dict_to_client(increment_state_dict), self.experiment_name, self.sub_density)

                # based on self.list_mask to get the corresponding model
                self.list_state_dict, sub_model_time = self.split_model(start)
        else:
            self.list_state_dict, sub_model_time = self.split_model(start)

        if interval_signal:
            self.sub_density, self.client_model_size = self.get_server_to_client_metadata()


        # add extra idx to list_state_dict for Asynchronous Federated Learning
        for i in range(len(self.list_state_dict)):
            self.list_state_dict[i] = [idx, self.list_state_dict[i]]

        sum_time += timer() - server_start

        return self.list_state_dict, self.model_idx, sum_time, self.list_client_loss, self.sub_density, self.increment_model_size, self.client_model_size


class FedMapClient:
    def __init__(self, model, config, use_adaptive, extra_params, exp_config, args, device):
        self.args = args
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
        self.parse_init_extra_params(extra_params)
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

    @abstractmethod
    def parse_init_extra_params(self, extra_params):
        # Initialize train_loader, etc.
        pass

    def convert_to_sparse(self):
        self.model = self.model.to_sparse()
        old_lr = self.optimizer.state_dict()["param_groups"][0]["lr"]
        self.optimizer = self.exp_config.optimizer_class(params=self.model.parameters(),
                                                         **self.exp_config.optimizer_params)
        if self.exp_config.lr_scheduler_class is not None:
            lr_scheduler_state_dict = deepcopy(self.lr_scheduler.state_dict())
            self.lr_scheduler = self.exp_config.lr_scheduler_class(optimizer=self.optimizer,
                                                                   **self.exp_config.lr_scheduler_params)
            self.lr_scheduler.load_state_dict(lr_scheduler_state_dict)
        self.optimizer.param_groups[0]["lr"] = old_lr
        self.optimizer_wrapper = OptimizerWrapper(self.model, self.optimizer, self.lr_scheduler)

        self.is_sparse = True

    @abstractmethod
    def parse_init_extra_params(self, extra_params):
        # Initialize train_loader, etc.
        pass

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
    def process_state_dict_to_server(self, sparse_model) -> dict:
        """
        Process state dict before sending to server, e.g. keep values only, extra param in adjustment round.
        if not self.is_sparse: send dense
        elif self.adjustment_round: send sparse values + extra grad values
        else: send sparse values only
        To be overridden by subclasses.
        """
        clean_state_dict = self.cleanup_state_dict_to_server(sparse_model)

        for key, param in clean_state_dict.items():
            if param.is_sparse:
                clean_state_dict[key] = param._values()
        # but in our model, different client have different density, it will have different structure

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
            num_bias = 0 if layer.bias is None else layer.bias.nelement()
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

    @torch.no_grad()
    def load_state_dict(self, idx_state_dict):
        idx, state_dict = idx_state_dict
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

            num_proc_data += len(inputs)



        return num_proc_data, loss, list_grad

    def main(self, idx, logger):
        self.model.train()
        model_density = round(self.calc_model_params(),4)
        num_proc_data, loss, list_grad = self.train_model()
        accumulated_grad = dict()
        lr = self.optimizer_wrapper.optimizer.param_groups[0]['lr']
        state_dict_to_server = self.process_state_dict_to_server(copy.deepcopy(self.model).to_sparse())
        return self.model.state_dict(), num_proc_data, lr, model_density, state_dict_to_server, self.client_idx + 1, accumulated_grad


class FedMapFL(ABC):
    def __init__(self, args, config, server, client_list):
        self.config = config

        # self.use_adaptive = True

        self.max_round = config.MAX_ROUND
        self.server = server
        self.client_list = client_list
        self.server.client_list = client_list

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
        # self.computerlr = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        self.client_density = args.client_density
        self.client_state = [True] * len(client_list)
        self.size_client_need_upload = [0] * len(client_list)
        self.server_to_client = [0] * len(client_list)
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

        import numpy as np

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
        self.server_up_complete = 0
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
        import numpy as np

        server_up = 0
        server_down = 0
        n = 20
        for i in range(n):
            server_up += np.random.lognormal(mean=0, sigma=self.variance_server_up) * self.average_server_up
            server_down += np.random.lognormal(mean=0, sigma=self.variance_server_down) * self.average_server_down
        server_up = server_up / n
        server_down = server_down / n

        for i in range(len(self.average_upload_speed)):
            dp = 0
            up = 0
            for j in range(n):
                dp += np.random.lognormal(mean=0, sigma=self.variance_upload) * self.average_download_speed[i]
                up += np.random.lognormal(mean=0, sigma=self.variance_download) * self.average_upload_speed[i]

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
        if self.config.EXP_NAME == "ImageNet100":
            list_model_size = []
            for i in range(len(list_state_dict)):
                if list_state_dict[i] == None or density[i] == 0:
                    list_model_size.append(0)
                    continue

                file_size = self.config.model_size * density[i]
                list_model_size.append(file_size)
        else:
            list_model_size = []
            for i in range(len(list_state_dict)):
                ds = self.config.EXP_NAME + str(density[i])
                if ds in self.density_size.keys():
                    list_model_size.append(self.density_size[ds])
                else:
                    client_model = list_state_dict[i]
                    if client_model == None or density[i] == 0:
                        with open((exp + '.pkl'), 'wb') as f:
                            pickle.dump(client_model, f)
                        file_size = os.path.getsize(exp + '.pkl')
                        file_size = file_size / (1024 * 1024)
                        os.remove(exp + '.pkl')
                        self.density_size['zero'] = file_size

                        list_model_size.append(0)
                        self.density_size[ds] = 0
                        continue
                    with open((exp + '.pkl'), 'wb') as f:
                        pickle.dump(client_model, f)
                    file_size = os.path.getsize(exp + '.pkl')
                    file_size = file_size / (1024 * 1024)
                    if self.config.EXP_NAME == "MNIST":
                        file_size = (file_size) * 100
                    file_size = round(file_size, 4)
                    list_model_size.append(file_size)
                    self.density_size[ds] = file_size
                    os.remove(exp + '.pkl')

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

        # train the free clients
        for i in range(len(self.client_list)):
            if self.client_state[i]:

                client = self.client_list[i]
                sd, npc, last_lr, ds, sparse_model, client_idx, sgrd = client.main(idx, logger)
                self.client_train_time[i].append(self.config.asyn_interval)
                self.sum_client_train_time[i].append(
                    self.sum_client_train_time[i][-1] + self.client_train_time[i][-1])
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

    def simulate_client_to_server(self,list_sparse_model, density,begin_time_client_upload, list_state_dict):
        '''
        To simulate the client to server process, We set the server to download the clients in the order they uploaded them.
        However, the server's download bandwidth is actually greater than the total upload bandwidth of all clients in our experiment environment.
        The rule of server download sequence has no effect in most cases.
        But for Asyn-FL, due to only aggregate one client at once, we set extra rule to define the sequence of client aggregation order in line 1908.
        '''

        model_size = self.get_real_size(list_sparse_model , self.server.experiment_name, density)

        # just for save the display data
        for i in range(len(self.client_list)):
            self.sum_server_download[i] += model_size[i]
        self.server.list_model_size.append(model_size)


        server_up, server_down, download_speed, upload_speed = self.get_internet_speed()

        # the volume of client need to upload
        self.size_client_need_upload = [cs + ms for cs, ms in zip(self.size_client_need_upload, model_size)]

        from control.sub_algorithm import simulate_client_to_server

        # calculate when the client begin to transmission, add extra client train time
        time_client_begin_transmission = []
        for i in range(len(self.client_state)):
            assert self.server_to_client[i] > 0 or self.size_client_need_upload[i] > 0
            assert self.server_to_client[i] == 0 or self.size_client_need_upload[i] == 0
            if self.client_state[i]:
                begin_time_client_upload[i] = self.communicate_time_from_server_to_client[i] + self.client_train_time[i][-1]
                time_client_begin_transmission.append(begin_time_client_upload[i])
            else:
                # but it has some deviation, because it is the simulate time for client with statu false.
                time_client_begin_transmission.append(self.communicate_time_from_server_to_client[i])

        # computer when the client model arrive the server
        # simulate_client_to_server: The server downloads the client models according to the order in which the clients send the models to the server, but the server's download capacity is much greater than the client's download capacity, so multiple client models can be downloaded at the same time.
        server_receive_time = np.array(
            simulate_client_to_server(time_client_begin_transmission, copy.copy(self.size_client_need_upload),
                                      upload_speed, server_down))
        server_close_time = copy.copy(server_receive_time)

        # The server aggregate to the global model when the client model arrived.

        if self.server.merge == 'fedasyn':

            max_true_client = 1
            # because only the first client model is useful.
            # This order ensures that the recently trained client has the last priority in the next round of training.
            if self.start_client_idx == len(server_close_time):
                server_download_sequence = list(range(0, len(server_close_time)))
            else:
                server_download_sequence = list(range(self.start_client_idx, len(server_close_time))) + list(
                    range(0, self.start_client_idx))
        else:
            # because no limitation of only aggregate one client, we don't need to set extra rule.
            max_true_client = len(server_close_time)
            copy_time_client_begin_transmission = copy.copy(time_client_begin_transmission)
            for i in range(len(time_client_begin_transmission)):
                if not self.client_state[i]:
                    copy_time_client_begin_transmission[i] = 0

            # download the client model by the client upload time
            server_download_sequence = [index for index, value in
                                        sorted(enumerate(copy_time_client_begin_transmission), key=lambda x: x[1])]


        # To simulate the server download process
        for i in server_download_sequence:
            if self.client_state[i]:
                if server_close_time[i] > self.threshold:
                    self.client_state[i] = False
                    server_close_time[i] = self.threshold
                    self.size_client_need_upload[i] = self.size_client_need_upload[i] - upload_speed[i] * (
                            self.threshold - time_client_begin_transmission[i])
                    if self.size_client_need_upload[i] <= 0:
                        self.size_client_need_upload[i] = 0.001
                    assert self.server_to_client[i] > 0 or self.size_client_need_upload[i] > 0
                    assert self.server_to_client[i] == 0 or self.size_client_need_upload[i] == 0

                else:
                    assert server_close_time[i] <= self.threshold
                    if max_true_client > 0:
                        self.size_client_need_upload[i] = 0
                        self.client_upload_time[i].append(server_close_time[i] - begin_time_client_upload[i])
                        self.sum_client_upload_time[i] += self.client_upload_time[i][-1]
                        max_true_client = max_true_client - 1

                    else:
                        self.size_client_need_upload[i] = self.size_client_need_upload[i] - upload_speed[i] * (
                                self.threshold - time_client_begin_transmission[i])
                        if self.size_client_need_upload[i] <= 0:
                            self.size_client_need_upload[i] = 0.001
                        self.client_state[i] = False
            else:
                assert self.server_to_client[i] > 0 or self.size_client_need_upload[i] > 0
                assert self.server_to_client[i] == 0 or self.size_client_need_upload[i] == 0

                if self.size_client_need_upload[i] > 0:
                    # which means the client is uploading the last round of client.
                    if server_close_time[i] > self.threshold:

                        self.size_client_need_upload[i] = self.size_client_need_upload[i] - upload_speed[i] * (
                                self.threshold - time_client_begin_transmission[i])
                        if self.size_client_need_upload[i] < 0: self.size_client_need_upload[i] = 0.001
                        server_close_time[i] = self.threshold
                    else:
                        assert self.server_to_client[i] == 0
                        assert list_state_dict[i] == None
                        if max_true_client > 0:
                            self.size_client_need_upload[i] = 0
                            list_state_dict[i] = self.list_store_model_server[i]
                            self.client_state[i] = True
                            self.client_upload_time[i].append(server_close_time[i] - begin_time_client_upload[i])
                            self.sum_client_upload_time[i] += self.client_upload_time[i][-1]
                            max_true_client = max_true_client - 1
                            self.start_client_idx = i + 1
                        else:
                            self.size_client_need_upload[i] = 0.001
                            self.client_state[i] = False
                            server_close_time[i] = self.threshold
                if self.server_to_client[i] > 0:
                    server_receive_time[i] = self.threshold + 1000

        return list_state_dict



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

        # the time_client_begin_transmission from server to client
        self.list_store_model_server = [None for _ in range(len(self.client_density))]
        self.list_store_model_client = [None for _ in range(len(self.client_density))]
        idx = self.train_number[0]
        begin_time_client_upload = [0] * len(self.client_state)
        begin_time_server_merge = 0
        begin_time_client_download = [0] * len(self.client_state)
        debug = False
        # begin FL training
        standard_client_number = int(self.args.part[0] * self.args.number_clients)


        while True:
            # because the idx = train_number[0], the signal of adj round
            if idx % self.interval == 0 and idx != self.previous_idx:
                self.interval_signal = True
            else:
                self.interval_signal = False

            self.previous_idx = idx

            # When to stop the FL
            if (self.standard_time > self.config.MAX_TIME and idx > self.config.MAX_ROUND and self.server.early_stop) or self.server.list_acc[-1] > 1 :
                logger.info('early stop at' + str(idx) + ' round of FL')
                self.server.save_display_data()
                self.server.delete_checkpoint()
                self.server.save_final_model()
                break


            list_state_dict, list_last_lr, list_sparse_model, density = self.train_client(idx, logger)



            model_size = self.get_real_size(list_sparse_model, self.server.experiment_name, density)

            # just for record the display data
            for i in range(len(self.client_list)):
                self.sum_server_download[i] += model_size[i]
            self.server.list_model_size.append(model_size)


            server_up, server_down, download_speed, upload_speed = self.get_internet_speed()

            # control the client to server
            self.size_client_need_upload = [cs + ms for cs, ms in zip(self.size_client_need_upload, model_size)]

            from control.sub_algorithm import simulate_client_to_server, determine_density, simluate_server_to_client
            time_client_begin_transmission = []
            for i in range(len(self.client_state)):
                assert self.server_to_client[i] > 0 or self.size_client_need_upload[i] > 0
                assert self.server_to_client[i] == 0 or self.size_client_need_upload[i] == 0
                if self.client_state[i]:
                    time_client_begin_transmission.append(
                        self.communicate_time_from_server_to_client[i] + self.client_train_time[i][-1])
                    begin_time_client_upload[i] = time_client_begin_transmission[i]
                else:
                    # but it has some deviation, because it is the simulate time for client with statu false.
                    time_client_begin_transmission.append(self.communicate_time_from_server_to_client[i])

            # computer when the client model arrive the server
            # simulate_client_to_server: The server downloads the client models according to the order in which the clients send the models to the server, but the server's download capacity is much greater than the client's download capacity, so multiple client models can be downloaded at the same time.
            server_receive_time = np.array(
                simulate_client_to_server(time_client_begin_transmission, copy.copy(self.size_client_need_upload),
                                          upload_speed, server_down))
            server_close_time = copy.copy(server_receive_time)

            # for AsynFL, Aggregate the client model when the client model arrived
            # so max_true_client = 1
            # Calculate the download_sequence
            if self.server.merge == 'fedasyn':
                max_true_client = 1

                if start_client_idx == len(server_close_time):
                    server_download_sequence = list(range(0, len(server_close_time)))
                else:
                    server_download_sequence = list(range(start_client_idx, len(server_close_time))) + list(
                        range(0, start_client_idx))
            else:
                max_true_client = len(server_close_time)
                real_client_transmission = copy.copy(time_client_begin_transmission)
                for i in range(len(time_client_begin_transmission)):
                    if not self.client_state[i]:
                        real_client_transmission[i] = 0

                server_download_sequence = [index for index, value in
                                           sorted(enumerate(real_client_transmission), key=lambda x: x[1])]

            for i in server_download_sequence:
                if self.client_state[i]:
                    if server_close_time[i] > self.threshold:
                        self.client_state[i] = False
                        server_close_time[i] = self.threshold
                        self.size_client_need_upload[i] = self.size_client_need_upload[i] - upload_speed[i] * (
                                    self.threshold - time_client_begin_transmission[i])
                        if self.size_client_need_upload[i] <= 0:
                            self.size_client_need_upload[i] = 0.001
                        assert self.server_to_client[i] > 0 or self.size_client_need_upload[i] > 0
                        assert self.server_to_client[i] == 0 or self.size_client_need_upload[i] == 0
                    else:
                        assert server_close_time[i] <= self.threshold
                        if max_true_client > 0:
                            self.size_client_need_upload[i] = 0
                            self.client_upload_time[i].append(server_close_time[i] - begin_time_client_upload[i])
                            self.sum_client_upload_time[i] += self.client_upload_time[i][-1]
                            max_true_client = max_true_client - 1
                            start_client_idx = i + 1
                        else:
                            self.size_client_need_upload[i] = self.size_client_need_upload[i] - upload_speed[i] * (
                                    self.threshold - time_client_begin_transmission[i])
                            if self.size_client_need_upload[i] <= 0:
                                self.size_client_need_upload[i] = 0.001
                            self.client_state[i] = False
                else:
                    assert self.server_to_client[i] > 0 or self.size_client_need_upload[i] > 0
                    assert self.server_to_client[i] == 0 or self.size_client_need_upload[i] == 0
                    if self.size_client_need_upload[i] > 0:
                        # which means the client is uploading the last round of client.
                        if server_close_time[i] > self.threshold:

                            self.size_client_need_upload[i] = self.size_client_need_upload[i] - upload_speed[i] * (
                                        self.threshold - time_client_begin_transmission[i])
                            if self.size_client_need_upload[i] < 0: self.size_client_need_upload[i] = 0.001
                            server_close_time[i] = self.threshold
                        else:
                            assert self.server_to_client[i] == 0
                            assert list_state_dict[i] == None
                            if max_true_client > 0:
                                self.size_client_need_upload[i] = 0
                                list_state_dict[i] = self.list_store_model_server[i]
                                self.client_state[i] = True
                                self.client_upload_time[i].append(server_close_time[i] - begin_time_client_upload[i])
                                self.sum_client_upload_time[i] += self.client_upload_time[i][-1]
                                max_true_client = max_true_client - 1
                                start_client_idx = i + 1
                            else:
                                self.size_client_need_upload[i] = 0.001
                                self.client_state[i] = False
                                server_close_time[i] = self.threshold
                    if self.server_to_client[i] > 0:
                        server_receive_time[i] = self.threshold + 1000

            if idx < 2:
                self.standard_time = max(server_close_time)
            else:
                if self.chronous.lower().startswith('syn'):
                    # server wait all clients arrived
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

                        new_threshold = max(server_receive_time[:standard_client_number]) + self.config.asyn_interval

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
                                        self.sum_client_upload_time[i] += self.client_upload_time[i][-1]
                                    else:
                                        self.size_client_need_upload[i] -= upload_speed[i] * sum_time
                                        server_close_time[i] = new_threshold
                                        if self.size_client_need_upload[i] < 0: self.size_client_need_upload[i] = 0.001
                        self.threshold = new_threshold

                    for i in range(len(self.client_density)):
                        if self.client_state[i]:
                            self.average_round_time[i].append(server_close_time[i] - self.last_round_time[i])
                            self.last_round_time[i] = server_close_time[i]

                    self.standard_time = self.threshold

                    for i in range(len(self.client_density)):
                        if self.client_state[i]:
                            self.waste_time[i] += (self.threshold - server_close_time[i])

            list_optimizer = []


            if self.interval_signal:
                for client in self.client_list:
                    list_optimizer.append(client.optimizer_wrapper.optimizer.state_dict())


                if self.server.min_density != 1.0 and idx != 0:
                    # if self.server.first_stage or self.args.ex.lower().startswith(
                    #         'heterofl') or self.args.ex.lower().startswith('gmr'):

                    if (self.server.first_stage and not self.args.ex.lower().startswith('ims')) or self.args.ex.lower().startswith('heterofl') or self.args.ex.lower().startswith('gmr'):
                        print('adjust client density')
                        standard_T = sum(self.average_round_time[0]) / len(self.average_round_time[0])

                        print([(sum(self.average_round_time[i]) / len(self.average_round_time[i])) for i in range(len(self.average_round_time))])

                        for i in range(len(self.average_round_time)):

                            avg_t = sum(self.average_round_time[i]) / len(self.average_round_time[i])
                            rate = (standard_T - avg_t) / avg_t
                            new_td = min(round(self.client_density[i] * (1 + 0.5 * rate), 4), 1.00)

                            if new_td >= self.config.min_density or new_td >= self.config.min_density:
                                self.client_density[i] = new_td
                            else:
                                self.client_density[i] = max(self.config.min_density, self.config.min_density)
                            self.client_density = [round(cd, 3) for cd in self.client_density]

            if self.interval_signal:
                sgrad_to_upload = None
            else:
                sgrad_to_upload = None

            if self.client_state[0]:       begin_time_server_merge = self.standard_time
            list_state_dict, model_idx, sum_time, self.client_loss, sub_density, Increment_model_size, self.client_model_size \
                = self.server.main(idx, list_state_dict, self.list_num,
                                   self.standard_time,
                                   self.client_density, list_optimizer,
                                   self.train_number, self.interval_signal, self.average_round_time,
                                   sgrad_to_upload, self.list_time, self.sum_server_upload, self.sum_server_download,
                                   self.sum_time_download)

            sum_time = 0
            if self.server.first_stage and self.min_density != 1.0:
                for i in range(len(self.client_density)):
                    if not self.client_density[i] == self.server.client_density[i]:
                        self.server.first_stage = False

                        break

            self.client_density = self.server.client_density

            for i in self.client_state:
                if i:
                    self.standard_time = self.config.asyn_interval + self.standard_time
                    break

            if self.client_state[0]:
                self.server_merge_time.append(self.standard_time - begin_time_server_merge)
                self.sum_server_merge_time.append(self.sum_server_merge_time[-1] + self.server_merge_time[-1])

            for i in range(len(self.client_density)):
                if self.client_state[i]:
                    begin_time_client_download[i] = self.standard_time

            False_number = 0

            for st in self.client_state:
                if not st:
                    False_number += 1

            if debug:
                print('First server_to_client' + str(self.server_to_client))
                # print('third self.client_to_server' + str(self.size_client_need_upload))

            if not False_number == len(self.client_density):
                for i in range(len(self.client_density)):
                    if not self.client_state[i]:
                        assert self.server_to_client[i] > 0 or self.size_client_need_upload[i] > 0
                        assert self.server_to_client[i] == 0 or self.size_client_need_upload[i] == 0
                        if self.server_to_client[i] > 0:
                            self.server_to_client[i] = 0.001 if self.server_to_client[i] <= download_speed[
                                i] * sum_time else \
                                self.server_to_client[i] - download_speed[i] * sum_time

                        if self.size_client_need_upload[i] > 0:
                            self.size_client_need_upload[i] = 0.001 if self.size_client_need_upload[i] <= upload_speed[
                                i] * sum_time else \
                                self.size_client_need_upload[i] - upload_speed[i] * sum_time
                # send model to the client
                # if debug:
                #     print('Third self.client_to_server' + str(self.size_client_need_upload))

            list_client_size = self.client_model_size
            list_upload_size = [0] * len(self.client_density)
            if self.Residual:
                Biggest = None
                for i in range(len(self.client_density)):
                    if self.client_state[i]:
                        index = model_idx[i][-1]
                        if Biggest == None or index > Biggest: Biggest = index

                if Biggest != None:
                    list_upload_size = Increment_model_size[:Biggest + 1] + [0] * (
                                len(Increment_model_size) - Biggest - 1)

            else:
                wait_load_model = []
                for i in range(len(self.client_density)):
                    if self.client_state[i]:
                        list_upload_size[i] = self.client_model_size[i]
                list_upload_size, sort_perm = self.process_list(list_upload_size)
            self.sum_server_upload += sum(list_upload_size)


            if debug:
                print(sum(list_upload_size))

            for i in range(len(self.client_list)):

                self.client_start_work_time[i] = self.server_to_client[i] / download_speed[i]
                if self.client_state[i]:
                    assert self.server_to_client[i] == 0 and self.size_client_need_upload[i] == 0
                    self.server_to_client[i] += list_client_size[i]
            self.sum_time_download += self.standard_time - self.server_up_complete
            self.server_up_complete = 0

            if self.Residual:
                time_from_server_to_client = [self.client_model_size[i] / min(server_up, download_speed[i]) for i
                                              in range(len(self.client_model_size))]
                self.server_up_complete = sum(list_upload_size) / server_up + self.standard_time
            else:


                time_from_server_to_client = [0] * len(self.client_list)
                for i in range(len(sort_perm)):
                    for j in sort_perm[i]:
                        time_from_server_to_client[j] = self.server_up_complete + list_upload_size[i] / min(server_up,
                                                                                                            download_speed[
                                                                                                                j])
                    self.server_up_complete = self.server_up_complete + list_upload_size[i] / server_up

                self.server_up_complete = self.server_up_complete + self.standard_time

            for i in range(len(self.client_list)):
                if not self.client_state[i]:
                    time_from_server_to_client[i] = self.server_to_client[i] / download_speed[i]

            if self.chronous.lower().startswith('syn'):
                self.threshold = 1000 + self.standard_time

            else:
                need_client = [0]
                for i in range(len(self.client_state)):
                    if self.client_state[i]:
                        need_client.append(list_client_size[i])

                self.threshold = self.standard_time + max(max(time_from_server_to_client[:standard_client_number]),
                                                          self.server_up_complete - self.standard_time) + self.config.asyn_interval

            if debug:
                print('Second server_to_client' + str(self.server_to_client))
            for i in range(len(self.client_list)):
                if self.client_state[i]:
                    # 如果客户端一直是正常的，但是这一次客户端无法下载完成
                    if time_from_server_to_client[i] + self.standard_time > self.threshold:
                        self.list_store_model_client[i] = list_state_dict[i]
                        self.server_to_client[i] = (time_from_server_to_client[
                                                        i] + self.standard_time - self.threshold) * download_speed[i]
                        if self.server_to_client[i] < 0:
                            self.server_to_client[i] = 0.001
                        self.client_state[i] = False
                        time_from_server_to_client[i] = self.threshold - self.standard_time
                        assert self.server_to_client[i] > 0 or self.size_client_need_upload[i] > 0
                        assert self.server_to_client[i] == 0 or self.size_client_need_upload[i] == 0

                    else:
                        self.server_to_client[i] = 0
                        self.client_download_time[i].append(
                            time_from_server_to_client[i] + self.standard_time - begin_time_client_download[i])
                        self.sum_client_download_time[i] += self.client_download_time[i][-1]

                else:
                    assert self.server_to_client[i] > 0 or self.size_client_need_upload[i] > 0
                    assert self.server_to_client[i] == 0 or self.size_client_need_upload[i] == 0
                    if self.server_to_client[i] > 0:
                        if time_from_server_to_client[i] + self.standard_time > self.threshold:
                            self.server_to_client[i] -= download_speed[i] * (self.threshold - self.standard_time)
                            if self.server_to_client[i] < 0:
                                self.server_to_client[i] = 0.001
                            time_from_server_to_client[i] = self.threshold - self.standard_time
                        else:
                            self.server_to_client[i] = 0
                            time_from_server_to_client[i] = self.client_start_work_time[i]
                            assert self.server_to_client[i] == 0 and self.size_client_need_upload[i] == 0
                            self.client_state[i] = True
                            self.client_download_time[i].append(
                                time_from_server_to_client[i] + self.standard_time - begin_time_client_download[i])
                            self.sum_client_download_time[i] += self.client_download_time[i][-1]
                            list_state_dict[i] = self.list_store_model_client[i]

                    if self.size_client_need_upload[i] > 0:
                        assert self.server_to_client[i] == 0
                        time_from_server_to_client[i] = 0
            global_lr = None
            if self.server.ip_optimizer_wrapper != None and self.client_state[0]:

                self.server.ip_optimizer_wrapper.lr_scheduler_step()
                global_lr = self.server.ip_optimizer_wrapper.get_last_lr()

            for i in range(len(self.client_state)):
                if self.client_state[i]:
                    client = self.client_list[i]
                    client.load_state_dict(list_state_dict[i])
                    if global_lr is not None:
                        client.load_lr(global_lr)


            self.communicate_time_from_server_to_client = [self.standard_time + sc for sc in time_from_server_to_client]
            idx = self.train_number[0]











