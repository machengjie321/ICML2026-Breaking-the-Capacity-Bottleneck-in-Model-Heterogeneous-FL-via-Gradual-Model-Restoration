import argparse
import os
from distutils.command.clean import clean
from typing import final

import numpy as np
from copy import deepcopy
import torch
from utils.save_load import mkdir_save
from utils.functional import disp_num_params, copy_dict
from timeit import default_timer as timer
from utils.functional import deepcopy_dict
from abc import ABC, abstractmethod
import copy

masksgd = False

def parse_args():
    parser = argparse.ArgumentParser()
    mutex = parser.add_mutually_exclusive_group(required=True)
    mutex.add_argument('-a', '--adaptive',
                       help="Use adaptive pruning",
                       action='store_true',
                       dest='use_adaptive')

    mutex.add_argument('-na', '--no-adaptive',
                       help="Do not use adaptive pruning",
                       action='store_false',
                       dest='use_adaptive')



    parser.add_argument('-c', '--client-selection',
                        help="If use client-selection",
                        action='store_true',
                        dest='client_selection',
                        default=False,
                        required=False)

    parser.add_argument('-t', '--target-density',
                        help="Target density",
                        action='store',
                        dest='target_density',
                        type=float,
                        default=1.0,
                        required=False)

    parser.add_argument('-m', '--max-density',
                        help="Max density",
                        action='store',
                        dest='max_density',
                        type=float,
                        required=False)

    parser.add_argument('-s', '--seed',
                        help="The seed to use for the prototype",
                        action='store',
                        dest='seed',
                        type=int,
                        default=0,
                        required=False)

    parser.add_argument('-e', '--exp-name',
                        help="Experiment name",
                        action='store',
                        dest='experiment_name',
                        type=str,
                        default='test',
                        required=False)

    parser.add_argument('-g', '--gpu',
                        help="gpu_device",
                        action='store',
                        dest='device',
                        type=str,
                        default=0,
                        required=False)

    parser.add_argument('-fast', '--fast_prune',
                        help="fast_prune",
                        action='store_true',
                        dest='fast',
                        default=False,
                        required=False)

    return parser.parse_args()
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=10, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        from collections import deque
        self.old_score = deque(maxlen=(patience//2))
        self.average_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta


    def __call__(self, val_loss):
        '''
            功能：早停法 计算函数
            input:
                val_loss         验证损失
                model            模型
                model_path       模型保存地址
        '''
        score = val_loss
        self.old_score.append(score)
        self.average_score = sum(self.old_score)/len(self.old_score)

        if self.best_score is None:
            self.best_score = score
            return True
        elif score <= self.best_score + self.delta:
            self.counter += 1

            print(f'EarlyStopping counter: {self.counter} out of {self.patience}, best score is {self.best_score}, current score is {score}')

            # if self.counter >= self.patience and score >= self.average_score:
            #     logger.info('out of the patience, but score is big than average_score')
            #     print('out of the patience, but score is big than average_score')
            if self.counter >= self.patience:

                print('out of patience')
                self.early_stop = True
                return False
        else:
            self.best_score = score
            self.counter = 0
            return True
#experiments/FEMNIST/PMT.py -a -i -s 0 -e

class AdaptiveServer(ABC):
    def __init__(self, args, config, model, save_interval=50):
        self.config = config
        self.device = torch.device("cuda:"+str(args.device) if torch.cuda.is_available() else "cpu")
        self.experiment_name = args.experiment_name
        self.save_path = os.path.join("results", config.EXP_NAME, args.experiment_name)
        self.save_interval = save_interval
        self.args = args
        self.use_adaptive = args.use_adaptive
        self.client_selection = args.client_selection
        patience = self.config.patience
        if args.target_density == 1.0:
            patience = patience *3
        self.early_stoping = EarlyStopping(patience=int(patience))
        self.target_density = args.target_density
        from collections import deque
        self.save_for_split = deque(config.PMT_ACC)
        self.sp = self.save_for_split.popleft()
        self.test_split_statu = True
        self.test_split_acc = 0
        self.last_test_time = 0
        self.interval_time = 10
        self.full_time = 0
        self.part_time = []
        self.est_time = 0
        self.model_size = 0
        self.mask = None
        





        self.model = model.to(self.device)
        self.model.train()
        
        self.old_model = deepcopy(self.model.state_dict())


        if not self.use_adaptive:
            mkdir_save(self.model, os.path.join(self.save_path, "init_model.pt"))

        self.indices = None

        self.ip_train_loader = None
        self.ip_test_loader = None
        self.ip_optimizer_wrapper = None
        self.ip_control = None

        self.test_loader = None
        self.control = None
        self.init_test_loader()
        self.init_clients()
        self.list_prune_loss,self.list_prune_acc = [],[]
        self.init_control()
        self.init_ip_config()
        self.time = []
        self.inital_training = True


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

    def initial_pruning(self, list_est_time = None , list_loss = None, list_acc = [], list_model_size = None):
        svdata, pvdata = self.ip_train_loader.len_data, self.config.IP_DATA_BATCH * self.config.CLIENT_BATCH_SIZE
        assert svdata >= pvdata, "server data ({}) < required data ({})".format(svdata, pvdata)
        server_inputs, server_outputs = [], []
        dev = self.device
        for _ in range(self.config.IP_DATA_BATCH):
            inp, out = self.ip_train_loader.get_next_batch()
            server_inputs.append(inp.to(dev))
            server_outputs.append(out.to(dev))

        prev_density = None
        prev_num = 5
        prev_ind = []
        start = timer()


        steps = int(self.config.IP_MAX_ROUNDS/self.config.IP_ADJ_INTERVAL)+1
        # r = (self.target_density / 1) ** (1 / (steps - 1))  # 公比
        # series = 1 * r ** np.arange(steps)
        # print(series)
        for server_i in range(1, self.config.IP_MAX_ROUNDS + 1):
            model_size = self.model.calc_num_all_active_params(True)



            for server_inp, server_out in zip(server_inputs, server_outputs):
                list_grad,_ = self.ip_optimizer_wrapper.step(server_inp, server_out)
                if self.config.EXP_NAME =='stackoverflow':
                    self.control.accumulate_wg_square(self.old_model)
                    self.old_model = copy.deepcopy(self.model)
                else:
                    for (key, param), g in zip(self.model.named_parameters(), list_grad):
                        assert param.size() == g.size()
                        self.control.accumulate(key, g ** 2)

            if server_i  % self.config.IP_ADJ_INTERVAL == 0:
                loss, acc = self.model.evaluate(self.ip_test_loader)
                print("Inital Pruning Round #{} (Experiment = {}).".format(server_i, self.experiment_name))
                print("Loss/acc (at round #{}) = {}/{}".format(server_i, loss,
                                                                   acc))
                # grad_dict = self.control.accumulate_weight_dict

                # abs_all_wg = None
                # o_g = None
                # for x in grad_dict.values():
                #     if o_g is None:
                #         o_g =  x.view(-1).abs()
                #     else:
                #         o_g = torch.cat([o_g, (x).view(-1).abs()], dim=0)
                # nonzero_g = o_g[o_g != 0]

                # min_g = torch.min(nonzero_g)

                # for (name_g, g) in grad_dict.items():
                #     g[g==0] = min_g
                #     if abs_all_wg is None:
                #         abs_all_wg = (g).view(-1).abs()
                #     else:
                #         abs_all_wg = torch.cat([abs_all_wg, (g).view(-1).abs()], dim=0)

                # threshold = abs_all_wg.sort(descending=True)[0][int(series[int(server_i/self.config.IP_ADJ_INTERVAL)] * abs_all_wg.nelement())-1]


                # for layer, layer_prefix in zip(self.model.prunable_layers, self.model.prunable_layer_prefixes):
                #     abs_layer_wg = (grad_dict[layer_prefix + ".weight"]).abs()
                #     layer.mask = abs_layer_wg >= threshold


                # with torch.no_grad():
                #     for layer in self.model.prunable_layers:
                #         layer.weight *= layer.mask


                # self.control.accumulate_weight_dict = dict()
                # # cur_density = disp_num_params(self.model)



        len_pre_rounds = len(list_acc)
        print("End initial pruning. Total rounds = {}. Total elapsed time = {}.".format(
            len_pre_rounds * self.config.EVAL_DISP_INTERVAL, timer() - start))

        return len_pre_rounds

    def save_display_data(self,list_loss,list_acc,list_est_time,list_model_size):
        mkdir_save(list_loss, os.path.join(self.save_path, "loss.pt"))
        mkdir_save(list_acc, os.path.join(self.save_path, "fed_avg_acc.pt"))
        mkdir_save(list_est_time, os.path.join(self.save_path, "est_time.pt"))
        mkdir_save(list_model_size, os.path.join(self.save_path, "model_size.pt"))
        mkdir_save(self.model, os.path.join(self.save_path, "model.pt"))
        mkdir_save(self.time, os.path.join(self.save_path, "time.pt"))

    def get_real_model(self):

        clean_model = copy.deepcopy(self.model)
        clean_state_dict = clean_model.state_dict()
        for layer, prefix in zip(self.model.prunable_layers, self.model.prunable_layer_prefixes):
            # works for both layers
            key_w = prefix + ".weight"
            if key_w in self.model.state_dict().keys():
                weight = self.model.state_dict()[key_w]
                w_mask = self.mask[key_w]
                real_weight = (weight * w_mask)
                clean_state_dict[key_w] = real_weight
        clean_model.load_state_dict(clean_state_dict)
        for layer in clean_model.prunable_layers:
            mask = layer.state_dict()['weight'] != 0
            layer.mask.copy_(mask)
        return clean_model

    def save_checkpoint_for_split(self, acc, fedavg_model):
        '''
            功能：当验证损失减少时保存模型
            input:
                val_loss         验证损失
                model            模型
                model_path       模型保存地址
        '''

        checkpoint = {'self.model': fedavg_model,
                          'self.control.accumulate_weight_dict':self.control.accumulate_weight_dict
                          }
        checkpoint_path = os.path.join(self.save_path, 'split', str(acc), 'checkpoint.pth')

        mkdir_save(checkpoint, checkpoint_path)

    def test_pruning_performance(self,avg_acc,fed_avg_model):
        if len(self.save_for_split) > 0 and avg_acc > self.save_for_split[0]:
            for i in range(len(self.save_for_split)):

                if self.save_for_split[0] > avg_acc:
                    break
                self.sp = self.save_for_split.popleft()
            self.test_split_statu = True
        if self.test_split_statu and avg_acc >= self.sp:
            self.save_checkpoint_for_split(self.sp, fed_avg_model)
            self.test_split_acc = avg_acc
            self.test_split_statu = False

        elif not self.test_split_statu and self.sp <= avg_acc < self.test_split_acc:
            self.save_checkpoint_for_split(self.sp, fed_avg_model)
            self.test_split_acc = avg_acc
            self.test_split_statu = False

    @torch.no_grad()
    def load_mask(self):
        masks = self.mask
        with torch.no_grad():
            server_dict = self.model.state_dict()
            for key, param in self.model.state_dict().items():
                if key in masks.keys():
                    server_dict[key] = server_dict[key] * masks[key]
            self.model.load_state_dict(server_dict)
            for layer in self.model.prunable_layers:
                mask = layer.state_dict()['weight'] != 0
                layer.mask.copy_(mask)








    def main(self, idx, list_sd, list_num_proc, lr,  start, list_loss, list_acc, list_est_time,
             list_model_size, density_limit=None):
        total_num_proc = sum(list_num_proc)
        grad_dict = dict()
        weight_dict = dict()
        with torch.no_grad():
            for key, param in self.model.state_dict().items():
                avg_inc_val = None
                for num_proc, state_dict in zip(list_num_proc, list_sd):
                    if key in state_dict.keys():
                        inc_val = state_dict[key] - param

                        if avg_inc_val is None:
                            avg_inc_val = num_proc / total_num_proc * inc_val
                        else:
                            avg_inc_val += num_proc / total_num_proc * inc_val

                if avg_inc_val is None or key.endswith("num_batches_tracked"):
                    continue
                else:
                    param.add_(avg_inc_val)      


        

        if not self.use_adaptive:
            
            current_time = sum(list_est_time)

            if current_time - self.last_test_time > int(self.interval_time) or current_time == 0:
                self.last_test_time = current_time
                
                loss, acc = self.model.evaluate(self.test_loader)
                # disp_num_params(self.model)
                list_loss.append(loss)
                list_acc.append(acc)
                self.time.append(sum(list_est_time))

                if not self.use_adaptive:

                    print("Round #{} (Experiment = {}).".format(idx, self.experiment_name))
                    print("Loss/acc (at round #{}) = {}/{}".format(idx, loss,
                                                                   acc))
                    print("Estimated time = {}".format(self.time[-1]))
                    print("Elapsed time = {}".format(timer() - start))
                    print("Current lr = {}".format(lr))
                    disp_num_params(self.model)
                    self.early_stoping(acc)


        else:
            current_time = sum(list_est_time)
            if current_time - self.last_test_time > int(self.config.MAX_TIME_PMT/15) or current_time == 0:
                self.last_test_time = current_time
                loss, acc = self.model.evaluate(self.test_loader)
                clean_model = self.get_real_model()
                prune_loss, prune_acc = clean_model.evaluate(self.test_loader)

                # disp_num_params(self.model)
                disp_num_params(clean_model)
                list_loss.append(loss)
                list_acc.append(acc)
                self.list_prune_acc.append(prune_acc)
                self.list_prune_loss.append(prune_loss)
                if not self.use_adaptive:
                    window = 10
                    if len(list_acc) < window:
                        window = len(list_acc)-1
                    if len(list_loss) >1:
                        final_acc = np.mean(np.array(list_acc[-1 - window:-1]))
                    else:
                        final_acc = list_acc[-1]
                    # self.test_pruning_performance(final_acc,self.model)

                    print("Round #{} (Experiment = {}).".format(idx, self.experiment_name))
                    print("Loss/acc (at round #{}) = {}/{}".format(idx, loss,
                                                                   final_acc))
                    print("Estimated time = {}".format(current_time))
                    print("Elapsed time = {}".format(timer() - start))
                    print("Current lr = {}".format(lr))
                    self.early_stoping(acc)



                self.old_model = deepcopy(self.model.state_dict())
                if self.config.EXP_NAME =='stackoverflow':
                    pass                
                else:
                    self.mask = self.control.partial_model_training([self.target_density]*len(list_sd))







        if self.config.EXP_NAME =='stackoverflow':
            self.est_time = 1 * self.target_density   

 
        else:
            if len(list_est_time) == 0:
                full_time = self.config.TIME_CONSTANT
            
                for layer, comp_coeff in zip(self.model.prunable_layers, self.config.COMP_COEFFICIENTS):
                    full_time += layer.mask.nelement() * (comp_coeff + self.config.COMM_COEFFICIENT)

                self.interval_time = self.config.EVAL_DISP_INTERVAL*full_time


                self.est_time = self.config.TIME_CONSTANT

                for layer, comp_coeff in zip(self.model.prunable_layers, self.config.COMP_COEFFICIENTS):
                    self.est_time += layer.mask.sum() * (comp_coeff + self.config.COMM_COEFFICIENT)
                self.model_size = self.model.calc_num_all_active_params(True)
        

        
        # print(self.model.prunable_layers[0].weight)
        list_est_time.append(self.est_time)
        # print(est_time)
        list_model_size.append(self.model_size)

        if idx % self.config.EVAL_DISP_INTERVAL == 0 and not self.use_adaptive:
            self.save_display_data(list_loss, list_acc, list_est_time, list_model_size)
        # clean_model = self.get_real_model()
            
        # if self.args.MaskSGD:
        #     clean_model = self.model
        self.load_mask()


        return  self.mask, [self.model.state_dict() for _ in range(self.config.NUM_CLIENTS)]


class AdaptiveClient:
    def __init__(self, model, config, use_adaptive, args):
        self.config = config
        self.device = torch.device("cuda:"+str(args.device) if torch.cuda.is_available() else "cpu")
        self.use_adaptive = use_adaptive
        self.model = deepcopy(model).to(self.device)
        self.optimizer = None
        self.optimizer_scheduler = None
        self.optimizer_wrapper = None
        self.train_loader = None
        self.args = args
        self.mask = None
        self.list_mask = [None for _ in range(len(self.model.prunable_layers))]
        self.test_loader = None




    @abstractmethod
    def init_optimizer(self, *args, **kwargs):
        pass

    @abstractmethod
    def init_train_loader(self, *args, **kwargs):
        pass

    def main(self):
        self.model.train()
        num_proc_data = 0

        lr = self.optimizer_wrapper.get_last_lr()

        accumulated_grad = dict()
        for _ in range(self.config.NUM_LOCAL_UPDATES):

            inputs, labels = self.train_loader.get_next_batch()
            if not self.args.MaskSGD:
                list_grad,loss = self.optimizer_wrapper.step(inputs.to(self.device), labels.to(self.device))

            else:
                self.optimizer_wrapper.step2(inputs.to(self.device), labels.to(self.device), self.mask)
                # loss, acc = self.model.evaluate(self.test_loader)
                # print(acc)

            num_proc_data += len(inputs)

        # loss, acc = self.model.evaluate(self.test_loader)
        # print(acc)

        self.optimizer_wrapper.lr_scheduler_step()
        # disp_num_params(self.model)


        return self.model.state_dict(), num_proc_data,  lr

    def load_mask(self, masks):
        
        self.mask = masks

    # def load_state_dict(self, state_dict):
    #     self.model.load_state_dict(state_dict)
    @torch.no_grad()
    def load_state_dict(self, state_dict):
        
        
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
            
class AdaptiveFL(ABC):
    def __init__(self, args, config, server, client_list):
        self.config = config
        self.args = args

        self.use_adaptive = args.use_adaptive
        self.tgt_d, self.max_d = args.target_density, 1.0
        self.use_fast = args.fast
        self.max_round = config.MAX_ROUND

        self.server = server
        self.client_list = client_list

        self.list_loss, self.list_acc, self.list_est_time, self.list_model_size = [], [], [], []
        self.start_adj_round = None


    def main(self):
        len_pre_rounds = 0

        for client in self.client_list:

            client.test_loader = self.server.test_loader


        start = timer()
        # len_pre_rounds = self.server.initial_pruning(self.list_est_time, self.list_loss, self.list_acc,
        #                                                  self.list_model_size)
        
        
        if self.config.EXP_NAME =='stackoverflow':   

    
            self.server.mask = self.server.control.partial_model_training([self.server.target_density]*len(self.client_list),use_coff = False)

            
        print("Clients loading server model...")
        for client in self.client_list:
            client.load_state_dict(self.server.model.state_dict())
            # client.load_mask([layer.mask for layer in self.server.model.prunable_layers])
        idx = 1
        while True:
            # print(idx)
            list_state_dict, list_num, list_last_lr = [], [], []



            if self.server.early_stoping.early_stop:
                print('early stop at round'+str(idx))
                break
            if self.server.use_adaptive:
                if len(self.list_est_time)>0 and sum(self.list_est_time) > self.config.MAX_TIME_PMT:
                    print('Stop at round'+str(idx))
                    break

            for client in self.client_list:
                sd, npc, last_lr = client.main()
                list_state_dict.append(sd)
                list_num.append(npc)

                list_last_lr.append(last_lr)
            last_lr = list_last_lr[0]



            density_limit = None
            if self.max_d is not None:
                density_limit = self.max_d

            masks, new_list_sd = self.server.main(idx, list_state_dict, list_num, last_lr, start,
                                                      self.list_loss, self.list_acc, self.list_est_time,
                                                      self.list_model_size, density_limit)

            for client, new_sd in zip(self.client_list, new_list_sd):
                client.load_state_dict(new_sd)
                # client.load_mask(masks)
            idx = idx+1



        window = 10
        if len(self.list_acc)<window:
            window = len(self.list_acc)
        final_acc = np.array(self.list_acc[-1-window:-1])
        final_prune_acc = np.array(self.server.list_prune_acc[-1 - window:-1])


        return np.mean(final_acc)*100,np.mean(final_prune_acc)*100



