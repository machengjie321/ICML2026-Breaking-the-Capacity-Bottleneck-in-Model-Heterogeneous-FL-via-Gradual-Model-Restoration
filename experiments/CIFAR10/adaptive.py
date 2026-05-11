import os


# os.chdir(r'D:\PR-FL')
# project_path = r"D:\PR-FL"
# import sys
# if project_path not in sys.path:
#     sys.path.append(project_path)
import torch
from networkx.algorithms.bipartite import density

from bases.fl.simulation_real.Partial_Model_Training import AdaptiveServer, AdaptiveClient, AdaptiveFL, parse_args
from bases.optim.optimizer import SGD, MaskedSGD
from torch.optim import lr_scheduler
from bases.optim.optimizer_wrapper import OptimizerWrapper
from bases.vision.load import get_data_loader
from control.sub_algorithm import ControlModule
from bases.vision.FLsampler import FLSampler
from bases.nn.models.vgg import VGG11
from configs.cifar10 import *
import configs.cifar10 as config

from utils.save_load import mkdir_save, load
from utils.functional import compute_same_params_ratio

class CIFAR10AdaptiveServer(AdaptiveServer):
    def init_test_loader(self):
        self.test_loader = get_data_loader(EXP_NAME, data_type="test", batch_size=1000, num_workers=8, pin_memory=True)

    def init_clients(self):
        rand_perm = torch.randperm(NUM_TRAIN_DATA).tolist()
        indices = []
        len_slice = NUM_TRAIN_DATA // num_slices

        for i in range(num_slices):
            indices.append(rand_perm[i * len_slice: (i + 1) * len_slice])

        models = [self.model for _ in range(NUM_CLIENTS)]
        self.indices = indices
        return models, indices

    def init_control(self):
        self.control = ControlModule(self.model, config=config)

    def init_ip_config(self):
        self.ip_train_loader = get_data_loader(EXP_NAME, data_type="train", batch_size=CLIENT_BATCH_SIZE,
                                               subset_indices=self.indices[0][:IP_DATA_BATCH * CLIENT_BATCH_SIZE],
                                               shuffle=True, num_workers=8, pin_memory=True)

        self.ip_test_loader = get_data_loader(EXP_NAME, data_type="test", batch_size=1000, num_workers=8,
                                              pin_memory=True)

        ip_optimizer = SGD(self.model.parameters(), lr=INIT_LR)
        self.ip_optimizer_wrapper = OptimizerWrapper(self.model, ip_optimizer)
        self.ip_control = ControlModule(model=self.model, config=config)

    def save_exp_config(self):
        pass


class CIFAR10AdaptiveClient(AdaptiveClient):
    def init_optimizer(self):
        self.optimizer = SGD(self.model.parameters(), lr=INIT_LR,)
        import torch.optim.lr_scheduler as lr_scheduler
        # self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.8, patience=8
        #                                                 , verbose=True)
        self.optimizer_scheduler = lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.5 ** (1 / LR_HALF_LIFE))
        self.optimizer_wrapper = OptimizerWrapper(self.model, self.optimizer, self.optimizer_scheduler)
        # self.optimizer_wrapper = OptimizerWrapper(self.model, self.optimizer,)

    def init_train_loader(self, tl):
        self.train_loader = tl


import torch
import numpy as np
import random
def set_seed(seed):
    torch.manual_seed(seed)  # PyTorch CPU
    np.random.seed(seed)  # NumPy
    random.seed(seed)  # Python 内置随机数
    torch.cuda.manual_seed(seed)  # PyTorch GPU（如果使用）
    torch.cuda.manual_seed_all(seed)  # 多 GPU
    torch.backends.cudnn.deterministic = True  # 确保 CUDA 计算可复现
    torch.backends.cudnn.benchmark = False  # 关闭自动优化（保证结果一致）

if __name__ == "__main__":

    args = parse_args()
    if not args.use_adaptive:
        args.experiment_name = 'PMT'
        set_seed(args.seed)



        num_users = 100
        num_slices = num_users if args.client_selection else NUM_CLIENTS
        # density_list = [ 0.3,0.4,0.5,0.6,0.7,0.8,0.9]
        density_list = [0.3,0.4,0.5,0.6,] 
        density_list.sort(reverse=True)
        args.MaskSGD = False
        for density in density_list:
            args.experiment_name = 'PMT'+ str(density)
            args.target_density = density

            server = CIFAR10AdaptiveServer(args, config, VGG11())
            list_models, list_indices = server.init_clients()

            sampler = FLSampler(list_indices, MAX_ROUND, NUM_LOCAL_UPDATES * CLIENT_BATCH_SIZE, args.client_selection,
                                num_slices)
            print("Sampler initialized")

            train_loader = get_data_loader(EXP_NAME, data_type="train", batch_size=CLIENT_BATCH_SIZE, shuffle=False,
                                           sampler=sampler, num_workers=8, pin_memory=True)

            client_list = [CIFAR10AdaptiveClient(list_models[idx], config, args.use_adaptive,args) for idx in range(NUM_CLIENTS)]
            for client in client_list:
                client.init_optimizer()
                client.init_train_loader(train_loader)



            fl_runner = AdaptiveFL(args, config, server, client_list)
            final_acc,final_std = fl_runner.main()
    if args.use_adaptive:
        final_result = {}
        save_path = os.path.join("results", config.EXP_NAME, 'PMT', 'split')
        # 遍历文件夹中的所有文件和子文件夹
        fil_dir = os.listdir(save_path)
        # fil_dir.sort(reverse=True)
        for filename in fil_dir:
            full_path = os.path.join(save_path,filename ,'checkpoint.pth')
            checkpoint = load(full_path)
            for var_name, value in checkpoint.items():
                if var_name.startswith('self.model'):
                    model = value
            density_list = [0.025,0.05,0.1,0.2,0.3, 0.5, 0.8, 1.0]
            # density_list.sort(reverse=True)
            for target_density in density_list:
                args.experiment_name = 'PMT_'+filename+'_'+str(target_density)
                args.target_density = target_density
                set_seed(args.seed)
                num_users = 100
                num_slices = num_users if args.client_selection else NUM_CLIENTS

                server = CIFAR10AdaptiveServer(args, config, model)
                list_models, list_indices = server.init_clients()

                sampler = FLSampler(list_indices, MAX_ROUND, NUM_LOCAL_UPDATES * CLIENT_BATCH_SIZE,
                                    args.client_selection,
                                    num_slices)
                train_loader = get_data_loader(EXP_NAME, data_type="train", batch_size=CLIENT_BATCH_SIZE, shuffle=False,
                                               sampler=sampler, num_workers=8, pin_memory=True)

                client_list = [CIFAR10AdaptiveClient(list_models[idx], config, args.use_adaptive, args) for idx in
                               range(NUM_CLIENTS)]
                for client in client_list:
                    client.init_optimizer()
                    client.init_train_loader(train_loader)



                fl_runner = AdaptiveFL(args, config, server, client_list)
                final_acc, final_std = fl_runner.main()

                for var_name, value in checkpoint.items():
                    if var_name.startswith('self.model'):
                        model = value
                same_ratio = compute_same_params_ratio(model, server.model)

                final_result[args.experiment_name] = [final_acc, same_ratio, final_std]
                print(args.experiment_name + ":" + str(final_result[args.experiment_name]))

        for key in final_result.keys():
            print(key, ":", str(final_result[key]))
        import json
        # 保存为 txt 文件
        with open("output.txt", "w", encoding="utf-8") as f:
            json.dump(final_result, f, ensure_ascii=False, indent=4)  # 美化格式，支持 UTF-8