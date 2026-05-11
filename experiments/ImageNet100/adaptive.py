import os


# os.chdir(r'D:\PR-FL')
# project_path = r"D:\PR-FL"
# import sys
# if project_path not in sys.path:
#     sys.path.append(project_path)
    
import torch
from bases.fl.simulation_real.Partial_Model_Training import AdaptiveServer, AdaptiveClient, AdaptiveFL, parse_args
from bases.optim.optimizer import SGD, MaskedSGD
from torch.optim import lr_scheduler
from bases.optim.optimizer_wrapper import OptimizerWrapper
from bases.vision.load import get_data_loader
from control.sub_algorithm import ControlModule

from bases.nn.models.resnet import resnet18
from configs.imagenet100 import *
import configs.imagenet100 as config

from utils.save_load import mkdir_save, load
from bases.vision.FLsampler import FLSampler
from utils.functional import compute_same_params_ratio

class INAdaptiveServer(AdaptiveServer):
    def init_test_loader(self):
        self.test_loader = get_data_loader(EXP_NAME, data_type="val", batch_size=200, num_workers=2, pin_memory=True)

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

        self.ip_test_loader = get_data_loader(EXP_NAME, data_type="val", batch_size=200, num_workers=8,
                                              pin_memory=True)

        ip_optimizer = SGD(self.model.parameters(), lr=INIT_LR)
        self.ip_optimizer_wrapper = OptimizerWrapper(self.model, ip_optimizer)
        self.ip_control = ControlModule(model=self.model, config=config)


    def save_exp_config(self):
        pass


class INAdaptiveClient(AdaptiveClient):
    def init_optimizer(self):
        self.optimizer = SGD(self.model.parameters(), lr=INIT_LR, momentum=0, weight_decay=0)
        self.optimizer_scheduler = lr_scheduler.StepLR(self.optimizer, step_size=STEP_SIZE,
                                                       gamma=0.5 ** (STEP_SIZE / LR_HALF_LIFE))
        self.optimizer_wrapper = OptimizerWrapper(self.model, self.optimizer, self.optimizer_scheduler)

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
        args.MaskSGD = False
        num_users = 100
        num_slices = num_users if args.client_selection else NUM_CLIENTS
        density_list = [0.7,0.9]
        density_list.sort(reverse=True)

        for density in density_list:
            args.experiment_name = 'PMT'+ str(density)
            args.target_density = density
            server = INAdaptiveServer(args, config, resnet18(NUM_CLASSES))
            list_models, list_indices = server.init_clients()

            sampler = FLSampler(list_indices, MAX_ROUND, NUM_LOCAL_UPDATES * CLIENT_BATCH_SIZE, args.client_selection,
                                num_slices)

            train_loader = get_data_loader(EXP_NAME, data_type="train", batch_size=CLIENT_BATCH_SIZE, shuffle=False,
                                           sampler=sampler, num_workers=config.train_num, pin_memory=True)

            client_list = [INAdaptiveClient(list_models[idx], config, args.use_adaptive,args) for idx in range(NUM_CLIENTS)]
            for client in client_list:
                client.init_optimizer()
                client.init_train_loader(train_loader)


            fl_runner = AdaptiveFL(args, config, server, client_list)
            final_acc,final_std = fl_runner.main()

    if args.use_adaptive:
        final_result = {}
        save_path = os.path.join("results", config.EXP_NAME, 'PMT', 'split')
        # 遍历文件夹中的所有文件和子文件夹
        for filename in os.listdir(save_path):
            full_path = os.path.join(save_path,filename ,'checkpoint.pth')
            checkpoint = load(full_path)
            for var_name, value in checkpoint.items():
                if var_name.startswith('self.model'):
                    model = value

            for target_density in [0.025,0.05,0.1,0.2,0.3, 0.5, 0.8, 1.0]:

                args.experiment_name = 'PMT_'+filename+'_'+str(target_density)
                args.target_density = target_density
                set_seed(args.seed)

                num_users = 100
                num_slices = num_users if args.client_selection else NUM_CLIENTS

                server = INAdaptiveServer(args, config, model)
                list_models, list_indices = server.init_clients()

                sampler = FLSampler(list_indices, MAX_ROUND, NUM_LOCAL_UPDATES * CLIENT_BATCH_SIZE,
                                    args.client_selection,
                                    num_slices)

                train_loader = get_data_loader(EXP_NAME, data_type="train", batch_size=CLIENT_BATCH_SIZE, shuffle=False,
                                               sampler=sampler, num_workers=config.train_num, pin_memory=True)

                client_list = [INAdaptiveClient(list_models[idx], config, args.use_adaptive, args) for idx in
                               range(NUM_CLIENTS)]
                for client in client_list:
                    client.init_optimizer()
                    client.init_train_loader(train_loader)

                fl_runner = AdaptiveFL(args, config, server, client_list)
                final_acc, final_std = fl_runner.main()
                same_ratio = compute_same_params_ratio(model, server.model)
