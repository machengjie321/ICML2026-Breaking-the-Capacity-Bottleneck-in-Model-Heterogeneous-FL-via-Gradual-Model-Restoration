import os

# os.chdir(r'D:\PR-FL')
# project_path = r"D:\PR-FL"
# import sys
# if project_path not in sys.path:
#     sys.path.append(project_path)

from bases.fl.simulation_real.Partial_Model_Training import AdaptiveServer, AdaptiveClient, AdaptiveFL, parse_args
from bases.optim.optimizer import SGD, MaskedSGD
from bases.optim.optimizer_wrapper import OptimizerWrapper
from bases.vision.load import get_data_loader
from bases.nn.models.leaf import Conv2
from bases.vision.FLsampler import FLSampler
from control.sub_algorithm import ControlModule
from configs.femnist import *
import configs.femnist as config
from utils.save_load import mkdir_save, load
from utils.functional import compute_same_params_ratio
import sys


class FEMNISTAdaptiveServer(AdaptiveServer):
    def init_test_loader(self):
        self.test_loader = get_data_loader(EXP_NAME, data_type="test", num_workers=2, pin_memory=True)

    def init_clients(self):
        if self.client_selection:
            list_usr = [[i] for i in range(num_users)]
        else:
            nusr = num_users // NUM_CLIENTS  # num users for the first NUM_CLIENTS - 1 clients
            list_usr = [list(range(nusr * j, nusr * (j + 1) if j != NUM_CLIENTS - 1 else num_users)) for j in
                        range(NUM_CLIENTS)]
        models = [self.model for _ in range(NUM_CLIENTS)]
        return models, list_usr

    def init_control(self):
        self.control = ControlModule(self.model, config=config)


    def init_ip_config(self):
        self.ip_train_loader = get_data_loader(EXP_NAME, data_type="train", batch_size=CLIENT_BATCH_SIZE, shuffle=True,
                                               num_workers=8, user_list=[0], pin_memory=True)
        self.ip_test_loader = get_data_loader(EXP_NAME, data_type="test", num_workers=8, pin_memory=True)
        ip_optimizer = SGD(self.model.parameters(), lr=INIT_LR)
        self.ip_optimizer_wrapper = OptimizerWrapper(self.model, ip_optimizer)
        self.ip_control = ControlModule(model=self.model, config=config)


    def save_exp_config(self):
        pass


class FEMNISTAdaptiveClient(AdaptiveClient):
    def init_optimizer(self):
        self.optimizer = SGD(self.model.parameters(), lr=INIT_LR)
        if self.args.MaskSGD:
            self.optimizer = MaskedSGD(self.model.named_parameters(), lr=INIT_LR)
        self.optimizer_wrapper = OptimizerWrapper(self.model, self.optimizer)

    def init_train_loader(self, tl):
        self.train_loader = tl


def get_indices_list():
    cur_pointer = 0
    indices_list = []
    for ul in list_users:
        num_data = 0
        for user_id in ul:
            train_meta = torch.load(os.path.join("datasets", "FEMNIST", "processed", "train_{}.pt".format(user_id)),weights_only=True)
            num_data += len(train_meta[0])
        indices_list.append(list(range(cur_pointer, cur_pointer + num_data)))
        cur_pointer += num_data

    return indices_list



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
    import json
    args = parse_args()
    args.MaskSGD = False

    if not args.use_adaptive:
        set_seed(args.seed)

        args.experiment_name = 'PMT'

        density_list = [0.025, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
        density_list.sort(reverse=True)

        num_user_path = os.path.join("datasets", "FEMNIST", "processed", "num_users.pt")
        if not os.path.isfile(num_user_path):
            get_data_loader(EXP_NAME, data_type="train", batch_size=CLIENT_BATCH_SIZE, shuffle=False, num_workers=0,
                            pin_memory=True)
        num_users = torch.load(num_user_path,weights_only=True)


        for density in density_list:
            args.experiment_name = 'PMT'+ str(density)
            args.target_density = density
            server = FEMNISTAdaptiveServer(args, config, Conv2())
            list_models, list_users = server.init_clients()

            # get_indices_list() indicates what the subscripts are for the data held on each client
            sampler = FLSampler(get_indices_list(), MAX_ROUND, NUM_LOCAL_UPDATES * CLIENT_BATCH_SIZE, args.client_selection,
                                NUM_CLIENTS)
            print("Sampler initialized")

            train_loader = get_data_loader(EXP_NAME, data_type="train", batch_size=CLIENT_BATCH_SIZE, shuffle=False,
                                           sampler=sampler, num_workers=0, pin_memory=True)
            # get_data_loader(EXP_NAME, data_type="train", batch_size=CLIENT_BATCH_SIZE, shuffle=False, subset_indices=)


            client_list = [FEMNISTAdaptiveClient(list_models[idx], config, args.use_adaptive,args) for idx in range(NUM_CLIENTS)]
            for client in client_list:
                client.init_optimizer()
                client.init_train_loader(train_loader)

            print("All initialized. Experiment is {}. Use adaptive = {}. Client selection = {}. "
                  "Num users = {}. Seed = {}. Max round = {}. "
                  "Target density = {}".format(EXP_NAME, args.use_adaptive, args.client_selection,
                                               num_users, args.seed, MAX_ROUND, args.target_density))

            fl_runner = AdaptiveFL(args, config, server, client_list)
            final_acc,final_std = fl_runner.main()
            print(final_acc, final_std)

    if args.use_adaptive:
        final_result = {}
        save_path = os.path.join("results", config.EXP_NAME, 'PMT', 'split')
        fil_dir = os.listdir(save_path)
        fil_dir = fil_dir[3:-1]
        fil_dir.sort(reverse=True)

        # fil_dir = ['0.83']
        # 遍历文件夹中的所有文件和子文件夹
        from datetime import datetime
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(str(args.MaskSGD) + str(config.MAX_TIME_PMT) + "output.txt", "a", encoding="utf-8") as f:
            f.write(f"\n==== New Entry ({current_time}) ====\n")  # 记录写入时间
            f.write("\n")

        for filename in fil_dir:
            full_path = os.path.join(save_path,filename ,'checkpoint.pth')
            checkpoint = load(full_path)
            for var_name, value in checkpoint.items():
                if var_name.startswith('self.model'):
                    model = value
            density_list = [0.025,0.05,0.1,0.2,0.3,0.4,0.6,0.8,1.0]
            density_list = [ 0.4, 0.6, 0.8, 1.0]
            density_list.sort(reverse=True)
            for target_density in density_list:

                args.experiment_name = 'PMT_'+filename+'_'+str(target_density)
                args.target_density = target_density
                set_seed(args.seed)

                num_user_path = os.path.join("datasets", "FEMNIST", "processed", "num_users.pt")
                if not os.path.isfile(num_user_path):
                    get_data_loader(EXP_NAME, data_type="train", batch_size=CLIENT_BATCH_SIZE, shuffle=False, num_workers=0,
                                    pin_memory=True)
                num_users = torch.load(num_user_path,weights_only=True)
                import copy
                server = FEMNISTAdaptiveServer(args, config, copy.deepcopy(model))
                list_models, list_users = server.init_clients()

                # get_indices_list() indicates what the subscripts are for the data held on each client
                sampler = FLSampler(get_indices_list(), MAX_ROUND, NUM_LOCAL_UPDATES * CLIENT_BATCH_SIZE,
                                    args.client_selection,
                                    NUM_CLIENTS)


                train_loader = get_data_loader(EXP_NAME, data_type="train", batch_size=CLIENT_BATCH_SIZE, shuffle=False,
                                               sampler=sampler, num_workers=2, pin_memory=True)
                # get_data_loader(EXP_NAME, data_type="train", batch_size=CLIENT_BATCH_SIZE, shuffle=False, subset_indices=)

                client_list = [FEMNISTAdaptiveClient(list_models[idx], config, args.use_adaptive, args) for idx in
                               range(NUM_CLIENTS)]
                for client in client_list:
                    client.init_optimizer()
                    client.init_train_loader(train_loader)


                fl_runner = AdaptiveFL(args, config, server, client_list)
                final_acc,final_prune_acc = fl_runner.main()
                for var_name, value in checkpoint.items():
                    if var_name.startswith('self.model'):
                        model = value
                same_ratio = compute_same_params_ratio(model, server.model)

                final_result[args.experiment_name] = [final_acc, final_prune_acc, same_ratio,]
                print(args.experiment_name+":"+str(final_result[args.experiment_name]))
                # 保存为 txt 文件
                with open(str(args.MaskSGD) + str(config.MAX_TIME_PMT) + "output.txt", "a", encoding="utf-8") as f:
                    json.dump(str(args.experiment_name)+str(final_result[args.experiment_name]), f, ensure_ascii=False, indent=4)  # 美化格式，支持 UTF-8
                    f.write("\n")







        for key in final_result.keys():
            print(key, ":", str(final_result[key]))



