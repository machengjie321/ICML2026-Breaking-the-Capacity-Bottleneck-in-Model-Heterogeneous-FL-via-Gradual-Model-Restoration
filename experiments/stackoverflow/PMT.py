import os


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
from bases.optim.optimizer import SGD
from bases.optim.optimizer_wrapper import OptimizerWrapper
from bases.vision.load import get_data_loader
from bases.nn.models.bp_transformer import transformer
from configs.StackOverflow import *
import configs.StackOverflow as config
from control.utils import ControlScheduler

from utils.functional import select_best_gpu
from utils.save_load import mkdir_save

class FEMNISTAdaptiveServer(AdaptiveServer):
    def init_test_loader(self):
        self.test_loader = get_data_loader(EXP_NAME, data_type="val", num_workers=2, pin_memory=True)

    def init_clients(self):
        models = [self.model for _ in range(NUM_CLIENTS)]
        if True:
            N = NUM_TRAIN_DATA
            num_slices = 200 if args.client_selection else  NUM_CLIENTS

            # 随机打乱 0..N-1
            g = torch.Generator().manual_seed(0)
            rand_perm = torch.randperm(N, generator=g).tolist()

            indices = []
            len_slice = N // num_slices
            
            for i in range(num_slices):
                start = i * len_slice
                if i == num_slices - 1:
                    end = N           # 最后一个 client 吃掉所有剩余样本
                else:
                    end = (i + 1) * len_slice
                indices.append(rand_perm[start:end])

            self.indices = indices

            # IID 情况下，list_usr 只起“逻辑分组”的作用，
            # 可以简单照你原来的 FEMNIST 写法用 num_users 去均分：
            num_users = NUM_USERS
            if self.client_selection:
                list_usr = [[i] for i in range(num_users)]
            else:
                nusr = num_users // NUM_CLIENTS
                list_usr = [
                    list(range(nusr * j, nusr * (j + 1) if j != NUM_CLIENTS - 1 else num_users))
                    for j in range(NUM_CLIENTS)
                ]



            
            return models, indices, list_usr

    def init_control(self):
        self.control = ControlModule(self.model, config=config)


    def init_ip_config(self):
        self.ip_train_loader = get_data_loader(EXP_NAME, data_type="train", batch_size=CLIENT_BATCH_SIZE, shuffle=True,
                                               num_workers=train_num, user_list=[0], pin_memory=True)
        self.ip_test_loader = get_data_loader(EXP_NAME, data_type="val", num_workers=test_num, pin_memory=True)
        self.optimizer = torch.optim.SGD(
                        model.parameters(),
                        lr=INIT_LR,
                        momentum =  modementum,
                        weight_decay=weight_decay
                    )
        import torch.optim.lr_scheduler as lr_scheduler
        self.scheduler = lr_scheduler.MultiStepLR(
                    self.optimizer,
                    milestones=[2500, 5000, 7500],
                    gamma=0.5
                )
        self.ip_optimizer_wrapper = OptimizerWrapper(self.model,   self.optimizer,
        self.scheduler,)
        self.ip_control = ControlModule(model=self.model, config=config)

    def save_exp_config(self):
        exp_config = {"exp_name": EXP_NAME, "seed": args.seed, "batch_size": CLIENT_BATCH_SIZE,
                      "num_local_updates": NUM_LOCAL_UPDATES, "mdd": MAX_DEC_DIFF, "init_lr": INIT_LR,
                      "ahl": ADJ_HALF_LIFE, "use_adaptive": self.use_adaptive,
                      "client_selection": args.client_selection}
        if self.client_selection:
            exp_config["num_users"] = num_users
        mkdir_save(exp_config, os.path.join(self.save_path, "exp_config.pt"))


class FEMNISTAdaptiveClient(AdaptiveClient):
    def init_optimizer(self):
        self.optimizer = torch.optim.SGD( self.model.parameters(),
                        lr=INIT_LR,
                        momentum =  modementum,
                        weight_decay=weight_decay
                    )

        self.optimizer_wrapper = OptimizerWrapper(self.model, self.optimizer, clip=True)



    def init_train_loader(self, tl):
        self.train_loader = tl

    def init_test_loader(self, tl):
        self.test_loader = tl
        
def get_indices_list_stackoverflow(user_ids, list_users):
    """
    root: "./datasets/stackoverflow"
    split: "train" or "val"
    list_users: 和 FEMNIST 一样，是一个 list[num_clients]，
                每个元素是 user_id 的列表 (这些 user 属于这个 client)

    返回:
        indices_list: list[num_clients]，每个元素是这个 client 所有样本的 index 列表
    """


    indices_list = []
    for ul in list_users:
        if len(ul) == 0:
            indices_list.append([])
            continue
        ul_tensor = torch.tensor(ul, dtype=torch.long)
        mask = torch.isin(user_ids, ul_tensor)         # [N]
        idx  = torch.nonzero(mask, as_tuple=False).view(-1).tolist()
        indices_list.append(idx)

    return indices_list


def build_list_users_stackoverflow( num_clients, client_selection=False):
    """
    root: "./datasets/stackoverflow"
    split: "train" or "val"
    """

    num_users = NUM_USERS
    
    if client_selection:
        # 和 FEMNIST 一样：每个“候选单元”只含一个 user
        list_users = [[i] for i in range(num_users)]
    else:
        # 把 user 等分给 num_clients 个 client
        nusr = num_users // num_clients
        list_users = []
        for j in range(num_clients):
            if j != num_clients - 1:
                us = list(range(nusr * j, nusr * (j + 1)))
            else:
                us = list(range(nusr * j, num_users))
            list_users.append(us)

    return list_users, num_users



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
    root = "./datasets/stackoverflow"
    split = "train"
    pt_path = os.path.join(root, f"stackoverflow_{split}.pt")
    obj = torch.load(pt_path)
    user_ids = torch.tensor(obj["user_ids"], dtype=torch.long)  

    if not args.use_adaptive:
        set_seed(args.seed)

        args.experiment_name = 'PMT'

        density_list = [0.025, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
        density_list = [0.025, 0.05, 0.1, 0.2, 0.3,0.5,1.0]
        density_list = [0.2, 0.3,0.5,1.0]
        print(args.device=="0")
        
        if args.device == "0":
            density_list = [0.5,1.0]
            density_list = [0.5]
        elif args.device == "1":
            density_list = [0.2,0.3]
            density_list = [0.2]
        else:
            density_list = [0.025, 0.05, 0.1, ]
            density_list = [0.025,]
        density_list.sort(reverse=True)


        # 0.025, 0.05, 0.1, 0.2, 0.4, 0.6, 

        for density in density_list:
            args.experiment_name = 'PMT'+ str(density)
            args.target_density = density
            model = transformer()
            # model.prune_by_pct(1-density)
            # print(model.density())
    
    
            server = FEMNISTAdaptiveServer(args, config, model)
            server.initial_pruning()
            list_models, list_indices, list_users = server.init_clients()
            

            # get_indices_list() indicates what the subscripts are for the data held on each client
            ds = get_data_loader(EXP_NAME, data_type="train", batch_size=CLIENT_BATCH_SIZE, shuffle=False,
                                            num_workers=config.train_num, pin_memory=True).dataset
            
            from bases.vision.data_loader import DataLoader

            client_list = [FEMNISTAdaptiveClient(list_models[idx], config, args.use_adaptive,args) for idx in range(NUM_CLIENTS)]
            sampler = FLSampler(list_indices, MAX_ROUND, NUM_LOCAL_UPDATES * CLIENT_BATCH_SIZE, args.client_selection,
                                NUM_CLIENTS)
            train_loader = get_data_loader(EXP_NAME, data_type="train", batch_size=CLIENT_BATCH_SIZE, shuffle=False,
                                           sampler=sampler, num_workers=0, pin_memory=True)
            print('get_data_loader')
            for client in client_list:
                client.init_optimizer()
                client.init_train_loader(train_loader)


            fl_runner = AdaptiveFL(args, config, server, client_list)
            final_acc,final_std = fl_runner.main()
            print(final_acc, final_std)




