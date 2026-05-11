import os
import torch
from bases.fl.simulation_real.Prune_Recover_FL import FedMapServer, FedMapClient,FedMapFL, parse_args
from bases.optim.optimizer import SGD
from bases.optim.optimizer_wrapper import OptimizerWrapper
from bases.vision.load import get_data_loader
from bases.nn.models.leaf import Conv2
from bases.vision.sampler import FLSampler
from control.sub_algorithm import ControlModule
from configs.cifar10 import *
import configs.cifar10 as config
from utils.save_load import mkdir_save, load
from os.path import join
from bases.nn.models.vgg import VGG11
from utils.save_load import mkdir_save
import torch
import pynvml

result_path = join("results", "CIFAR10",'split')
os.chdir('/mnt/d/PR-FL/')
def select_best_gpu(min_memory=6 * 1024):  # min_memory 以 MB 为单位，默认 11GB
    pynvml.nvmlInit()
    num_gpus = pynvml.nvmlDeviceGetCount()

    best_gpu = None
    max_free_memory = 0

    for i in range(num_gpus):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        free_memory = mem_info.free // 1024 ** 2  # 转换为 MB

        print(f"GPU {i}: Free memory: {free_memory} MB")

        if free_memory > max_free_memory and free_memory >= min_memory:
            max_free_memory = free_memory
            best_gpu = i

    pynvml.nvmlShutdown()

    if best_gpu is None:
        raise RuntimeError(f"No GPU found with at least {min_memory / 1024} GB free memory!")

    print(f"Selected GPU {best_gpu} with {max_free_memory} MB free memory.")
    return best_gpu

class INFedMapServer(FedMapServer):
    def get_init_extra_params(self):

        return [([i for i in range(19 * j, 19 * (j + 1) if j != 9 else 193)], self.client_is_sparse) for j in range(10)]

    def init_test_loader(self):
        self.test_loader = get_data_loader(EXP_NAME, data_type="test",batch_size=1000, num_workers=config.test_num, pin_memory=True)

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
        self.control = ControlModule(self.model, config=config, args=args)

    def init_ip_config(self):
        self.ip_train_loader = get_data_loader(EXP_NAME, data_type="train", batch_size=CLIENT_BATCH_SIZE,
                                               subset_indices=self.indices[0][:IP_DATA_BATCH * CLIENT_BATCH_SIZE],
                                               shuffle=True, num_workers=config.train_num, pin_memory=True)

        self.ip_test_loader = get_data_loader(EXP_NAME, data_type="test", batch_size=1000, num_workers=config.test_num,
                                              pin_memory=True)

        self.optimizer = SGD(self.model.parameters(), lr=INIT_LR)
        import torch.optim.lr_scheduler as lr_scheduler

        self.optimizer_wrapper = OptimizerWrapper(self.model, self.optimizer)
        self.ip_control = ControlModule(model=self.model, config=config, args=args)

    def save_exp_config(self):
        exp_config = {"exp_name": EXP_NAME, "seed": args.seed, "batch_size": CLIENT_BATCH_SIZE,
                      "num_local_updates": NUM_LOCAL_UPDATES, "mdd": MAX_DEC_DIFF, "init_lr": INIT_LR,
                      "lrhl": LR_HALF_LIFE, "ahl": ADJ_HALF_LIFE, "use_adaptive": self.use_adaptive,
                      "client_selection": args.client_selection}
        if args.client_selection:
            exp_config["num_users"] = num_users
        mkdir_save(exp_config, os.path.join(self.save_path, "exp_config.pt"))



class INFedMapClient(FedMapClient):
    def init_optimizer(self):
        self.optimizer = SGD(self.model.parameters(), lr=INIT_LR)
        if self.args.lr_scheduler:
            import torch.optim.lr_scheduler as lr_scheduler
            self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.8, patience=100
            ,verbose=True)
        else:
            self.scheduler = None

        self.optimizer_wrapper = OptimizerWrapper(self.model, self.optimizer, self.scheduler,)





    def init_train_loader(self, tl):
        self.train_loader = tl

    def init_test_loader(self, tl):
        self.test_loader = tl


def get_indices_list(alpha = 0.5):
    n_clients = NUM_CLIENTS
    dirichlet_alpha = alpha

    n_classes = NUM_CLASSES
    import numpy as np
    from utils.functional import dirichlet_split_noniid
    train_loader = get_data_loader(EXP_NAME, data_type="train", batch_size=CLIENT_BATCH_SIZE, shuffle=False,
                                   num_workers=config.train_num, pin_memory=True)
    labels = np.array(train_loader.dataset.targets)

    # 我们让每个client不同label的样本数量不同，以此做到Non-IID划分
    client_idcs = dirichlet_split_noniid(
        labels, alpha=dirichlet_alpha, n_clients=n_clients)
    return client_idcs


class args:
    def __init__(self, parse_args):
        self.seed = 0
        self.lr_scheduler = False
        self.ex = parse_args.ex
        self.client_selection = False
        self.stal = 'poly'
        self.stal_a = 0.6
        self.patience = parse_args.patience
        self.lr_scheduler_step = False
        self.client_model_norm = False
        self.mu = parse_args.mu
        self.need_client_acc = False


        if self.ex.lower().startswith('fed_avg'):
            self.min_density = 1.0
            self.merge = 'fed_avg'
            self.chronous = 'syn'
            self.recover = True
            self.Res = False

        elif self.ex.lower().startswith('fedprox'):
            self.min_density = 1.0
            self.merge = 'fed_avg'
            self.chronous = 'syn'
            self.recover = True
            self.Res = False
            self.client_model_norm = True

        elif self.ex.lower().startswith('fed_asyn'):
            self.min_density = 1.0
            self.merge = 'fed_asyn'
            self.chronous = 'asyn'
            self.recover = True
            self.Res = False

        elif self.ex.lower().startswith('heterofl'):
            self.min_density = 0.02
            self.merge = 'heterofl'
            self.chronous = 'syn'
            self.recover = False
            self.Res = True

        elif self.ex.lower().startswith('pr_fl'):
            self.min_density = 0.02
            self.merge = 'buff_mask_fed_avg'
            self.chronous = 'asyn'
            self.recover = True
            self.Res = True
            self.need_client_acc = True
        else:
            assert False

        self.experiment_name = 'Cifar10'
        if self.ex.lower().startswith('heterofl'):
            self.experiment_name = 'Cifar10_Re2_'


        # n:number,u:update,un:update_number
        self.accumulate = parse_args.accumulate

        self.interval = parse_args.interval
        self.device = select_best_gpu(min_memory=3 * 1024)
        self.increase = parse_args.increase

        self.resume = parse_args.resume
        self.number_clients = parse_args.num_clients
        self.sample_client = parse_args.sample_client
        self.sample_data_degree = parse_args.sample_data_degree
        cd = [1.0, 0.5, 0.2, 0.1, 0.05]
        self.part = [0.1, 0.1, 0.2, 0.3, 0.3]

        if self.sample_client == 'medium':
            part = [0.1, 0.1, 0.2, 0.3, 0.3]
        elif self.sample_client == 'high':
            part = [0.1, 0.1, 0.1, 0.1, 0.6]
        elif self.sample_client == 'low':
            part = [0.2, 0.2, 0.2, 0.2, 0.2]
        else:
            assert False
        self.client_density = int(part[0] * self.number_clients) * [1.0] + int(part[1] * self.number_clients) * [
            0.5] + int(part[2] * self.number_clients) * [0.2] + int(part[3] * self.number_clients) * [0.1]
        self.client_density = self.client_density + (self.number_clients - len(self.client_density)) * [0.05]
        import numpy as np
        np.random.lognormal(mean=0, sigma=0.1)
        self.average_download_speed = [download_speed * i for i in self.client_density]
        self.average_upload_speed = [upload_speed * i for i in self.client_density]
        self.server_up_speed = parse_args.server_up_speed

        assert self.stal.lower().startswith('con') or self.stal.lower().startswith(
            'poly') or self.stal.lower().startswith('hinge')
        assert self.stal_a <= 1

        self.sample = 'niid' if parse_args.sample else 'iid'

        self.density = [x for i, x in enumerate(self.client_density) if x not in self.client_density[:i]]


        self.experiment_name = (('' if self.server_up_speed == 10 else 'su_' + str(
            self.server_up_speed) + '_') + self.sample + '_' + (self.sample_client + '_') + (
                                    '' if self.sample_data_degree == 0.6 else '_' + str(self.sample_data_degree)) + (
                                            '_' + str(
                                        self.number_clients)) + '_' + self.ex + '_' + self.experiment_name + '_' + str(
            self.patience) + '_' + str(self.density) + \
                                '_' + str('' if self.accumulate == 'wg' else '_' + self.accumulate) + '_' + str(
                    self.interval))



if __name__ == "__main__":



    args = args(parse_args())
    device = torch.device("cuda:"+str(args.device) if torch.cuda.is_available() else "cpu")
    seed, resume, use_adaptive = 0, False, True

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    import numpy as np

    np.random.seed(args.seed)
    import random

    random.seed(0)

    num_users = 100
    num_slices = num_users if args.client_selection else NUM_CLIENTS

    server = INFedMapServer(config, args, VGG11(), seed, SGD, {"lr": config.INIT_LR}, use_adaptive, device)
    list_train_loader = []


    import os

    # 打印当前工作目录
    print("当前工作目录是:", result_path)


    print("Sampler initialized")
    acc_list, loss_list ,density= [],[],[]
    list = os.listdir(result_path)
    print(list)
    for index in list:
        if 'heterofl' in index: continue
        acc_list, loss_list ,density= [],[],[]

        checkpoint = load(os.path.join(
            result_path,index,'split',str(config.save_for_split[-1]), 'checkpoint.pth'))

        for var_name, value in checkpoint.items():
            if var_name.startswith('self.model'):
                server.model = value
            if var_name.startswith('self.control.accumulate_weight_dict'):
                server.control.accumulate_weight_dict = value
        list_state_dict, model_idx, sub_model_time, server.list_mask, _, list_sparse_state_dict, client_density = server.control.sub_adjust_fast(
            client_density=[0.05,0.10,0.20,0.40,0.60,0.80,1.0,1.0,1.0,1.0], use_coff=False,
            min_density=0.02, )


        acc_list,loss_list, density, _, _ = server.test_split(0,server.model)
        print(index)
        print(acc_list)
        print(density)