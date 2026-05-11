import os
import torch
import sys
from bases.nn.models.bp_vgg import VGG11
sys.path.append(r'D:\PR-FL')
from bases.fl.simulation_real.Prune_Recover_FL_fiarse import FedMapServer, FedMapClient,FedMapFL, parse_args
from bases.optim.optimizer import SGD
from bases.optim.optimizer_wrapper import OptimizerWrapper
from bases.vision.load import get_data_loader

from bases.vision.sampler_test import FLSampler,get_FL_sampler
from control.sub_algorithm import ControlModule
import torch.optim as optim
from torch.optim import lr_scheduler

from configs.cifar10 import *
import configs.cifar10 as config
from utils.functional import dirichlet_split_noniid
from utils.save_load import mkdir_save
from utils.functional import select_best_gpu

class INFedMapServer(FedMapServer):

    def init_test_loader(self):
        self.test_loader = get_data_loader(EXP_NAME, data_type="test", num_workers=config.test_num, pin_memory=True)
        # self.test_data, self.test_label = self.collate_to_device(self.test_loader)

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


        ip_optimizer = SGD(self.model.parameters(), lr=INIT_LR)
        from torch.optim import lr_scheduler
        # self.optimizer_scheduler = lr_scheduler.StepLR(ip_optimizer, step_size=1, gamma=0.5 ** (1 / LR_HALF_LIFE))
        self.ip_optimizer_wrapper = OptimizerWrapper(self.model, ip_optimizer)


    def save_exp_config(self):
        exp_config = {"exp_name": EXP_NAME, "seed": args.seed, "batch_size": CLIENT_BATCH_SIZE,
                      "num_local_updates": NUM_LOCAL_UPDATES, "mdd": MAX_DEC_DIFF, "init_lr": INIT_LR,
                      "lrhl": LR_HALF_LIFE, "ahl": ADJ_HALF_LIFE, "use_adaptive": self.use_adaptive,
                      "client_selection": args.client_selection}
        if args.client_selection:
            exp_config["num_users"] = num_users
        mkdir_save(exp_config, os.path.join(self.save_path, "exp_config.pt"))



class INFedMapClient(FedMapClient):
    def init_optimizer(self, wg=0):

        self.optimizer =optim.SGD(self.model.parameters(), lr=INIT_LR,weight_decay=wg)
        if self.args.lr_scheduler:
            import torch.optim.lr_scheduler as lr_scheduler
            self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.8, patience=100
            ,verbose=True)
        else:
            self.scheduler = None

        self.optimizer_wrapper = OptimizerWrapper(self.model, self.optimizer, self.scheduler,)
        # from torch.optim import lr_scheduler
        # self.optimizer_scheduler = lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.5 ** (1 / LR_HALF_LIFE))
        self.optimizer_wrapper = OptimizerWrapper(self.model, self.optimizer, clip=self.args.clip)




    def init_train_loader(self, tl):
        self.train_loader = tl

    def init_test_loader(self, tl):
        self.test_loader = tl

def get_indices_list(dirichlet_alpha, n_clients):

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
        self.client_model_norm = False
        self.bias_prune = parse_args.bias_prune
        self.need_client_acc = False
        if self.ex.lower().startswith('gmr'):
            self.min_density = 0.02
            self.merge = 'buff_mask_fed_avg'
            self.chronous = 'asyn'
            self.recover = False
            self.Res = True
            self.need_client_acc = False
        elif self.ex.lower().startswith('ims'):
            self.min_density = 0.02
            self.merge = 'buff_mask_fed_avg'
            self.chronous = 'asyn'
            self.recover = True
            self.Res = False
            self.need_client_acc = True
        elif self.ex.lower().startswith('asyn'):
            self.min_density = 0.02
            self.merge = 'buff_mask_fed_avg'
            self.chronous = 'syn'
            self.recover = True
            self.Res = True
            self.need_client_acc = True
        elif self.ex.lower().startswith('buff'):
            self.min_density = 0.02
            self.merge = 'mask_fed_avg'
            self.chronous = 'asyn'
            self.recover = True
            self.Res = True
            self.need_client_acc = True
        elif self.ex.lower().startswith('pure2'):
            self.min_density = 0.02
            self.merge = 'mask_fed_avg'
            self.chronous = 'syn'
            self.recover = False
            self.Res = False

            
        elif self.ex.lower().startswith('pure'):
            return 
            self.min_density = 0.02
            self.merge = 'mask_fed_avg'
            self.chronous = 'asyn'
            self.recover = True
            self.Res = False
            self.need_client_acc = True

        elif self.ex.lower().startswith('mask_fed_avg'):
            self.min_density = 0.02
            self.merge = 'buff_fed_avg'
            self.chronous = 'asyn'
            self.recover = True
            self.Res = True
            
        elif self.ex.lower().startswith('pr_fl'):
            self.min_density = 0.02
            self.merge = 'buff_mask_fed_avg'
            self.chronous = 'asyn'
            self.recover = True
            self.Res = True
            self.need_client_acc = True
        elif self.ex.lower().startswith('gradient_avg'):
            self.min_density = 0.02
            self.merge = 'gradient_avg'
            self.chronous = 'asyn'
            self.recover = True
            self.Res = True
            
        elif self.ex.lower().startswith('re_mask_fed_avg'):
            self.min_density = 0.02
            self.merge = 'buff_fed_avg'
            self.chronous = 'asyn'
            self.recover = True
            self.Res = True
            self.need_client_acc = True
        elif self.ex.lower().startswith('re_gradient_avg'):
            self.min_density = 0.02
            self.merge = 'gradient_avg'
            self.chronous = 'asyn'
            self.recover = True
            self.Res = True
            self.need_client_acc = True
        else:
            assert False

        self.experiment_name = 'Cifar10'
  

        self.density_gradient =  False
        # n:number,u:update,un:update_number
        self.accumulate = parse_args.accumulate

        self.interval = parse_args.interval
        self.device = select_best_gpu(min_memory=11 * 1024)
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
        self.average_download_speed = [(self.get_random_variance() * download_speed) * i for i in self.client_density]
        self.average_upload_speed = [(self.get_random_variance() * upload_speed) * i for i in self.client_density]
        self.server_up_speed = parse_args.server_up_speed

        assert self.stal.lower().startswith('con') or self.stal.lower().startswith(
            'poly') or self.stal.lower().startswith('hinge')
        assert self.stal_a <= 1

        self.sample = 'niid' if parse_args.sample else 'iid'

        self.density = [x for i, x in enumerate(self.client_density) if x not in self.client_density[:i]]

        self.clip = parse_args.clip
        self.experiment_name = (self.sample+('' if self.increase == 0.2 else '_'+str(self.increase))+('' if self.server_up_speed == 10 else '_'+str(self.server_up_speed)+'_')+'_'+(self.sample_client)+('' if self.sample_data_degree == 0.6 else '_'+str(self.sample_data_degree)) + ( '_'+str(self.number_clients))+'_'+self.ex+'_'+self.experiment_name + ('' if str(self.density) == str(cd) else '_' + str(self.density)) + \
                               str('' if self.accumulate == 'wg' else '_' + self.accumulate)+('' if self.patience == 5 else '_'+str(self.patience))+ ('_clip' if self.clip else '') ) 

        self.experiment_name = ('Ablation_'+self.sample+('' if self.interval == 50 else '_'+str(self.interval))+('' if self.increase == 2.0 else '_'+str(self.increase))+('' if self.server_up_speed == 10 else '_'+str(self.server_up_speed)+'_')+'_'+(self.sample_client)+'_'+self.ex)

    def get_random_variance(self):
        # import numpy as np
        # n = 20
        # variance = 0
        # for i in range(n):
        #     variance += np.random.lognormal(mean=0, sigma=0.3)
        #
        # variance = variance/n
        return 1


if __name__ == "__main__":
    try:
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
        # if args.bias_prune:
        #     from bases.nn.models.bp_vgg import VGG11
        # else:
        #     from bases.nn.models.vgg import VGG11
        # 
        server = INFedMapServer(config, args, VGG11(), seed, SGD, {"lr": config.INIT_LR}, use_adaptive, device)
        list_models, list_indices = server.init_clients()
        ds = get_data_loader(EXP_NAME, data_type="train", batch_size=CLIENT_BATCH_SIZE, shuffle=False,
                                        num_workers=config.train_num, pin_memory=True).dataset
        from bases.vision.data_loader import DataLoader
        list_train_loader = []
        assert args.sample == 'iid' or args.sample == 'niid'
        if args.sample == 'iid':
            pass
        elif args.sample == 'niid':
            list_indices = get_indices_list(args.sample_data_degree, args.number_clients)

        sp = get_FL_sampler(list_indices, MAX_ROUND, NUM_LOCAL_UPDATES * CLIENT_BATCH_SIZE, args.client_selection,
                            num_slices)
        for i in range(len(list_indices)):
            sampler = FLSampler(sp[i])
            train_loader = DataLoader(ds, batch_size=CLIENT_BATCH_SIZE, shuffle=False, sampler=sampler, num_workers=config.train_num,
                        pin_memory=True)  # 每个 worker 预加载 2 个 batch)
            list_train_loader.append(train_loader)


        print("Sampler initialized")

        client_list = [INFedMapClient(list_models[idx], config,use_adaptive,  exp_config=server.exp_config, args=args, device=device) for idx in range(args.number_clients)]


        for i in range(len(client_list)):
            client_list[i].init_optimizer()
            client_list[i].init_train_loader(list_train_loader[i])
            client_list[i].init_test_loader(server.test_loader)

        fl_runner = FedMapFL(args, config, server, client_list)
    
        statu = fl_runner.main()
    except Exception as e:
        print(f"Error occurred: {e}")
        statu = 2  # 或者你自己设定的错误码

    import sys
    sys.exit(statu)
