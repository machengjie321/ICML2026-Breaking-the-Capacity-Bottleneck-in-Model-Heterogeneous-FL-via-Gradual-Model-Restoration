import os
import torch
from sympy import print_tree
import sys
from bases.vision.datasets import collate_fn_stackoverflow
from bases.fl.simulation_real.Prune_Recover_FL_fiarse import FedMapServer, FedMapClient,FedMapFL, parse_args
from bases.optim.optimizer import SGD
from bases.optim.optimizer_wrapper import OptimizerWrapper
from bases.vision.load import get_data_loader
from bases.nn.models.bp_transformer import transformer
from bases.vision.sampler_test import FLSampler,get_FL_sampler
from control.sub_algorithm import ControlModule
from configs.StackOverflow import *
import configs.StackOverflow as config
from control.utils import ControlScheduler

from utils.functional import select_best_gpu
from utils.save_load import mkdir_save


class StackOverflowFedMapServer(FedMapServer):

    def init_test_loader(self):
        self.test_loader = get_data_loader(EXP_NAME, data_type="val", num_workers=test_num, pin_memory=True)


    def init_clients(self, sample):
        
        models = [self.model for _ in range(self.number_clients)]

        # -------------------------
        # 1) IID: 样本随机均分
        # -------------------------
        if sample == 'iid':
            N = NUM_TRAIN_DATA
            num_slices = 200 if args.client_selection else  self.args.number_clients

            # 随机打乱 0..N-1
            g = torch.Generator().manual_seed(self.seed)
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
                nusr = num_users // self.number_clients
                list_usr = [
                    list(range(nusr * j, nusr * (j + 1) if j != self.number_clients - 1 else num_users))
                    for j in range(self.number_clients)
                ]

            return models, indices, list_usr

        # -------------------------
        # 2) Non-IID: 按 user 划分
        # -------------------------
        else:
            
            # 建 list_users（每个 client 管哪些 user）
            list_users, _ = build_list_users_stackoverflow(
                self.number_clients, client_selection=self.client_selection
            )
            # 按 user_ids -> 样本 index 划分
            print('first_step')
            indices = get_indices_list_stackoverflow(user_ids, list_users)
            print('second_step')
            self.indices = indices
            
            return models, indices, list_users



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


class StackOverflowFedMapClient(FedMapClient):
    def init_optimizer(self):
        self.optimizer = torch.optim.SGD( self.model.parameters(),
                        lr=INIT_LR,
                        momentum =  modementum,
                        weight_decay=weight_decay
                    )

        self.optimizer_wrapper = OptimizerWrapper(self.model, self.optimizer, clip=self.args.clip)




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

class args:
    def __init__(self, parse_args):
        self.seed = 0
        self.lr_scheduler = False
        ex_lower = parse_args.ex.lower()
        if ex_lower.startswith(('w/o_gmr', 'wo_gmr', 'gmr')):
            self.ex = 'wo_gmr'
        elif ex_lower.startswith(('w/o_asyn', 'wo_asyn', 'asyn')):
            self.ex = 'wo_asyn'
        elif ex_lower.startswith(('pr_fl', 'fedgmr')):
            self.ex = 'FedGMR'
        else:
            self.ex = parse_args.ex
        self.client_selection = False
        self.stal = 'poly'
        self.stal_a = 0.6
        self.patience = parse_args.patience
        self.lr_scheduler_step = False
        self.client_model_norm = False
        self.mu = parse_args.mu
        self.need_client_acc = False
        print(self.ex)
        self.density_gradient =  False
        # n:number,u:update,un:update_number
        self.accumulate = parse_args.accumulate
        

        self.bias_prune = parse_args.bias_prune
        if self.ex.lower().startswith('wo_gmr'):
            self.min_density = 0.02
            self.merge = 'buff_mask_fed_avg'
            self.chronous = 'asyn'
            self.recover = False
            self.Res = True
        elif self.ex.lower().startswith('ims'):
            self.min_density = 0.02
            self.merge = 'buff_mask_fed_avg'
            self.chronous = 'asyn'
            self.recover = True
            self.Res = False
            self.need_client_acc = True
        elif self.ex.lower().startswith('pure2'):
            self.min_density = 0.02
            self.merge = 'mask_fed_avg'
            self.chronous = 'syn'
            self.recover = False
            self.Res = False
            self.need_client_acc = True
        elif self.ex.lower().startswith('pure'):
            self.min_density = 0.02
            self.merge = 'mask_fed_avg'
            self.chronous = 'asyn'
            self.recover = True
            self.Res = False
            self.need_client_acc = True

        elif self.ex.lower().startswith('wo_asyn'):
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

        elif self.ex.lower().startswith('fedgmr'):
            self.min_density = 0.02
            self.merge = 'buff_mask_fed_avg'
            self.chronous = 'asyn'
            self.recover = True
            self.Res = True
            self.need_client_acc = True
        elif self.ex.lower().startswith('mask_fed_avg'):
            self.min_density = 0.02
            self.merge = 'buff_fed_avg'
            self.chronous = 'asyn'
            self.recover = False
            self.Res = True
        elif self.ex.lower().startswith('gradient_avg'):
            self.min_density = 0.02
            self.merge = 'gradient_avg'
            self.chronous = 'asyn'
            self.recover = False
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


        self.experiment_name = 'StackOverflow'
        self.use_coeff = True
        



        self.interval = parse_args.interval
        self.device = select_best_gpu(min_memory= 8 * 1024)
        self.increase = parse_args.increase

        self.resume = parse_args.resume
        self.number_clients = parse_args.num_clients
        self.sample_client = parse_args.sample_client
        self.sample_data_degree = parse_args.sample_data_degree
        cd = [0.5, 0.5, 0.2, 0.1, 0.05]
        self.part = [0.1,0.1,0.2,0.3,0.3]

        if self.sample_client == 'medium':
            part = [0.1,0.1,0.2,0.3,0.3]
        elif self.sample_client == 'high':
            part = [0.1, 0.1, 0.1,0.1,0.6]
        elif self.sample_client == 'low':
            part = [0.2, 0.2, 0.2, 0.2,0.2]
        else:
            assert False
        self.client_density = int(part[0]*self.number_clients)*[1.0] + int(part[1]*self.number_clients)*[0.5] + int(part[2]*self.number_clients)*[0.2]+ int(part[3]*self.number_clients)*[0.1]
        self.client_density = self.client_density + (self.number_clients - len(self.client_density)) * [0.05]
        print(self.client_density)
        import numpy as np
        np.random.lognormal(mean=0, sigma=0.1)
        self.average_download_speed = [(self.get_random_variance() * download_speed) *i for i in self.client_density]
        self.average_upload_speed = [(self.get_random_variance() * upload_speed) *i for i in self.client_density]
        self.server_up_speed = parse_args.server_up_speed

        assert self.stal.lower().startswith('con') or self.stal.lower().startswith('poly') or self.stal.lower().startswith('hinge')
        assert self.stal_a <= 1

        self.sample = 'niid' if parse_args.sample else 'iid'



        self.density = [x for i, x in enumerate(self.client_density) if x not in self.client_density[:i]]

        self.clip = True
        
        # self.experiment_name = (('Ab_small_')+self.sample+('' if self.increase == 2 else '_'+str(self.increase))+('' if self.server_up_speed == 10 else '_'+str(self.server_up_speed)+'_')+'_'+(self.sample_client)+('' if self.sample_data_degree == 0.6 else '_'+str(self.sample_data_degree)) + ('' if self.number_clients == 10 else '_'+str(self.number_clients))+'_'+self.ex+'_'+self.experiment_name + ('' if str(self.density) == str(cd) else '_' + str(self.density)) + \
        #                        str('' if self.accumulate == 'wg' else '_' + self.accumulate)+('' if self.patience == 5 else '_'+str(self.patience))+ ('_clip' if self.clip else '')+ ('_coeff' if self.use_coeff else '') ) 
        self.experiment_name = ('Ab_'+self.sample+('' if self.increase == 2 else '_'+str(self.increase))+('' if self.server_up_speed == 10 else '_'+str(self.server_up_speed)+'_')+'_'+(self.sample_client)+('' if self.sample_data_degree == 0.6 else '_'+str(self.sample_data_degree)) + ('' if self.number_clients == 10 else '_'+str(self.number_clients))+'_'+self.ex+ ('' if str(self.density) == str(cd) else '_' + str(self.density)) + \
                               str('' if self.accumulate == 'wg' else '_' + self.accumulate)+('' if self.patience == 5 else '_'+str(self.patience))+ ('_unclip' if not self.clip else '')) 
        self.experiment_name = ('Ablation_'+self.sample+'_'+(self.sample_client)+'_'+self.ex)
        
    def get_random_variance(self):

        return 1

if __name__ == "__main__":

    args = args(parse_args())
    device = torch.device("cuda:"+str(args.device) if torch.cuda.is_available() else "cpu")
    num_users = NUM_USERS
    seed, resume, use_adaptive = 0, False, True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    import numpy as np
    np.random.seed(args.seed)
    import random
    random.seed(args.seed)

    get_data_loader(EXP_NAME, data_type="train", batch_size=CLIENT_BATCH_SIZE, shuffle=False, num_workers=train_num, pin_memory=True)
    root = "./datasets/stackoverflow"
    split = "train"
    pt_path = os.path.join(root, f"stackoverflow_{split}.pt")
    obj = torch.load(pt_path)
    user_ids = torch.tensor(obj["user_ids"], dtype=torch.long)  


    # if args.bias_prune:
    #     from bases.nn.models.bp_leaf import Conv2
    # else:
    #     from bases.nn.models.leaf import Conv2
        
    model = transformer(bern=True)if args.ex.lower().startswith('fiarse') else transformer()
    
    server = StackOverflowFedMapServer(config, args, model, seed, SGD, {"lr": config.INIT_LR}, use_adaptive, device)
    

    list_models, list_indices, list_users = server.init_clients(args.sample)
    

        
    ds = get_data_loader(EXP_NAME, data_type="train", batch_size=CLIENT_BATCH_SIZE, shuffle=False,
                                    num_workers=config.train_num, pin_memory=True).dataset
    
    from bases.vision.data_loader import DataLoader

    sp = get_FL_sampler(list_indices, MAX_ROUND, NUM_LOCAL_UPDATES * CLIENT_BATCH_SIZE, args.client_selection, args.number_clients)
    
    list_train_loader = []
    for i in range(len(list_indices)):
        sampler = FLSampler(sp[i])
        train_loader = DataLoader(ds, batch_size=CLIENT_BATCH_SIZE, shuffle=False, sampler=sampler, num_workers=config.train_num,
                      pin_memory=True, collate_fn=collate_fn_stackoverflow)  # 每个 worker 预加载 2 个 batch)
        list_train_loader.append(train_loader)


    print("Sampler initialized")

    client_list = [StackOverflowFedMapClient(list_models[idx], config, use_adaptive, exp_config = server.exp_config, args=args, device = device) for idx in range(args.number_clients)]

    i = 0
    for client in client_list:
        client.init_optimizer()
        client.init_train_loader(list_train_loader[i])
        client.init_test_loader(server.test_loader)
        i = i+1


    fl_runner = FedMapFL(args, config, server, client_list)
    statu = fl_runner.main()
    try:
        statu = fl_runner.main()
    except Exception as e:
        print(f"Error occurred: {e}")
        statu = 2  # 或者你自己设定的错误码

    import sys
    sys.exit(statu)()
