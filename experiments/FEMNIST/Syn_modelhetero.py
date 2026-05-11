import os
import torch
from sympy import print_tree
import sys
# sys.path.append(r'D:\PR-FL')
from bases.fl.simulation_real.Prune_Recover_FL_fedrolex import FedMapServer, FedMapClient,FedMapFL, parse_args
from bases.optim.optimizer import SGD
from bases.optim.optimizer_wrapper import OptimizerWrapper
from bases.vision.load import get_data_loader
from bases.nn.models.leaf import Conv2
from bases.vision.sampler_test import FLSampler,get_FL_sampler
from control.sub_algorithm import ControlModule
from configs.femnist import *
import configs.femnist as config
from control.utils import ControlScheduler

from utils.functional import select_best_gpu
from utils.save_load import mkdir_save


class FEMNISTFedMapServer(FedMapServer):
    def get_init_extra_params(self):
        return [([i for i in range(19 * j, 19 * (j + 1) if j != 9 else 193)], self.client_is_sparse) for j in range(self.number_clients)]

    def init_test_loader(self):
        self.test_loader = get_data_loader(EXP_NAME, data_type="test", num_workers=test_num, pin_memory=True)
        self.test_data, self.test_label = self.collate_to_device(self.test_loader)

    def init_clients(self):
        rand_perm = torch.randperm(NUM_TRAIN_DATA).tolist()
        indices = []
        len_slice = NUM_TRAIN_DATA // num_slices

        for i in range(num_slices):
            indices.append(rand_perm[i * len_slice: (i + 1) * len_slice])
            
        
        if self.method.lower().startswith('heterofl') or  self.method.lower().startswith('fjord') or self.method.lower().startswith('fedrolex'):
            list_state_dict, _ = self.split_model()
            models = []
            # print(self.model.state_dict().keys())
            for i in range(len(self.args.client_density)):
                model = Conv2(bern=self.args.bern, model_rate=self.args.client_density[i])
                # print(self.args.client_density[i])
                # print(model.state_dict()['classifier.0.weight'].size())
            
                model.load_state_dict(list_state_dict[i])
                models.append(model)
        else:
            models = [self.model for _ in range(self.args.number_clients)]

        # models = [self.model for _ in range(self.number_clients)]
        self.indices = indices
        if self.client_selection:
            list_usr = [[i] for i in range(num_users)]
        else:
            nusr = num_users // self.number_clients  # num users for the first NUM_CLIENTS - 1 clients
            list_usr = [list(range(nusr * j, nusr * (j + 1) if j != self.number_clients - 1 else num_users)) for j in
                        range(self.number_clients)]

        return models, indices, list_usr



    def init_control(self):
        self.control = ControlModule(self.model, config=config)

    def init_ip_config(self):
        self.ip_train_loader = get_data_loader(EXP_NAME, data_type="train", batch_size=CLIENT_BATCH_SIZE, shuffle=True,
                                               num_workers=train_num, user_list=[0], pin_memory=True)
        self.ip_test_loader = get_data_loader(EXP_NAME, data_type="test", num_workers=test_num, pin_memory=True)
        ip_optimizer = SGD(self.model.parameters(), lr=INIT_LR)
        self.ip_optimizer_wrapper = OptimizerWrapper(self.model, ip_optimizer)
        self.ip_control = ControlModule(model=self.model, config=config)

    def save_exp_config(self):
        exp_config = {"exp_name": EXP_NAME, "seed": args.seed, "batch_size": CLIENT_BATCH_SIZE,
                      "num_local_updates": NUM_LOCAL_UPDATES, "mdd": MAX_DEC_DIFF, "init_lr": INIT_LR,
                      "ahl": ADJ_HALF_LIFE, "use_adaptive": self.use_adaptive,
                      "client_selection": args.client_selection}
        if self.client_selection:
            exp_config["num_users"] = num_users
        mkdir_save(exp_config, os.path.join(self.save_path, "exp_config.pt"))


class FEMNISTFedMapClient(FedMapClient):
    def init_optimizer(self):
        self.optimizer = SGD(self.model.parameters(), lr=INIT_LR)
        

            
        if self.args.lr_scheduler:
            import torch.optim.lr_scheduler as lr_scheduler
            if self.method.lower().startswith('heterofl') or  self.method.lower().startswith('fjord') or self.method.lower().startswith('fedrolex'):
                self.optimizer_wrapper = OptimizerWrapper(self.model, self.optimizer,lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.8, patience=50,
                                                                verbose=True,clip = True))
            else:
                
                self.optimizer_wrapper = OptimizerWrapper(self.model, self.optimizer,lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.8, patience=50,
                                                            verbose=True))
        else:
            if self.method.lower().startswith('heterofl') or  self.method.lower().startswith('fjord') or self.method.lower().startswith('fedrolex'):
                self.optimizer_wrapper = OptimizerWrapper(self.model, self.optimizer,clip = True)
            else:
                self.optimizer_wrapper = OptimizerWrapper(self.model, self.optimizer,)



    def init_train_loader(self, tl):
        self.train_loader = tl

    def init_test_loader(self, tl):
        self.test_loader = tl

def get_indices_list():
    cur_pointer = 0
    indices_list = []
    for ul in list_users:
        num_data = 0
        for user_id in ul:
            train_meta = torch.load(
                os.path.join("datasets", "FEMNIST", "processed", "train_{}.pt".format(user_id)))
            num_data += len(train_meta[0])
        indices_list.append(list(range(cur_pointer, cur_pointer + num_data)))
        cur_pointer += num_data

    return indices_list


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
        self.bern = False
        self.need_client_acc = False


        # n:number,u:update,un:update_number
        self.accumulate = parse_args.accumulate

        self.interval = parse_args.interval
        self.device = select_best_gpu(min_memory=5 * 1024)
        self.increase = parse_args.increase

        self.resume = parse_args.resume
        self.number_clients = parse_args.num_clients
        self.sample_client = parse_args.sample_client
        self.sample_data_degree = parse_args.sample_data_degree
        print(self.ex)


        if self.ex.lower().startswith('fed_avg'):
            self.min_density = 1.0
            self.merge = 'fed_avg'
            self.chronous = 'syn'
            self.recover = True
            self.Res = False
            self.accumulate = 'w'



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
            self.Res = False
            
        elif self.ex.lower().startswith('fedrolex'):
            self.min_density = 0.02
            self.merge = 'heterofl'
            self.chronous = 'syn'
            self.recover = False
            self.Res = False
        elif self.ex.lower().startswith('fjord'):
            self.min_density = 0.02
            self.merge = 'heterofl'
            self.chronous = 'syn'
            self.recover = False
            self.Res = False

        elif self.ex.lower().startswith('pr_fl'):
            self.min_density = 0.02
            self.merge = 'buff_mask_fed_avg'
            self.chronous = 'asyn'
            self.recover = True
            self.Res = True
            self.need_client_acc = True
            
        elif self.ex.lower().startswith('fiarse'):
            self.min_density = 0.02
            self.merge = 'buff_mask_fed_avg'
            self.chronous = 'syn'
            self.recover = False
            self.Res = False
            self.need_client_acc = False
            self.bern = True
            self.accumulate = 'w'
        else:
            assert False
            

        self.experiment_name = "FEMNIST"

        cd = [1.0, 0.5, 0.2, 0.1, 0.05]
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
        import numpy as np
        np.random.lognormal(mean=0, sigma=0.1)
        self.average_download_speed = [(self.get_random_variance() * download_speed) *i for i in self.client_density]
        self.average_upload_speed = [(self.get_random_variance() * upload_speed) *i for i in self.client_density]
        self.server_up_speed = parse_args.server_up_speed

        assert self.stal.lower().startswith('con') or self.stal.lower().startswith('poly') or self.stal.lower().startswith('hinge')
        assert self.stal_a <= 1

        self.sample = 'niid' if parse_args.sample else 'iid'



        self.density = [x for i, x in enumerate(self.client_density) if x not in self.client_density[:i]]


        self.experiment_name = (self.sample+('' if self.server_up_speed == 10 else 'su_'+str(self.server_up_speed)+'_')+'_'+(self.sample_client)+('' if self.sample_data_degree == 0.6 else '_'+str(self.sample_data_degree)) +('' if self.number_clients == 10 else '_'+str(self.number_clients))+'_'+self.ex+'_'+self.experiment_name + ('' if str(self.density) == str(cd) else '_' + str(self.density)) + \
                                str('' if self.accumulate == 'wg' else '_' + self.accumulate)+('' if self.patience == 5 else '_'+str(self.patience)) + ('' if not self.bern else '_bern'))+('NBP')

    def get_random_variance(self):

        return 1

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
    random.seed(args.seed)

    num_users = 200
    num_slices = num_users if args.client_selection else NUM_CLIENTS

    num_user_path = os.path.join("datasets", "FEMNIST", "processed", "num_users.pt")

    if not os.path.isfile(num_user_path):
        get_data_loader(EXP_NAME, data_type="train", batch_size=CLIENT_BATCH_SIZE, shuffle=False, num_workers=train_num,
                        pin_memory=True)
    num_users = torch.load(num_user_path)
    model =  Conv2(bern=True) if args.bern else Conv2()
    # print(model.state_dict().keys())
    # raise 0
    
    # model =  Conv2()


    server = FEMNISTFedMapServer(config, args, model, seed, SGD, {"lr": config.INIT_LR}, use_adaptive, device)

    list_models, list_indices, list_users = server.init_clients()
    list_train_loader = []
    assert args.sample == 'iid' or args.sample == 'niid'
    if args.sample == 'iid':
        pass
    elif args.sample == 'niid':
        list_indices = get_indices_list()
    ds = get_data_loader(EXP_NAME, data_type="train", batch_size=CLIENT_BATCH_SIZE, shuffle=False,
                                    num_workers=config.train_num, pin_memory=True).dataset
    from bases.vision.data_loader import DataLoader

    sp = get_FL_sampler(list_indices, MAX_ROUND, NUM_LOCAL_UPDATES * CLIENT_BATCH_SIZE, args.client_selection, num_slices)
    for i in range(len(list_indices)):
        sampler = FLSampler(sp[i])
        train_loader = DataLoader(ds, batch_size=CLIENT_BATCH_SIZE, shuffle=False, sampler=sampler, num_workers=config.train_num,
                      pin_memory=True)  # 每个 worker 预加载 2 个 batch)
        list_train_loader.append(train_loader)

    print("Sampler initialized")

    client_list = [FEMNISTFedMapClient(list_models[idx], config, use_adaptive, exp_config=server.exp_config, args=args, device=device) for idx in range(args.number_clients)]

    i = 0
    for client in client_list:
        client.init_optimizer()
        client.init_train_loader(list_train_loader[i])
        client.init_test_loader(server.test_loader)
        i = i+1


    fl_runner = FedMapFL(args, config, server, client_list)
    try:
        statu = fl_runner.main()
    except Exception as e:
        print(f"Error occurred: {e}")
        statu = 2  # 或者你自己设定的错误码

    import sys
    sys.exit(statu)
