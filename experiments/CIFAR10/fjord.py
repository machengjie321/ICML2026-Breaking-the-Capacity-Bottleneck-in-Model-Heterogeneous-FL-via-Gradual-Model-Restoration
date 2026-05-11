import random

import numpy as np
import torch
import torch.nn.functional as F

import configs.cifar10 as config
from bases.fl.simulation_real.fjord import FjordClient, FjordServer, parse_args
from bases.nn.odleaf import VGG11
from bases.optim.optimizer import SGD
from bases.optim.optimizer_wrapper import OptimizerWrapper
from bases.vision.load import get_data_loader
from bases.vision.sampler_test import FLSampler, get_FL_sampler
from configs.cifar10 import *
from control.sub_algorithm import ControlModule
from experiments.fjord_timed_common import TimedFjordFL, TimedFjordServerMixin
from utils.save_load import mkdir_save
from utils.functional import select_best_gpu


class CIFAR10FedMapServer(TimedFjordServerMixin, FjordServer):
    def __init__(self, config, args, model, save_interval=50):
        super().__init__(config, args, model, save_interval=save_interval)
        self.setup_timed_server(args)

    def init_test_loader(self):
        self.test_loader = get_data_loader(
            EXP_NAME,
            data_type="test",
            num_workers=config.test_num,
            batch_size=1000,
            pin_memory=True,
        )

    def init_clients(self):
        rand_perm = torch.randperm(NUM_TRAIN_DATA).tolist()
        indices = []
        len_slice = NUM_TRAIN_DATA // num_slices
        for i in range(num_slices):
            indices.append(rand_perm[i * len_slice: (i + 1) * len_slice])

        models = [self.model for _ in range(self.args.number_clients)]
        self.indices = indices
        return models, indices

    def init_control(self):
        self.control = ControlModule(self.model, config=config)

    def init_ip_config(self):
        ip_optimizer = SGD(self.model.parameters(), lr=INIT_LR)
        self.ip_optimizer_wrapper = OptimizerWrapper(self.model, ip_optimizer)

    def save_exp_config(self):
        exp_config = {
            "exp_name": EXP_NAME,
            "seed": args.seed,
            "batch_size": CLIENT_BATCH_SIZE,
            "num_local_updates": NUM_LOCAL_UPDATES,
            "mdd": MAX_DEC_DIFF,
            "init_lr": INIT_LR,
            "lrhl": LR_HALF_LIFE,
            "ahl": ADJ_HALF_LIFE,
            "use_adaptive": self.use_adaptive,
            "client_selection": args.client_selection,
        }
        if args.client_selection:
            exp_config["num_users"] = num_users
        mkdir_save(exp_config, f"{self.save_path}/exp_config.pt")


class CIFAR10FedMapClient(FjordClient):
    def __init__(self, model, config, args, model_rate):
        super().__init__(model, config, args, model_rate)
        self.test_loader = None

    def init_optimizer(self):
        self.optimizer = SGD(self.model.parameters(), lr=INIT_LR, weight_decay=0)
        self.optimizer_wrapper = OptimizerWrapper(self.model, self.optimizer)

    def init_train_loader(self, tl):
        self.train_loader = tl

    def init_test_loader(self, tl):
        self.test_loader = tl

    def load_rate(self, rate):
        self.model_rate = rate

    @torch.no_grad()
    def evaluate(self, test_loader, rate=None, mode="sum"):
        if rate is None:
            rate = self.model_rate
        self.model.eval()
        test_loss = 0
        n_correct = 0
        n_total = 0
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = self.model(inputs, p=rate)
            batch_loss = F.cross_entropy(outputs, labels)
            test_loss += batch_loss.item()
            labels_predicted = torch.argmax(outputs, dim=1)
            if labels.dim() == 2:
                labels = torch.argmax(labels, dim=1)
            n_total += labels.size(0)
            n_correct += torch.sum(torch.eq(labels_predicted, labels)).item()
        if mode == "mean" and n_total > 0:
            test_loss /= n_total
        self.model.train()
        return test_loss, n_correct / n_total if n_total > 0 else 0

    def main(self):
        model_rate = self.model_rate
        self.model.train()
        num_proc_data = 0
        density = sorted(set([0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00, model_rate]))
        idx = density.index(model_rate)
        for i in range(self.config.NUM_LOCAL_UPDATES):
            rate = density[i % (idx + 1)]
            inputs, labels = self.train_loader.get_next_batch()
            self.optimizer_wrapper.step3(inputs.to(self.device), labels.to(self.device), rate)
            num_proc_data += len(inputs)
        return self.model.state_dict(), num_proc_data, model_rate


def get_indices_list(dirichlet_alpha, n_clients):
    train_loader = get_data_loader(
        EXP_NAME,
        data_type="train",
        batch_size=CLIENT_BATCH_SIZE,
        shuffle=False,
        num_workers=config.train_num,
        pin_memory=True,
    )
    labels = np.array(train_loader.dataset.targets)
    from utils.functional import dirichlet_split_noniid

    return dirichlet_split_noniid(labels, alpha=dirichlet_alpha, n_clients=n_clients)


class args:
    def __init__(self, parse_args):
        self.seed = 0
        self.lr_scheduler = False
        self.ex = parse_args.ex
        self.client_selection = False
        self.stal = "poly"
        self.stal_a = 0.6
        self.patience = parse_args.patience
        self.wait_stable_mult = max(1, int(parse_args.wait_stable_mult))
        self.lr_scheduler_step = False
        self.client_model_norm = False
        self.bern = parse_args.bern
        self.need_client_acc = False
        self.accumulate = parse_args.accumulate
        self.interval = parse_args.interval
        self.device = select_best_gpu(min_memory=3 * 1024)
        self.increase = parse_args.increase
        self.resume = parse_args.resume
        self.number_clients = parse_args.num_clients
        self.sample_client = parse_args.sample_client
        self.sample_data_degree = parse_args.sample_data_degree
        self.server_up_speed = parse_args.server_up_speed

        ex_lower = self.ex.lower()
        if ex_lower.startswith("gmr_fjord"):
            self.min_density = 0.02
            self.merge = "heterofl"
            self.chronous = "asyn"
            self.recover = True
            self.Res = True
            self.need_client_acc = True
        elif ex_lower.startswith("abgmr_fjord"):
            self.min_density = 0.02
            self.merge = "heterofl"
            self.chronous = "asyn"
            self.recover = False
            self.Res = True
        elif ex_lower.startswith("ims_fjord"):
            self.min_density = 0.02
            self.merge = "heterofl"
            self.chronous = "asyn"
            self.recover = True
            self.Res = False
            self.need_client_acc = True
        elif ex_lower.startswith("asyn_fjord"):
            self.min_density = 0.02
            self.merge = "heterofl"
            self.chronous = "syn"
            self.recover = True
            self.Res = True
            self.need_client_acc = True
        elif ex_lower.startswith("fjord"):
            self.min_density = 0.02
            self.merge = "heterofl"
            self.chronous = "syn"
            self.recover = False
            self.Res = False
        else:
            raise AssertionError(f"Unsupported experiment: {self.ex}")

        self.experiment_name = "Cifar10"
        cd = [1.0, 0.5, 0.2, 0.1, 0.05]

        if self.sample_client == "medium":
            self.part = [0.1, 0.1, 0.2, 0.3, 0.3]
        elif self.sample_client == "high":
            self.part = [0.1, 0.1, 0.1, 0.1, 0.6]
        elif self.sample_client == "low":
            self.part = [0.2, 0.2, 0.2, 0.2, 0.2]
        else:
            raise AssertionError(f"Unsupported sample_client: {self.sample_client}")

        self.client_density = (
            int(self.part[0] * self.number_clients) * [1.0]
            + int(self.part[1] * self.number_clients) * [0.5]
            + int(self.part[2] * self.number_clients) * [0.2]
            + int(self.part[3] * self.number_clients) * [0.1]
        )
        self.client_density += (self.number_clients - len(self.client_density)) * [0.05]
        np.random.lognormal(mean=0, sigma=0.1)
        self.average_download_speed = [download_speed * i for i in self.client_density]
        self.average_upload_speed = [upload_speed * i for i in self.client_density]

        assert self.stal.lower().startswith("con") or self.stal.lower().startswith("poly") or self.stal.lower().startswith("hinge")
        assert self.stal_a <= 1

        self.sample = "niid" if parse_args.sample else "iid"
        self.density = [x for i, x in enumerate(self.client_density) if x not in self.client_density[:i]]
        self.experiment_name = (
            self.sample
            + ("" if self.server_up_speed == 10 else "su_" + str(self.server_up_speed) + "_")
            + "_"
            + self.sample_client
            + ("" if self.sample_data_degree == 0.6 else "_" + str(self.sample_data_degree))
            + ("" if self.number_clients == 10 else "_" + str(self.number_clients))
            + "_"
            + self.ex
            + "_"
            + self.experiment_name
            + ("" if str(self.density) == str(cd) else "_" + str(self.density))
            + ("" if self.accumulate == "wg" else "_" + self.accumulate)
            + ("" if self.patience == 5 else "_" + str(self.patience))
            + ("" if self.wait_stable_mult == 1 else "_wsm_" + str(self.wait_stable_mult))
            + ("" if not self.bern else "_bern")
        )


if __name__ == "__main__":
    args = args(parse_args())
    device = torch.device("cuda:" + str(args.device) if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    num_users = 100
    num_slices = num_users if args.client_selection else NUM_CLIENTS
    model = VGG11()
    server = CIFAR10FedMapServer(config, args, model)
    list_models, list_indices = server.init_clients()

    ds = get_data_loader(
        EXP_NAME,
        data_type="train",
        batch_size=CLIENT_BATCH_SIZE,
        shuffle=False,
        num_workers=config.train_num,
        pin_memory=True,
    ).dataset
    from bases.vision.data_loader import DataLoader

    if args.sample == "niid":
        list_indices = get_indices_list(args.sample_data_degree, args.number_clients)

    sp = get_FL_sampler(
        list_indices,
        MAX_ROUND,
        NUM_LOCAL_UPDATES * CLIENT_BATCH_SIZE,
        args.client_selection,
        num_slices,
    )
    list_train_loader = []
    for i in range(len(list_indices)):
        sampler = FLSampler(sp[i])
        train_loader = DataLoader(
            ds,
            batch_size=CLIENT_BATCH_SIZE,
            shuffle=False,
            sampler=sampler,
            num_workers=config.train_num,
            pin_memory=True,
        )
        list_train_loader.append(train_loader)

    print("Sampler initialized")

    client_list = [
        CIFAR10FedMapClient(list_models[idx], config, args=args, model_rate=args.client_density[idx])
        for idx in range(args.number_clients)
    ]
    for i, client in enumerate(client_list):
        client.init_optimizer()
        client.init_train_loader(list_train_loader[i])
        client.init_test_loader(server.test_loader)

    fl_runner = TimedFjordFL(args, config, server, client_list)
    try:
        status = fl_runner.main()
    except Exception as exc:
        print(f"Error occurred: {exc}")
        status = 2

    import sys

    sys.exit(status)
