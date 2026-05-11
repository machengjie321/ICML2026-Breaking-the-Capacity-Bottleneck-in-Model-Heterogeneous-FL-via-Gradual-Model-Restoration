import os
import random

import numpy as np
import torch
import torch.nn.functional as F

import configs.StackOverflow as config
from bases.fl.simulation_real.fjord import FjordClient, FjordServer, parse_args
from bases.nn.models.od_transformer import transformer
from bases.optim.optimizer_wrapper import OptimizerWrapper
from bases.vision.datasets import collate_fn_stackoverflow
from bases.vision.load import get_data_loader
from bases.vision.sampler_test import FLSampler, get_FL_sampler
from configs.StackOverflow import *
from control.sub_algorithm import ControlModule
from experiments.fjord_timed_common import TimedFjordFL, TimedFjordServerMixin
from utils.functional import select_best_gpu
from utils.save_load import mkdir_save


class StackOverflowFedMapServer(TimedFjordServerMixin, FjordServer):
    def __init__(self, config, args, model, save_interval=50):
        super().__init__(config, args, model, save_interval=save_interval)
        self.setup_timed_server(args)

    def init_test_loader(self):
        self.test_loader = get_data_loader(EXP_NAME, data_type="val", num_workers=test_num, pin_memory=True)

    def init_clients(self, sample):
        models = [self.model for _ in range(self.args.number_clients)]

        if sample == "iid":
            n_train = NUM_TRAIN_DATA
            num_slices = 200 if args.client_selection else NUM_CLIENTS
            generator = torch.Generator().manual_seed(self.seed)
            rand_perm = torch.randperm(n_train, generator=generator).tolist()
            indices = []
            len_slice = n_train // num_slices
            for i in range(num_slices):
                start = i * len_slice
                end = n_train if i == num_slices - 1 else (i + 1) * len_slice
                indices.append(rand_perm[start:end])
            self.indices = indices

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

        list_users, _ = build_list_users_stackoverflow(self.number_clients, client_selection=self.client_selection)
        indices = get_indices_list_stackoverflow(user_ids, list_users)
        self.indices = indices
        return models, indices, list_users

    def init_control(self):
        self.control = ControlModule(self.model, config=config)

    def init_ip_config(self):
        self.ip_train_loader = get_data_loader(
            EXP_NAME,
            data_type="train",
            batch_size=CLIENT_BATCH_SIZE,
            shuffle=True,
            num_workers=train_num,
            user_list=[0],
            pin_memory=True,
        )
        self.ip_test_loader = get_data_loader(EXP_NAME, data_type="val", num_workers=test_num, pin_memory=True)
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=INIT_LR,
            momentum=modementum,
            weight_decay=weight_decay,
        )
        import torch.optim.lr_scheduler as lr_scheduler

        self.scheduler = lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=[2500, 5000, 7500],
            gamma=0.5,
        )
        self.ip_optimizer_wrapper = OptimizerWrapper(self.model, self.optimizer, self.scheduler)
        self.ip_control = ControlModule(model=self.model, config=config)

    def save_exp_config(self):
        exp_config = {
            "exp_name": EXP_NAME,
            "seed": args.seed,
            "batch_size": CLIENT_BATCH_SIZE,
            "num_local_updates": NUM_LOCAL_UPDATES,
            "mdd": MAX_DEC_DIFF,
            "init_lr": INIT_LR,
            "ahl": ADJ_HALF_LIFE,
            "use_adaptive": self.use_adaptive,
            "client_selection": args.client_selection,
        }
        if args.client_selection:
            exp_config["num_users"] = num_users
        mkdir_save(exp_config, f"{self.save_path}/exp_config.pt")


class StackOverflowFedMapClient(FjordClient):
    def __init__(self, model, config, args, model_rate):
        super().__init__(model, config, args, model_rate)
        self.test_loader = None

    def init_optimizer(self):
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=INIT_LR,
            momentum=modementum,
            weight_decay=weight_decay,
        )
        self.optimizer_wrapper = OptimizerWrapper(self.model, self.optimizer, clip=self.args.clip)

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
        assert mode in ["sum", "mean"]
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        total_correct_top5 = 0
        pad_id = 0
        for batch in test_loader:
            labels = batch.to(self.device)
            logits = self.model(labels, p=rate)
            target = labels[:, 1:]
            _, vocab_size, _ = logits.shape
            loss = F.cross_entropy(logits, target)
            logits_flat = logits.permute(0, 2, 1).reshape(-1, vocab_size)
            target_flat = target.reshape(-1)
            mask = target_flat != pad_id
            logits_flat = logits_flat[mask]
            target_flat = target_flat[mask]
            n_tokens = target_flat.numel()
            if n_tokens == 0:
                continue
            top5 = logits_flat.topk(5, dim=1).indices
            correct_top5 = (top5 == target_flat.unsqueeze(1)).any(dim=1).sum().item()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens
            total_correct_top5 += correct_top5
        self.model.train()
        if total_tokens == 0:
            return 0.0, 0.0
        avg_loss = total_loss / total_tokens
        if mode == "sum":
            return avg_loss, total_correct_top5 / total_tokens
        return avg_loss, total_correct_top5 / total_tokens

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


def get_indices_list_stackoverflow(user_ids, list_users):
    indices_list = []
    for ul in list_users:
        if len(ul) == 0:
            indices_list.append([])
            continue
        ul_tensor = torch.tensor(ul, dtype=torch.long)
        mask = torch.isin(user_ids, ul_tensor)
        idx = torch.nonzero(mask, as_tuple=False).view(-1).tolist()
        indices_list.append(idx)
    return indices_list


def build_list_users_stackoverflow(num_clients, client_selection=False):
    num_users = NUM_USERS
    if client_selection:
        list_users = [[i] for i in range(num_users)]
    else:
        nusr = num_users // num_clients
        list_users = []
        for j in range(num_clients):
            if j != num_clients - 1:
                users = list(range(nusr * j, nusr * (j + 1)))
            else:
                users = list(range(nusr * j, num_users))
            list_users.append(users)
    return list_users, num_users


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
        self.need_client_acc = False
        self.accumulate = parse_args.accumulate
        self.interval = parse_args.interval
        self.device = select_best_gpu(min_memory=4 * 1024)
        self.increase = parse_args.increase
        self.resume = parse_args.resume
        self.number_clients = parse_args.num_clients
        self.sample_client = parse_args.sample_client
        self.sample_data_degree = parse_args.sample_data_degree
        self.server_up_speed = parse_args.server_up_speed
        self.clip = True
        self.use_coeff = True

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

        self.experiment_name = "StackOverflow"
        cd = [0.5, 0.5, 0.2, 0.1, 0.05]
        self.part = [0.1, 0.1, 0.2, 0.3, 0.3]

        if self.sample_client == "medium":
            part = [0.1, 0.1, 0.2, 0.3, 0.3]
        elif self.sample_client == "high":
            part = [0.1, 0.1, 0.1, 0.1, 0.6]
        elif self.sample_client == "low":
            part = [0.2, 0.2, 0.2, 0.2, 0.2]
        else:
            raise AssertionError(f"Unsupported sample_client: {self.sample_client}")
        self.part = part

        self.client_density = (
            int(part[0] * self.number_clients) * [1.0]
            + int(part[1] * self.number_clients) * [0.5]
            + int(part[2] * self.number_clients) * [0.2]
            + int(part[3] * self.number_clients) * [0.1]
        )
        self.client_density += (self.number_clients - len(self.client_density)) * [0.05]
        np.random.lognormal(mean=0, sigma=0.1)
        self.average_download_speed = [(self.get_random_variance() * download_speed) * i for i in self.client_density]
        self.average_upload_speed = [(self.get_random_variance() * upload_speed) * i for i in self.client_density]

        assert self.stal.lower().startswith("con") or self.stal.lower().startswith("poly") or self.stal.lower().startswith("hinge")
        assert self.stal_a <= 1

        self.sample = "niid" if parse_args.sample else "iid"
        self.density = [x for i, x in enumerate(self.client_density) if x not in self.client_density[:i]]
        self.experiment_name = (
            self.sample
            + ("" if self.increase == 0.2 else "_" + str(self.increase))
            + ("" if self.server_up_speed == 10 else "_" + str(self.server_up_speed) + "_")
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
            + ("_clip" if self.clip else "")
            + ("_coeff" if self.use_coeff else "")
        )

    def get_random_variance(self):
        return 1


if __name__ == "__main__":
    args = args(parse_args())
    device = torch.device("cuda:" + str(args.device) if torch.cuda.is_available() else "cpu")
    num_users = NUM_USERS

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    get_data_loader(
        EXP_NAME,
        data_type="train",
        batch_size=CLIENT_BATCH_SIZE,
        shuffle=False,
        num_workers=train_num,
        pin_memory=True,
    )
    root = "./datasets/stackoverflow"
    pt_path = os.path.join(root, "stackoverflow_train.pt")
    obj = torch.load(pt_path)
    user_ids = torch.tensor(obj["user_ids"], dtype=torch.long)

    model = transformer()
    server = StackOverflowFedMapServer(config, args, model)
    list_models, list_indices, list_users = server.init_clients(args.sample)

    ds = get_data_loader(
        EXP_NAME,
        data_type="train",
        batch_size=CLIENT_BATCH_SIZE,
        shuffle=False,
        num_workers=config.train_num,
        pin_memory=True,
    ).dataset
    from bases.vision.data_loader import DataLoader

    sp = get_FL_sampler(
        list_indices,
        MAX_ROUND,
        NUM_LOCAL_UPDATES * CLIENT_BATCH_SIZE,
        args.client_selection,
        NUM_CLIENTS,
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
            collate_fn=collate_fn_stackoverflow,
        )
        list_train_loader.append(train_loader)

    print("Sampler initialized")

    client_list = [
        StackOverflowFedMapClient(list_models[idx], config, args=args, model_rate=args.client_density[idx])
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
