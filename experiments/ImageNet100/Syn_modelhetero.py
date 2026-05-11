import os
import random

import numpy as np
import torch
import torch.nn as nn

import bases.fl.simulation_real.Prune_Recover_FL_fedrolex as hetero_backend
import configs.imagenet100 as config
from bases.nn.models.bp_resnet import resnet18 as BiasPruneResNet18
from bases.nn.models.resnet import resnet18 as DenseResNet18
from bases.optim.optimizer_wrapper import OptimizerWrapper
from bases.vision.load import get_data_loader
from bases.vision.sampler_test import FLSampler, get_FL_sampler
from configs.imagenet100 import *
from control.sub_algorithm import ControlModule
from utils.functional import select_best_gpu
from utils.save_load import mkdir_save

parse_args = hetero_backend.parse_args


def make_group_norm(channels):
    return nn.GroupNorm(4, channels)


def resolve_imagenet100_init_lr(exp_args):
    experiment_lower = exp_args.ex.lower()
    if (
        experiment_lower.startswith("heterofl")
        or experiment_lower.startswith("fedrolex")
        or experiment_lower.startswith("gmr_heterofl")
        or experiment_lower.startswith("gmr_fedrolex")
        or experiment_lower.startswith("abgmr_heterofl")
        or experiment_lower.startswith("abgmr_fedrolex")
    ):
        return 0.05
    return INIT_LR


def apply_runtime_init_lr(lr):
    global INIT_LR
    INIT_LR = lr
    config.INIT_LR = lr


def model_shapes_match(model, state_dict):
    current_state_dict = model.state_dict()
    for key, value in state_dict.items():
        if key not in current_state_dict:
            continue
        if tuple(current_state_dict[key].shape) != tuple(value.shape):
            return False
    return True


def infer_layer_planes_from_state_dict(state_dict):
    return [
        state_dict["conv1.weight"].shape[0],
        state_dict["layer2.0.conv1.weight"].shape[0],
        state_dict["layer3.0.conv1.weight"].shape[0],
        state_dict["layer4.0.conv1.weight"].shape[0],
    ]


def build_model(exp_args, model_rate=1.0, layer_planes=None):
    model_class = BiasPruneResNet18 if exp_args.bias_prune else DenseResNet18
    return model_class(
        num_classes=NUM_CLASSES,
        model_rate=model_rate,
        layer_planes=layer_planes,
        norm_layer=exp_args.norm,
    )


class ImageNet100FedMapServer(hetero_backend.FedMapServer):
    def init_test_loader(self):
        self.test_loader = get_data_loader(
            EXP_NAME,
            data_type="val",
            batch_size=200,
            num_workers=config.test_num,
            pin_memory=True,
        )

    def init_clients(self):
        rand_perm = torch.randperm(NUM_TRAIN_DATA).tolist()
        indices = []
        len_slice = NUM_TRAIN_DATA // num_slices

        for i in range(num_slices):
            indices.append(rand_perm[i * len_slice: (i + 1) * len_slice])

        list_state_dict, _ = self.split_model()
        models = []
        for client_idx, density in enumerate(self.args.client_density):
            model = build_model(self.args, model_rate=density)
            model.load_state_dict(list_state_dict[client_idx])
            models.append(model)

        self.indices = indices
        return models, indices

    def init_control(self):
        self.control = ControlModule(self.model, config=config)

    def init_ip_config(self):
        ip_optimizer = torch.optim.SGD(self.model.parameters(), lr=INIT_LR)
        import torch.optim.lr_scheduler as lr_scheduler

        self.optimizer_scheduler = lr_scheduler.StepLR(
            ip_optimizer,
            step_size=STEP_SIZE,
            gamma=0.5 ** (STEP_SIZE / LR_HALF_LIFE),
        )
        self.ip_optimizer_wrapper = OptimizerWrapper(self.model, ip_optimizer, self.optimizer_scheduler)

    def save_exp_config(self):
        exp_config = {
            "exp_name": EXP_NAME,
            "seed": args.seed,
            "batch_size": CLIENT_BATCH_SIZE,
            "num_local_updates": NUM_LOCAL_UPDATES,
            "mdd": MAX_DEC_DIFF,
            "init_lr": INIT_LR,
            "momentum": MOMENTUM,
            "weight_decay": WEIGHT_DECAY,
            "lrhl": LR_HALF_LIFE,
            "step_size": STEP_SIZE,
            "ahl": ADJ_HALF_LIFE,
            "use_adaptive": self.use_adaptive,
            "client_selection": args.client_selection,
        }
        if args.client_selection:
            exp_config["num_users"] = num_users
        mkdir_save(exp_config, os.path.join(self.save_path, "exp_config.pt"))


class ImageNet100FedMapClient(hetero_backend.FedMapClient):
    def init_optimizer(self):
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=INIT_LR,
            momentum=MOMENTUM,
            weight_decay=WEIGHT_DECAY,
        )
        self.optimizer_wrapper = OptimizerWrapper(self.model, self.optimizer, clip=True)

    def rebuild_model_from_state_dict(self, state_dict):
        previous_lr = None
        if self.optimizer_wrapper is not None and self.optimizer_wrapper.optimizer is not None:
            previous_lr = self.optimizer_wrapper.optimizer.param_groups[0]["lr"]

        layer_planes = infer_layer_planes_from_state_dict(state_dict)
        self.model = build_model(self.args, layer_planes=layer_planes)
        self.model.train()
        self.model.to(self.device)
        self.list_mask = [None for _ in range(len(self.model.prunable_layers))]
        self.mask_dict = {}
        self.client_is_sparse = False
        self.is_sparse = False

        self.init_optimizer()
        if previous_lr is not None:
            self.load_lr(previous_lr)

    def init_train_loader(self, tl):
        self.train_loader = tl

    def load_state_dict(self, idx_state_dict):
        _, state_dict, _ = hetero_backend.unpack_client_state_payload(idx_state_dict)
        if not model_shapes_match(self.model, state_dict):
            self.rebuild_model_from_state_dict(state_dict)
        super().load_state_dict(idx_state_dict)


def get_indices_list(dirichlet_alpha, n_clients, ds):
    labels = np.array(ds.targets)
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
        self.lr_scheduler_step = False
        self.client_model_norm = False
        self.need_client_acc = False
        self.bias_prune = parse_args.bias_prune
        self.accumulate = parse_args.accumulate
        self.interval = parse_args.interval
        self.device = select_best_gpu(min_memory=8 * 1024)
        self.increase = parse_args.increase
        self.resume = parse_args.resume
        self.number_clients = parse_args.num_clients
        self.sample_client = parse_args.sample_client
        self.sample_data_degree = parse_args.sample_data_degree
        self.norm = make_group_norm
        self.bern = False

        experiment_lower = self.ex.lower()
        if (
            experiment_lower.startswith("gmr_heterofl")
            or experiment_lower.startswith("gmr_fedrolex")
        ):
            self.min_density = 0.02
            self.merge = "heterofl"
            self.chronous = "asyn"
            self.recover = True
            self.Res = True
            self.need_client_acc = True
        elif (
            experiment_lower.startswith("abgmr_heterofl")
            or experiment_lower.startswith("abgmr_fedrolex")
        ):
            self.min_density = 0.02
            self.merge = "heterofl"
            self.chronous = "asyn"
            self.recover = False
            self.Res = True
        elif experiment_lower.startswith("heterofl") or experiment_lower.startswith("fedrolex"):
            self.min_density = 0.02
            self.merge = "heterofl"
            self.chronous = "syn"
            self.recover = False
            self.Res = False
        else:
            raise AssertionError(f"Unsupported experiment: {self.ex}")

        self.experiment_name = "ImageNet100"
        cd = [1.0, 0.5, 0.2, 0.1, 0.05]

        if self.sample_client == "medium":
            part = [0.1, 0.1, 0.2, 0.3, 0.3]
        elif self.sample_client == "high":
            part = [0.1, 0.1, 0.1, 0.1, 0.6]
        elif self.sample_client == "low":
            part = [0.2, 0.2, 0.2, 0.2, 0.2]
        else:
            raise AssertionError(f"Unsupported sample_client: {self.sample_client}")
        self.part = part

        self.client_density = int(part[0] * self.number_clients) * [1.0]
        self.client_density += int(part[1] * self.number_clients) * [0.5]
        self.client_density += int(part[2] * self.number_clients) * [0.2]
        self.client_density += int(part[3] * self.number_clients) * [0.1]
        self.client_density += (self.number_clients - len(self.client_density)) * [0.05]

        np.random.lognormal(mean=0, sigma=0.1)
        self.average_download_speed = [download_speed * i for i in self.client_density]
        self.average_upload_speed = [upload_speed * i for i in self.client_density]
        self.server_up_speed = parse_args.server_up_speed

        assert self.stal.lower().startswith("con") or self.stal.lower().startswith("poly") or self.stal.lower().startswith("hinge")
        assert self.stal_a <= 1

        self.sample = "niid" if parse_args.sample else "iid"
        self.init_lr = resolve_imagenet100_init_lr(self)
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
            + str("" if self.accumulate == "wg" else "_" + self.accumulate)
            + ("" if self.patience == 5 else "_" + str(self.patience))
        )


if __name__ == "__main__":
    args = args(parse_args())
    apply_runtime_init_lr(args.init_lr)
    device = torch.device("cuda:" + str(args.device) if torch.cuda.is_available() else "cpu")
    seed, resume, use_adaptive = 0, False, True

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    num_users = 200
    num_slices = num_users if args.client_selection else args.number_clients
    model = build_model(args)
    server = ImageNet100FedMapServer(
        config,
        args,
        model,
        seed,
        torch.optim.SGD,
        {"lr": config.INIT_LR},
        use_adaptive,
        device,
    )
    list_models, list_indices = server.init_clients()

    ds = get_data_loader(
        EXP_NAME,
        data_type="train",
        batch_size=CLIENT_BATCH_SIZE,
        shuffle=False,
        num_workers=config.train_num,
        pin_memory=True,
    ).dataset

    if args.sample == "niid":
        list_indices = get_indices_list(args.sample_data_degree, args.number_clients, ds)

    from bases.vision.data_loader import DataLoader

    sp = get_FL_sampler(
        list_indices,
        MAX_ROUND,
        NUM_LOCAL_UPDATES * CLIENT_BATCH_SIZE,
        args.client_selection,
        num_slices,
    )

    print("Sampler initialized")
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

    client_list = [
        ImageNet100FedMapClient(
            list_models[idx],
            config,
            use_adaptive,
            exp_config=server.exp_config,
            args=args,
            device=device,
        )
        for idx in range(args.number_clients)
    ]

    for i, client in enumerate(client_list):
        client.init_optimizer()
        client.init_train_loader(list_train_loader[i])

    fl_runner = hetero_backend.FedMapFL(args, config, server, client_list)
    try:
        status = fl_runner.main()
    except Exception as exc:
        print(f"Error occurred: {exc}")
        status = 2

    import sys

    sys.exit(status)
