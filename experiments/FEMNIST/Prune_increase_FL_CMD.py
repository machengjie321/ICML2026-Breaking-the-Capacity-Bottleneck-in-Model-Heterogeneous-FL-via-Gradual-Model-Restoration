import os
import random
import traceback

import numpy as np
import torch

import bases.fl.simulation_real.Prune_Recover_FL_fiarse as fiarse_backend
import bases.fl.simulation_real.Prune_Recover_FL_fedrolex as hetero_backend
import configs.femnist as config
from bases.nn.models.bp_leaf import Conv2 as BiasPruneConv2
from bases.nn.models.leaf import Conv2 as DenseConv2
from bases.optim.optimizer import SGD
from bases.optim.optimizer_wrapper import OptimizerWrapper
from bases.vision.load import get_data_loader
from bases.vision.sampler_test import FLSampler, get_FL_sampler
from configs.femnist import *
from control.sub_algorithm import ControlModule
from utils.functional import select_best_gpu
from utils.save_load import mkdir_save

parse_args = fiarse_backend.parse_args


def normalize_experiment_name(experiment_name):
    normalized_name = experiment_name.strip()
    experiment_lower = normalized_name.lower()
    if experiment_lower.startswith("hetero") and not experiment_lower.startswith("heterofl"):
        return "heterofl" + normalized_name[len("hetero"):]
    return normalized_name


def is_hetero_mode(experiment_name):
    experiment_lower = experiment_name.lower()
    return (
        experiment_lower.startswith("gmr_heterofl")
        or experiment_lower.startswith("gmr_fedrolex")
        or experiment_lower.startswith("gmr_fjord")
        or experiment_lower.startswith("abgmr_heterofl")
        or experiment_lower.startswith("abgmr_fedrolex")
        or experiment_lower.startswith("abgmr_fjord")
        or experiment_lower.startswith("ims_heterofl")
        or experiment_lower.startswith("ims_fedrolex")
        or experiment_lower.startswith("ims_fjord")
        or experiment_lower.startswith("asyn_heterofl")
        or experiment_lower.startswith("asyn_fedrolex")
        or experiment_lower.startswith("asyn_fjord")
        or experiment_lower.startswith("heterofl")
        or experiment_lower.startswith("hetero")
        or experiment_lower.startswith("fedrolex")
        or experiment_lower.startswith("fjord")
    )


def should_force_clip(experiment_name):
    experiment_lower = experiment_name.lower()
    return (
        experiment_lower.startswith("gmr_heterofl")
        or experiment_lower.startswith("gmr_fedrolex")
        or experiment_lower.startswith("gmr_fjord")
        or experiment_lower.startswith("abgmr_heterofl")
        or experiment_lower.startswith("abgmr_fedrolex")
        or experiment_lower.startswith("abgmr_fjord")
        or experiment_lower.startswith("ims_heterofl")
        or experiment_lower.startswith("ims_fedrolex")
        or experiment_lower.startswith("ims_fjord")
        or experiment_lower.startswith("asyn_heterofl")
        or experiment_lower.startswith("asyn_fedrolex")
        or experiment_lower.startswith("asyn_fjord")
        or experiment_lower.startswith("heterofl")
        or experiment_lower.startswith("hetero")
        or experiment_lower.startswith("fedrolex")
        or experiment_lower.startswith("fjord")
    )


def infer_hidden_sizes_from_state_dict(state_dict):
    return [
        state_dict["features.0.weight"].shape[0],
        state_dict["features.3.weight"].shape[0],
        state_dict["classifier.0.weight"].shape[0],
    ]


def model_shapes_match(model, state_dict):
    current_state_dict = model.state_dict()
    for key, value in state_dict.items():
        if key not in current_state_dict:
            continue
        if tuple(current_state_dict[key].shape) != tuple(value.shape):
            return False
    return True


def build_model(exp_args, model_rate=1.0, hidden_sizes=None):
    model_class = BiasPruneConv2 if exp_args.bias_prune else DenseConv2
    bern = exp_args.ex.lower().startswith("fiarse")
    if is_hetero_mode(exp_args.ex):
        return model_class(bern=bern, model_rate=model_rate, hidden_sizes=hidden_sizes)
    return model_class(bern=bern)


def select_backend(experiment_name):
    return hetero_backend if is_hetero_mode(experiment_name) else fiarse_backend


def make_experiment_classes(server_base, client_base):
    class FEMNISTFedMapServer(server_base):
        def __init__(self, config, args, *init_args, **init_kwargs):
            # Compatibility:
            # In some backends (e.g. fedrolex), init_control() is called before base
            # assigns self.args. We need self.args during init_control() for
            # local_topk/overlap options.
            self.args = args
            super().__init__(config, args, *init_args, **init_kwargs)

        def init_test_loader(self):
            self.test_loader = get_data_loader(EXP_NAME, data_type="test", num_workers=test_num, pin_memory=True)
            self.test_data, self.test_label = self.collate_to_device(self.test_loader)

        def init_clients(self):
            rand_perm = torch.randperm(NUM_TRAIN_DATA).tolist()
            indices = []
            len_slice = NUM_TRAIN_DATA // num_slices

            for i in range(num_slices):
                indices.append(rand_perm[i * len_slice: (i + 1) * len_slice])

            if is_hetero_mode(self.args.ex):
                list_state_dict, _ = self.split_model()
                models = []
                for client_idx, density in enumerate(self.args.client_density):
                    model = build_model(self.args, model_rate=density)
                    model.load_state_dict(list_state_dict[client_idx])
                    models.append(model)
            else:
                models = [self.model for _ in range(self.number_clients)]

            self.indices = indices
            if self.client_selection:
                list_usr = [[i] for i in range(num_users)]
            else:
                nusr = num_users // self.number_clients
                list_usr = [
                    list(range(nusr * j, nusr * (j + 1) if j != self.number_clients - 1 else num_users))
                    for j in range(self.number_clients)
                ]

            return models, indices, list_usr

        def init_control(self):
            self.control = ControlModule(self.model, config=config, args=self.args)

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
            self.ip_test_loader = get_data_loader(EXP_NAME, data_type="test", num_workers=test_num, pin_memory=True)
            ip_optimizer = SGD(self.model.parameters(), lr=INIT_LR)
            self.ip_optimizer_wrapper = OptimizerWrapper(self.model, ip_optimizer)
            self.ip_control = ControlModule(model=self.model, config=config, args=self.args)

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
            if self.client_selection:
                exp_config["num_users"] = num_users
            mkdir_save(exp_config, os.path.join(self.save_path, "exp_config.pt"))

    class FEMNISTFedMapClient(client_base):
        def init_optimizer(self):
            self.optimizer = SGD(self.model.parameters(), lr=INIT_LR)
            clip = self.args.clip or should_force_clip(self.args.ex)

            if self.args.lr_scheduler:
                import torch.optim.lr_scheduler as lr_scheduler

                scheduler = lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    mode="min",
                    factor=0.8,
                    patience=50,
                    verbose=True,
                )
                self.optimizer_wrapper = OptimizerWrapper(self.model, self.optimizer, scheduler, clip=clip)
            else:
                self.optimizer_wrapper = OptimizerWrapper(self.model, self.optimizer, clip=clip)

        def rebuild_model_from_state_dict(self, state_dict):
            previous_lr = None
            if self.optimizer_wrapper is not None and self.optimizer_wrapper.optimizer is not None:
                previous_lr = self.optimizer_wrapper.optimizer.param_groups[0]["lr"]

            hidden_sizes = infer_hidden_sizes_from_state_dict(state_dict)
            self.model = build_model(self.args, hidden_sizes=hidden_sizes)
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

        def init_test_loader(self, tl):
            self.test_loader = tl

        def load_state_dict(self, idx_state_dict):
            if len(idx_state_dict) == 3:
                if hasattr(idx_state_dict[1], "keys"):
                    _, state_dict, _ = idx_state_dict
                else:
                    _, _, state_dict = idx_state_dict
            elif len(idx_state_dict) == 2:
                _, state_dict = idx_state_dict
            else:
                raise ValueError(f"Unexpected state dict payload length: {len(idx_state_dict)}")
            if is_hetero_mode(self.args.ex) and not model_shapes_match(self.model, state_dict):
                self.rebuild_model_from_state_dict(state_dict)
            super().load_state_dict(idx_state_dict)

    return FEMNISTFedMapServer, FEMNISTFedMapClient


def get_indices_list():
    cur_pointer = 0
    indices_list = []
    for ul in list_users:
        num_data = 0
        for user_id in ul:
            train_meta = torch.load(
                os.path.join("datasets", "FEMNIST", "processed", "train_{}.pt".format(user_id))
            )
            num_data += len(train_meta[0])
        indices_list.append(list(range(cur_pointer, cur_pointer + num_data)))
        cur_pointer += num_data

    return indices_list


class ExperimentArgs:
    def __init__(self, parsed_args):
        self.seed = 0
        self.lr_scheduler = False
        self.original_ex = parsed_args.ex
        self.ex = normalize_experiment_name(parsed_args.ex)
        self.client_selection = False
        self.stal = "poly"
        self.stal_a = 0.6
        self.patience = parsed_args.patience
        self.lr_scheduler_step = False
        self.client_model_norm = False
        self.mu = parsed_args.mu
        self.need_client_acc = False
        self.accumulate = parsed_args.accumulate
        self.bias_prune = parsed_args.bias_prune
        self.use_coeff = True
        self.density_gradient = False
        self.interval = parsed_args.interval
        self.device = select_best_gpu(min_memory=4 * 1024)
        self.increase = parsed_args.increase
        self.recover_step_mode = (
            parsed_args.recover_step_mode.lower() if hasattr(parsed_args, "recover_step_mode") else "legacy"
        )
        if self.recover_step_mode not in {"legacy", "ladder", "fixed"}:
            self.recover_step_mode = "legacy"
        self.recover_step = parsed_args.recover_step if hasattr(parsed_args, "recover_step") else 0.1
        self.recover_trigger_mode = (
            parsed_args.recover_trigger_mode.lower() if hasattr(parsed_args, "recover_trigger_mode") else "early"
        )
        if self.recover_trigger_mode not in {"early", "time", "mixed"}:
            self.recover_trigger_mode = "early"
        self.recover_time_total = (
            float(parsed_args.recover_time_total) if hasattr(parsed_args, "recover_time_total") else -1.0
        )
        self.recover_time_points = (
            str(parsed_args.recover_time_points)
            if hasattr(parsed_args, "recover_time_points")
            else "0.2,0.4,0.6,0.8,1.0"
        )
        self.recover_time_ladder = (
            str(parsed_args.recover_time_ladder)
            if hasattr(parsed_args, "recover_time_ladder")
            else "0.05,0.1,0.2,0.5,1.0"
        )
        self.prune_warmup_rounds = (
            max(0, int(parsed_args.prune_warmup_rounds))
            if hasattr(parsed_args, "prune_warmup_rounds")
            else 0
        )
        self.local_topk_mode = (
            parsed_args.local_topk_mode.lower() if hasattr(parsed_args, "local_topk_mode") else "global"
        )
        if self.local_topk_mode not in {"global", "client_replace"}:
            self.local_topk_mode = "global"
        self.measure_topk_overlap = (
            bool(parsed_args.measure_topk_overlap) if hasattr(parsed_args, "measure_topk_overlap") else False
        )
        self.measure_min_sub_diff = (
            bool(parsed_args.measure_min_sub_diff) if hasattr(parsed_args, "measure_min_sub_diff") else False
        )
        self.overlap_topk_ratio = (
            float(parsed_args.overlap_topk_ratio) if hasattr(parsed_args, "overlap_topk_ratio") else 0.1
        )
        self.resume = parsed_args.resume
        self.number_clients = parsed_args.num_clients
        self.sample_client = parsed_args.sample_client
        self.sample_data_degree = parsed_args.sample_data_degree
        self.server_up_speed = parsed_args.server_up_speed
        self.clip = parsed_args.clip
        self.sample = "niid" if parsed_args.sample else "iid"

        print(self.original_ex)

        experiment_lower = self.ex.lower()
        if experiment_lower.startswith("fed_avg"):
            self.min_density = 0.5
            self.merge = "fed_avg"
            self.chronous = "syn"
            self.recover = True
            self.Res = False
        elif experiment_lower.startswith("fed_asyn"):
            self.min_density = 1.0
            self.merge = "fed_asyn"
            self.chronous = "asyn"
            self.recover = True
            self.Res = False
        elif experiment_lower.startswith("fiarse"):
            self.min_density = 0.02
            self.merge = "buff_mask_fed_avg"
            self.chronous = "syn"
            self.recover = False
            self.Res = False
            self.accumulate = "w"
            self.use_coeff = False
        elif experiment_lower.startswith("gmr_fiarse"):
            self.min_density = 0.02
            self.merge = "buff_mask_fed_avg"
            self.chronous = "asyn"
            self.recover = True
            self.Res = True
            self.accumulate = "w"
            self.use_coeff = True
            self.need_client_acc = True
        elif experiment_lower.startswith("abgmr_fiarse"):
            self.min_density = 0.02
            self.merge = "buff_mask_fed_avg"
            self.chronous = "asyn"
            self.recover = False
            self.Res = True
            self.accumulate = "w"
            self.use_coeff = True

        elif experiment_lower.startswith(("pr_fl", "fedgmr")):
            self.ex = "FedGMR"
            self.min_density = 0.02
            self.merge = "buff_mask_fed_avg"
            self.chronous = "asyn"
            self.recover = True
            self.Res = True
            self.need_client_acc = True

        elif (
            experiment_lower.startswith("gmr_heterofl")
            or experiment_lower.startswith("gmr_fedrolex")
            or experiment_lower.startswith("gmr_fjord")
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
            or experiment_lower.startswith("abgmr_fjord")
        ):
            self.min_density = 0.02
            self.merge = "heterofl"
            self.chronous = "asyn"
            self.recover = False
            self.Res = True

        elif (
            experiment_lower.startswith("asyn_heterofl")
            or experiment_lower.startswith("asyn_fedrolex")
            or experiment_lower.startswith("asyn_fjord")
        ):
            self.min_density = 0.02
            self.merge = "heterofl"
            self.chronous = "syn"
            self.recover = True
            self.Res = True
            self.need_client_acc = True
        elif (
            experiment_lower.startswith("heterofl")
            or experiment_lower.startswith("fedrolex")
            or experiment_lower.startswith("fjord")
        ):
            self.min_density = 0.02
            self.merge = "heterofl"
            self.chronous = "syn"
            self.recover = False
            self.Res = False
        else:
            raise AssertionError(f"Unsupported experiment: {self.original_ex}")

        # client_replace uses a different pruning path and is incompatible with IMS.
        self.disable_ims_for_local_replace = self.local_topk_mode == "client_replace"
        if self.disable_ims_for_local_replace and self.Res:
            self.Res = False

        self.experiment_name = "FEMNIST"

        default_density_tag = [0.5, 0.5, 0.2, 0.1, 0.05]
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
            int(self.part[0] * self.number_clients) * [1.0]
            + int(self.part[1] * self.number_clients) * [0.5]
            + int(self.part[2] * self.number_clients) * [0.2]
            + int(self.part[3] * self.number_clients) * [0.1]
        )
        self.client_density = self.client_density + (self.number_clients - len(self.client_density)) * [0.05]

        np.random.lognormal(mean=0, sigma=0.1)
        self.average_download_speed = [
            (self.get_random_variance() * download_speed) * density for density in self.client_density
        ]
        self.average_upload_speed = [
            (self.get_random_variance() * upload_speed) * density for density in self.client_density
        ]

        assert (
            self.stal.lower().startswith("con")
            or self.stal.lower().startswith("poly")
            or self.stal.lower().startswith("hinge")
        )
        assert self.stal_a <= 1

        self.density = [x for i, x in enumerate(self.client_density) if x not in self.client_density[:i]]
        self.experiment_name = (
            self.sample
            + ("" if self.increase == 2 else "_" + str(self.increase))
            + ("" if self.server_up_speed == 10 else "_" + str(self.server_up_speed) + "_")
            + "_"
            + self.sample_client
            + ("" if self.sample_data_degree == 0.6 else "_" + str(self.sample_data_degree))
            + ("" if self.number_clients == 10 else "_" + str(self.number_clients))
            + "_"
            + self.original_ex
            + ("" if str(self.density) == str(default_density_tag) else "_" + str(self.density))
            + str("" if self.accumulate == "wg" else "_" + self.accumulate)
            + ("" if self.patience == 5 else "_" + str(self.patience))
        )
        if self.recover_step_mode != "legacy":
            self.experiment_name += "_rsm_" + self.recover_step_mode
        if self.recover_step_mode == "fixed":
            self.experiment_name += "_rs_" + str(self.recover_step)
        if self.local_topk_mode != "global":
            self.experiment_name += "_ltm_" + self.local_topk_mode
        if self.disable_ims_for_local_replace:
            self.experiment_name += "_noims"
        if self.measure_topk_overlap:
            self.experiment_name += "_ovr_" + str(self.overlap_topk_ratio)
        if self.measure_min_sub_diff:
            self.experiment_name += "_msd"
        if self.recover_trigger_mode != "early":
            self.experiment_name += "_rtm_" + self.recover_trigger_mode
            if self.recover_time_total > 0:
                self.experiment_name += "_rtt_" + str(int(self.recover_time_total))
            if self.recover_time_points != "0.2,0.4,0.6,0.8,1.0":
                self.experiment_name += "_rtp_" + self.recover_time_points.replace(",", "-")
            if self.recover_time_ladder != "0.05,0.1,0.2,0.5,1.0":
                self.experiment_name += "_rtl_" + self.recover_time_ladder.replace(",", "-")
        if self.prune_warmup_rounds > 0:
            self.experiment_name += "_pw_" + str(self.prune_warmup_rounds)

    def get_random_variance(self):
        return 1


if __name__ == "__main__":
    parsed_args = parse_args()
    args = ExperimentArgs(parsed_args)
    backend = select_backend(args.ex)
    FEMNISTFedMapServer, FEMNISTFedMapClient = make_experiment_classes(
        backend.FedMapServer,
        backend.FedMapClient,
    )
    FedMapFL = backend.FedMapFL

    device = torch.device("cuda:" + str(args.device) if torch.cuda.is_available() else "cpu")
    seed, resume, use_adaptive = 0, False, True

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    num_users = 200
    num_slices = num_users if args.client_selection else NUM_CLIENTS

    num_user_path = os.path.join("datasets", "FEMNIST", "processed", "num_users.pt")
    if not os.path.isfile(num_user_path):
        get_data_loader(
            EXP_NAME,
            data_type="train",
            batch_size=CLIENT_BATCH_SIZE,
            shuffle=False,
            num_workers=train_num,
            pin_memory=True,
        )
    num_users = torch.load(num_user_path)

    if args.ex.lower().startswith("fedgmr") or parse_args.ex.lower().startswith("pr_fl"):
        # Legacy compatibility for FEMNIST PR-FL:
        # old script instantiated Conv2 twice before server init
        # (one throwaway model, then the server model), which changes RNG stream.
        _ = build_model(args)
    model = build_model(args)
    server = FEMNISTFedMapServer(
        config,
        args,
        model,
        seed,
        SGD,
        {"lr": config.INIT_LR},
        use_adaptive,
        device=device,
    )

    list_models, list_indices, list_users = server.init_clients()
    list_train_loader = []
    assert args.sample in {"iid", "niid"}
    if args.sample == "niid":
        list_indices = get_indices_list()

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
        num_slices,
    )
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
        FEMNISTFedMapClient(
            list_models[idx],
            config,
            use_adaptive,
            exp_config=server.exp_config,
            args=args,
            device=device,
        )
        for idx in range(args.number_clients)
    ]

    for idx, client in enumerate(client_list):
        client.init_optimizer()
        client.init_train_loader(list_train_loader[idx])
        client.init_test_loader(server.test_loader)

    fl_runner = FedMapFL(args, config, server, client_list)
    try:
        status = fl_runner.main()
    except Exception as e:
        traceback.print_exc()
        print(f"Error occurred: {e}")
        status = 2

    raise SystemExit(status)
