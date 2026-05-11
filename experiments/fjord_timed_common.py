import copy
import os
from collections import OrderedDict, deque

import numpy as np
import torch


class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0, client_num=10):
        self.num = 1 + client_num
        self.patience = [patience] * self.num
        self.verbose = verbose
        self.counter = [0] * self.num
        self.best_score = [None] * self.num
        self.average_score = [None] * self.num
        self.early_stop = [False] * self.num
        self.val_loss_min = np.inf
        self.delta = delta
        self.state = [None] * self.num
        self.last_score = [None] * self.num

    def __call__(self, acc, logger):
        score = acc
        self.state = [None] * self.num
        for i in range(self.num):
            if self.best_score[i] is None:
                self.best_score[i] = score[i]
                self.state[i] = True
                self.last_score[i] = score[i]
                continue
            if i == 0 and self.last_score[i] == score[i]:
                self.state[i] = True
                continue
            if score[i] <= self.best_score[i] + self.delta:
                self.last_score[i] = score[i]
                self.counter[i] += 1
                if self.counter[i] >= self.patience[i]:
                    if i == 0:
                        logger.info(
                            f"server out of patience, best score is {self.best_score[i]}, current score is {score[i]}"
                        )
                    else:
                        logger.info(
                            f"client {i} out of patience, best score is {self.best_score[i]}, current score is {score[i]}"
                        )
                    self.early_stop[i] = True
                    self.state[i] = False
                    continue
            else:
                self.last_score[i] = score[i]
                self.best_score[i] = score[i]
                self.counter[i] = 0
                self.state[i] = True
        return self.state


class TimedFjordServerMixin:
    def setup_timed_server(self, args):
        self.need_client_acc = args.need_client_acc
        self.chronous = args.chronous
        self.Res = args.Res
        self.patience = args.patience
        self.wait_stable_mult = max(1, int(getattr(args, "wait_stable_mult", 1)))
        self.min_density = args.min_density
        self.increse = float(args.increase)
        self.interval_signal = False
        self.average_round_time = None
        self.train_number = None
        self.current_time = 0
        self.time = []
        self.early_stop = False
        self.begin_save = False
        self.wait_for_stable = 0
        self.client_wait_for_stable = [0 for _ in range(len(self.client_density))]
        self.restore_density = sorted(set(self.client_density))
        self.early_stoping = EarlyStopping(
            patience=args.patience,
            client_num=len(self.client_density) if self.need_client_acc else 0,
        )
        self.list_client_loss = [[] for _ in range(len(self.client_density))]
        self.list_client_acc = [[] for _ in range(len(self.client_density))]
        self.list_client_density = [[] for _ in range(len(self.client_density))]
        self.list_train_number = [[] for _ in range(len(self.client_density))]
        self.fed_avg_loss = []
        self.fed_avg_acc = []
        self.list_loss = [0]
        self.list_acc = [0]
        self.sum_server_upload = [0]
        self.sum_server_download = [[0] for _ in range(len(self.client_density))]
        self.list_state_dict = self.build_client_state_dicts()

    def get_client_wait_for_stable_steps(self):
        # Client cooldown also scales with wait_stable_mult (e.g., 5/7).
        client_mult = max(2, self.wait_stable_mult)
        return max(1, int(self.patience * client_mult))

    def get_server_wait_for_stable_steps(self):
        return max(1, int(self.patience * self.wait_stable_mult))

    def tick_client_wait_for_stable(self):
        for i in range(len(self.client_wait_for_stable)):
            if self.client_wait_for_stable[i] > 0:
                self.client_wait_for_stable[i] -= 1

    def get_real_size(self, list_state_dict, exp, density):
        list_model_size = []
        for i in range(len(list_state_dict)):
            if list_state_dict[i] is None or density[i] == 0:
                list_model_size.append(0)
                continue
            list_model_size.append(self.config.model_size * density[i])
        return list_model_size

    def mask_state_dict(self, state_dict, mask):
        new_state = OrderedDict()
        for key, value in state_dict.items():
            if "weight" in key or "bias" in key:
                if "weight" in key:
                    if value.dim() == 4:
                        out_idx, in_idx = mask[key]
                        mask_tensor = torch.zeros_like(value)
                        mask_tensor[
                            out_idx[:, None, None, None],
                            in_idx[None, :, None, None],
                            :,
                            :,
                        ] = 1
                        new_state[key] = value * mask_tensor
                    elif value.dim() == 2:
                        out_idx, in_idx = mask[key]
                        mask_tensor = torch.zeros_like(value)
                        mask_tensor[out_idx[:, None], in_idx[None, :]] = 1
                        new_state[key] = value * mask_tensor
                    elif value.dim() == 1:
                        idx = mask[key]
                        mask_tensor = torch.zeros_like(value)
                        mask_tensor[idx] = 1
                        new_state[key] = value * mask_tensor
                    else:
                        raise NotImplementedError(f"{key} has unexpected dim {value.dim()}")
                else:
                    idx = mask[key]
                    mask_tensor = torch.zeros_like(value)
                    mask_tensor[idx] = 1
                    new_state[key] = value * mask_tensor
            else:
                new_state[key] = copy.deepcopy(value)
        return new_state

    def build_client_state_dicts(self, base_state_dict=None):
        if base_state_dict is None:
            base_state_dict = self.model.state_dict()
        return [self.mask_state_dict(base_state_dict, mask) for mask in self.list_mask]

    def evaluate_with_state_dict(self, state_dict):
        current_state = copy.deepcopy(self.model.state_dict())
        self.model.load_state_dict(state_dict)
        loss, acc = self.model.evaluate(self.test_loader)
        self.model.load_state_dict(current_state)
        return loss, acc

    def test_client_model(self, list_sd):
        if not self.need_client_acc:
            return
        test_client_interval = self.interval
        for i in range(len(self.client_density)):
            if self.train_number[i] % test_client_interval != 1:
                continue
            if list_sd[i] is not None:
                client_loss, client_acc = self.client_list[i].evaluate(self.test_loader, self.client_density[i])
            elif self.list_client_loss[i]:
                client_loss = self.list_client_loss[i][-1]
                client_acc = self.list_client_acc[i][-1]
            else:
                client_loss, client_acc = 0, 0
            self.list_client_acc[i].append(client_acc)
            self.list_client_loss[i].append(client_loss)
            self.list_client_density[i].append(self.client_density[i])
            self.list_train_number[i].append(self.train_number[i])

    def next_density(self, current_density):
        restore_density = sorted(set(self.restore_density + [self.min_density, current_density]))
        next_density_candidates = [d for d in restore_density if d > current_density + 0.01]
        if next_density_candidates:
            return min(next_density_candidates)
        if self.increse >= 1.0:
            return 1.0
        return min(1.0, current_density + min(current_density, 0.20))

    def save_display_data(self):
        self.save_metrics()

    def GMR(self):
        if not self.fed_avg_acc:
            return

        if self.early_stoping.num == 1:
            acc = [self.fed_avg_acc[-1]]
        else:
            if any(len(client_acc) == 0 for client_acc in self.list_client_acc):
                return
            acc = [self.fed_avg_acc[-1]] + [client_acc[-1] for client_acc in self.list_client_acc]
            self.logger.info(acc)
            print(acc)

        if (not self.recover) or (self.min_density == 1):
            if self.early_stoping.patience[0] != min(self.patience * 5, 50):
                self.logger.info("Enter the end stage, increase the patience to obtain the best acc")
                self.early_stoping.patience[0] = min(self.patience * 5, 50)

        state = self.early_stoping(acc, self.logger)
        server_recovered_this_round = False

        if not self.begin_save:
            if self.early_stoping.counter[0] >= self.patience:
                self.begin_save = True
                self.early_stoping.counter[0] = 0
                state[0] = None

        if state[0] is True and self.begin_save:
            self.save_display_data()
        elif state[0] is False and self.begin_save:
            self.save_display_data()
            if self.recover:
                if self.min_density >= 1.0:
                    self.early_stop = True
                else:
                    self.min_density = self.next_density(self.min_density)
                    server_recovered_this_round = True
                    for j in range(len(self.early_stoping.counter)):
                        self.early_stoping.best_score[j] = None
                        self.early_stoping.early_stop[j] = False
            else:
                self.early_stop = True

        if self.recover and self.begin_save and len(self.early_stoping.counter) > 1:
            for i in range(1, self.early_stoping.num):
                client_id = i - 1
                if self.client_wait_for_stable[client_id] > 0:
                    continue
                if state[i] is False:
                    self.logger.info(f"client{client_id} is out of patience")
                    self.client_density[client_id] = self.next_density(self.client_density[client_id])
                    self.client_wait_for_stable[client_id] = self.get_client_wait_for_stable_steps()
                    self.logger.info(
                        f"for client{client_id}, increase model client, client density {self.client_density[client_id]}"
                    )
                    print(
                        f"for client{client_id}, increase model client, client density {self.client_density[client_id]}"
                    )
                    self.early_stoping.best_score[0] = None
                    self.early_stoping.early_stop[0] = False
                    self.early_stoping.best_score[i] = None
                    self.early_stoping.early_stop[i] = False
                    self.save_display_data()

        for i in range(len(self.client_density)):
            if self.client_density[i] < self.min_density:
                self.client_density[i] = self.min_density
                if len(self.early_stoping.counter) > 1:
                    self.early_stoping.counter[0] = 0
                    self.early_stoping.best_score[0] = -1
                    self.early_stoping.early_stop[0] = False
                    self.early_stoping.counter[i + 1] = 0
                    self.early_stoping.best_score[i + 1] = -1
                    self.early_stoping.early_stop[i + 1] = False

        if server_recovered_this_round:
            self.wait_for_stable = max(self.wait_for_stable, self.get_server_wait_for_stable_steps())
            self.logger.info(f"enter wait_for_stable={self.wait_for_stable} after recovery")

    def Print_FL_Message(self, idx, time_download):
        print(f"Round #{idx} (Experiment = {self.experiment_name}).")
        print(f"Elapsed time = {self.time[-1] if self.time else 0}")
        print(
            f"fed_avg Loss/acc (at round #{idx}) = {self.fed_avg_loss[-1]}/{self.fed_avg_acc[-1]}   "
            f"Loss/acc={self.list_loss[-1]}/{self.list_acc[-1]}"
        )
        print("the density is " + ",".join([f"{client_density:.2f}" for client_density in self.client_density]))
        print("self.train_number :" + str(self.train_number))
        print("sum_server_upload :" + str(self.sum_server_upload[-1]) + "_" + str(time_download))
        self.logger.info(f"Round #{idx} (Experiment = {self.experiment_name}).")
        self.logger.info(f"Elapsed time = {self.time[-1] if self.time else 0}")
        self.logger.info(
            f"fed_avg Loss/acc (at round #{idx}) = {self.fed_avg_loss[-1]}/{self.fed_avg_acc[-1]}   "
            f"Loss/acc={self.list_loss[-1]}/{self.list_acc[-1]}"
        )
        self.logger.info("the density is " + ",".join([f"{client_density:.2f}" for client_density in self.client_density]))
        self.logger.info("self.train_number :" + str(self.train_number))
        self.logger.info("sum_server_upload :" + str(self.sum_server_upload[-1]) + "_" + str(time_download))

    def save_metrics(self):
        from utils.save_load import mkdir_save

        mkdir_save(self.list_acc, os.path.join(self.save_path, "self.list_acc.pt"))
        mkdir_save(self.fed_avg_acc, os.path.join(self.save_path, "fed_avg_acc.pt"))
        mkdir_save(self.time, os.path.join(self.save_path, "time.pt"))
        mkdir_save(self.client_density, os.path.join(self.save_path, "client_density.pt"))

    def main(
        self,
        idx,
        list_sd,
        list_num_proc,
        current_time,
        client_density,
        train_number,
        interval_signal,
        average_round_time,
        sum_server_upload,
        sum_server_download,
        time_download,
    ):
        self.interval_signal = interval_signal
        self.average_round_time = average_round_time
        self.train_number = train_number
        self.current_time = current_time
        self.client_density = copy.deepcopy(client_density)

        masked_list_sd = []
        for state_dict, mask in zip(list_sd, self.list_mask):
            if state_dict is None:
                masked_list_sd.append(None)
            else:
                masked_list_sd.append(self.mask_state_dict(state_dict, mask))

        valid_states = [state for state in masked_list_sd if state is not None]
        if not valid_states and not interval_signal:
            return self.list_state_dict, self.client_density

        self_sd = self.model.state_dict()
        fed_avg_sd = copy.deepcopy(self_sd)
        keys = [k for k in self.model.state_dict().keys() if not k.endswith("num_batches_tracked")]

        if valid_states:
            with torch.no_grad():
                sum_weight_dict = {k: torch.zeros_like(self_sd[k], device=self.device) for k in keys}
                sum_mask_dict = {k: torch.zeros_like(self_sd[k], device=self.device) for k in keys}

                for state in valid_states:
                    for key in keys:
                        value = state[key].to(self.device)
                        sum_weight_dict[key] += value
                        sum_mask_dict[key] += (value != 0)

                sd = OrderedDict()
                fed_avg_sd = OrderedDict()
                for key in keys:
                    divisor = torch.where(
                        sum_mask_dict[key] == 0,
                        torch.full_like(sum_mask_dict[key], 1e-10),
                        sum_mask_dict[key],
                    )
                    merged = torch.div(sum_weight_dict[key], divisor)
                    zero_mask = merged == 0
                    merged[zero_mask] = self_sd[key][zero_mask]
                    sd[key] = merged
                    fed_avg_sd[key] = sum_weight_dict[key] / max(len(valid_states), 1)

                self.model.load_state_dict(sd)

        if interval_signal:
            self.sum_server_upload.append(sum_server_upload)
            for i in range(len(self.client_density)):
                self.sum_server_download[i].append(sum_server_download[i])
            self.time.append(current_time)
            self.test_client_model(list_sd)

            if valid_states:
                fed_avg_loss, fed_avg_acc = self.evaluate_with_state_dict(fed_avg_sd)
            else:
                fed_avg_loss, fed_avg_acc = self.model.evaluate(self.test_loader)

            loss, acc = self.model.evaluate(self.test_loader)
            self.list_loss.append(loss)
            self.list_acc.append(acc)
            self.fed_avg_acc.append(fed_avg_acc)
            self.fed_avg_loss.append(fed_avg_loss)

            self.Print_FL_Message(idx, time_download)
            old_client_density = copy.deepcopy(self.client_density)
            self.tick_client_wait_for_stable()
            if self.wait_for_stable <= 0:
                self.GMR()
            else:
                self.wait_for_stable = self.wait_for_stable - 1
            if old_client_density != self.client_density:
                self.logger.info("Adjust model: New density list is " + str(self.client_density))
                print("Adjust model: New density list is " + str(self.client_density))

        self.list_mask = self.roll_split_universal(idx, self.model, self.client_density)
        self.list_state_dict = self.build_client_state_dicts()
        self.save_metrics()
        return self.list_state_dict, self.client_density


class TimedFjordFL:
    def __init__(self, args, config, server, client_list):
        self.args = args
        self.config = config
        self.server = server
        self.client_list = client_list
        self.server.client_list = client_list
        self.interval = args.interval
        self.chronous = args.chronous
        self.client_density = copy.deepcopy(args.client_density)
        self.client_state = [True] * len(client_list)
        self.average_download_speed = args.average_download_speed
        self.average_upload_speed = args.average_upload_speed
        self.variance_download = 0.3
        self.variance_upload = 0.3
        self.average_server_down = 50
        self.variance_server_down = 0.1
        self.average_server_up = args.server_up_speed
        self.variance_server_up = 0.1
        self.size_client_need_upload = [0] * len(client_list)
        self.server_to_client = [0] * len(client_list)
        self.client_start_work_time = [0] * len(client_list)
        self.list_num = [0] * len(client_list)
        self.train_number = [0] * len(client_list)
        self.last_round_time = [0] * len(client_list)
        self.average_round_time = [deque(maxlen=max(1, self.interval // 2)) for _ in range(len(client_list))]
        self.threshold = 10000
        self.standard_time = 0
        self.previous_idx = -1
        self.interval_signal = False
        self.start_client_idx = len(client_list)
        self.communicate_time_from_server_to_client = [0] * len(client_list)
        self.list_store_model_server = [None for _ in range(len(client_list))]
        self.list_store_model_client = [None for _ in range(len(client_list))]
        self.client_train_time = [[0.0] for _ in range(len(client_list))]
        self.sum_client_train_time = [[0.0] for _ in range(len(client_list))]
        self.client_upload_time = [[0.0] for _ in range(len(client_list))]
        self.sum_client_upload_time = [[0.0] for _ in range(len(client_list))]
        self.client_download_time = [[0.0] for _ in range(len(client_list))]
        self.sum_client_download_time = [[0.0] for _ in range(len(client_list))]
        self.server_merge_time = [0.0]
        self.sum_server_merge_time = [0.0]
        self.waste_time = [0] * len(client_list)
        self.sum_server_upload = 0
        self.sum_server_download = [0] * len(client_list)
        self.sum_time_download = 0
        self.server_up_complete = 0
        self.list_loss = [0]
        self.list_acc = [0]
        self.list_est_time = [0]
        self.list_model_size = [0]

    def get_internet_speed(self):
        download_speed, upload_speed = [], []
        server_up = 0
        server_down = 0
        n = 20
        for _ in range(n):
            server_up += np.random.lognormal(mean=0, sigma=self.variance_server_up) * self.average_server_up
            server_down += np.random.lognormal(mean=0, sigma=self.variance_server_down) * self.average_server_down
        server_up /= n
        server_down /= n

        for i in range(len(self.average_upload_speed)):
            dp = 0
            up = 0
            for _ in range(n):
                dp += np.random.lognormal(mean=0, sigma=self.variance_upload) * self.average_download_speed[i]
                up += np.random.lognormal(mean=0, sigma=self.variance_download) * self.average_upload_speed[i]
            download_speed.append(dp / n)
            upload_speed.append(up / n)
        return server_up, server_down, download_speed, upload_speed

    def process_list(self, input_list):
        index_dict = {}
        for index, value in enumerate(input_list):
            if value not in index_dict:
                index_dict[value] = []
            index_dict[value].append(index)
        unique_sorted_values = sorted(index_dict.keys())
        indices_lists = [index_dict[value] for value in unique_sorted_values]
        return unique_sorted_values, indices_lists

    def train_client(self):
        list_state_dict, density = [], []
        for i, client in enumerate(self.client_list):
            if self.client_state[i]:
                sd, npc, ds = client.main()
                self.client_train_time[i].append(self.config.asyn_interval)
                self.sum_client_train_time[i].append(
                    self.sum_client_train_time[i][-1] + self.client_train_time[i][-1]
                )
                self.list_store_model_server[i] = copy.deepcopy(sd)
                list_state_dict.append(sd)
                self.train_number[i] += 1
                self.list_num[i] = npc
                density.append(ds)
            else:
                list_state_dict.append(None)
                density.append(self.client_density[i])
        return list_state_dict, density

    def simulate_client_to_server(self, density, begin_time_client_upload, list_state_dict):
        model_size = self.server.get_real_size(list_state_dict, self.server.experiment_name, density)
        for i in range(len(self.client_list)):
            self.sum_server_download[i] += model_size[i]

        server_up, server_down, download_speed, upload_speed = self.get_internet_speed()
        self.size_client_need_upload = [cs + ms for cs, ms in zip(self.size_client_need_upload, model_size)]

        time_client_begin_transmission = []
        for i in range(len(self.client_state)):
            if self.client_state[i]:
                time_client_begin_transmission.append(
                    self.communicate_time_from_server_to_client[i] + self.client_train_time[i][-1]
                )
                begin_time_client_upload[i] = time_client_begin_transmission[i]
            else:
                time_client_begin_transmission.append(self.communicate_time_from_server_to_client[i])

        from control.sub_algorithm import simulate_client_to_server as cts

        server_receive_time = np.array(
            cts(
                time_client_begin_transmission,
                copy.copy(self.size_client_need_upload),
                upload_speed,
                server_down,
            )
        )
        server_close_time = copy.copy(server_receive_time)

        if self.args.merge == "fedasyn":
            max_true_client = 1
            if self.start_client_idx == len(server_close_time):
                server_download_sequence = list(range(len(server_close_time)))
            else:
                server_download_sequence = list(range(self.start_client_idx, len(server_close_time))) + list(
                    range(0, self.start_client_idx)
                )
        else:
            max_true_client = len(server_close_time)
            copy_time_client_begin_transmission = [
                0 if not self.client_state[i] else time_client_begin_transmission[i]
                for i in range(len(time_client_begin_transmission))
            ]
            server_download_sequence = [
                idx for idx, _ in sorted(enumerate(copy_time_client_begin_transmission), key=lambda x: x[1])
            ]

        for i in server_download_sequence:
            active = self.client_state[i]
            time_begin = time_client_begin_transmission[i]
            size_left = self.size_client_need_upload[i]

            if active:
                if server_close_time[i] > self.threshold:
                    self.client_state[i] = False
                    self.size_client_need_upload[i] = size_left - upload_speed[i] * (self.threshold - time_begin)
                    self.size_client_need_upload[i] = max(self.size_client_need_upload[i], 0.001)
                    server_close_time[i] = self.threshold
                else:
                    if max_true_client > 0:
                        self.size_client_need_upload[i] = 0
                        self.client_upload_time[i].append(server_close_time[i] - begin_time_client_upload[i])
                        max_true_client -= 1
                    else:
                        self.size_client_need_upload[i] = size_left - upload_speed[i] * (self.threshold - time_begin)
                        self.size_client_need_upload[i] = max(self.size_client_need_upload[i], 0.001)
                        self.client_state[i] = False
            else:
                if size_left > 0:
                    if server_close_time[i] > self.threshold:
                        self.size_client_need_upload[i] = size_left - upload_speed[i] * (self.threshold - time_begin)
                        self.size_client_need_upload[i] = max(self.size_client_need_upload[i], 0.001)
                        server_close_time[i] = self.threshold
                    else:
                        if max_true_client > 0:
                            self.size_client_need_upload[i] = 0
                            list_state_dict[i] = copy.deepcopy(self.list_store_model_server[i])
                            self.client_state[i] = True
                            self.client_upload_time[i].append(server_close_time[i] - begin_time_client_upload[i])
                            self.start_client_idx = i + 1
                            max_true_client -= 1
                        else:
                            self.size_client_need_upload[i] = 0.001
                            self.client_state[i] = False
                            server_close_time[i] = self.threshold
                if self.server_to_client[i] > 0:
                    server_receive_time[i] = self.threshold + 1000

        return (
            list_state_dict,
            server_up,
            download_speed,
            upload_speed,
            server_close_time,
            server_receive_time,
            begin_time_client_upload,
        )

    def update_time_tracking(
        self,
        idx,
        list_state_dict,
        begin_time_client_upload,
        standard_client,
        server_close_time,
        server_receive_time,
        download_speed,
        upload_speed,
    ):
        if idx < 2:
            self.standard_time = max(server_close_time)
            return

        if self.chronous.lower().startswith("syn"):
            for i in range(len(self.client_density)):
                self.average_round_time[i].append(server_close_time[i] - self.standard_time)
            self.standard_time = max(server_close_time)
        else:
            if self.args.merge != "fedasyn":
                new_threshold = max(server_receive_time[:standard_client]) + self.config.asyn_interval
                sum_time = new_threshold - self.threshold
                for i in range(len(self.client_list)):
                    if not self.client_state[i]:
                        if self.server_to_client[i] > 0:
                            self.server_to_client[i] = (
                                0.001
                                if self.server_to_client[i] <= download_speed[i] * sum_time
                                else self.server_to_client[i] - download_speed[i] * sum_time
                            )
                            server_close_time[i] = new_threshold
                        if self.size_client_need_upload[i] > 0:
                            if server_receive_time[i] <= new_threshold:
                                self.size_client_need_upload[i] = 0
                                list_state_dict[i] = copy.deepcopy(self.list_store_model_server[i])
                                self.client_state[i] = True
                                self.client_upload_time[i].append(
                                    server_close_time[i] - begin_time_client_upload[i]
                                )
                            else:
                                self.size_client_need_upload[i] -= upload_speed[i] * sum_time
                                server_close_time[i] = new_threshold
                                if self.size_client_need_upload[i] < 0:
                                    self.size_client_need_upload[i] = 0.001
                self.threshold = new_threshold

            for i in range(len(self.client_density)):
                if self.client_state[i]:
                    self.average_round_time[i].append(server_close_time[i] - self.last_round_time[i])
                    self.last_round_time[i] = server_close_time[i]

            self.standard_time = self.threshold
            for i in range(len(self.client_density)):
                if self.client_state[i]:
                    self.waste_time[i] += self.threshold - server_close_time[i]

    def prepare_server_to_client_transmission(
        self,
        list_state_dict,
        download_speed,
        upload_speed,
        begin_time_server_merge,
        begin_time_client_download,
    ):
        sum_time = 0
        if any(self.client_state):
            self.standard_time += self.config.asyn_interval

        if self.client_state[0]:
            self.server_merge_time.append(self.standard_time - begin_time_server_merge)

        for i in range(len(self.client_density)):
            if self.client_state[i]:
                begin_time_client_download[i] = self.standard_time

        false_count = sum([not st for st in self.client_state])
        if false_count != len(self.client_density):
            for i in range(len(self.client_density)):
                if not self.client_state[i]:
                    if self.server_to_client[i] > 0:
                        self.server_to_client[i] = max(
                            0.001,
                            self.server_to_client[i] - download_speed[i] * sum_time,
                        )
                    if self.size_client_need_upload[i] > 0:
                        self.size_client_need_upload[i] = max(
                            0.001,
                            self.size_client_need_upload[i] - upload_speed[i] * sum_time,
                        )

        list_client_size = self.server.get_real_size(list_state_dict, self.server.experiment_name, self.client_density)
        list_upload_size = [0] * len(self.client_density)
        for i in range(len(self.client_density)):
            if self.client_state[i]:
                list_upload_size[i] = list_client_size[i]

        list_upload_size, sort_perm = self.process_list(list_upload_size)
        self.sum_server_upload += sum(list_upload_size)

        for i in range(len(self.client_list)):
            self.client_start_work_time[i] = self.server_to_client[i] / download_speed[i]
            if self.client_state[i]:
                self.server_to_client[i] += list_client_size[i]

        self.sum_time_download += self.standard_time - self.server_up_complete
        self.server_up_complete = 0
        return list_upload_size, list_client_size, sort_perm

    def simulate_server_to_client_transmission(
        self,
        list_upload_size,
        list_client_size,
        sort_perm,
        download_speed,
        server_up,
        standard_client_number,
    ):
        time_from_server_to_client = [0] * len(self.client_list)
        for i in range(len(sort_perm)):
            for j in sort_perm[i]:
                time_from_server_to_client[j] = (
                    self.server_up_complete + list_upload_size[i] / min(server_up, download_speed[j])
                )
            self.server_up_complete += list_upload_size[i] / server_up

        self.server_up_complete += self.standard_time

        for i in range(len(self.client_list)):
            if not self.client_state[i]:
                time_from_server_to_client[i] = self.server_to_client[i] / download_speed[i]

        if self.chronous.lower().startswith("syn"):
            self.threshold = self.standard_time + 1000
        else:
            topK_time = max(time_from_server_to_client[:standard_client_number])
            delay_time = self.server_up_complete - self.standard_time
            self.threshold = self.standard_time + max(topK_time, delay_time) + self.config.asyn_interval

        return time_from_server_to_client

    def finalize_download_states(self, time_from_server_to_client, begin_time_client_download, list_state_dict, download_speed):
        for i in range(len(self.client_list)):
            if self.client_state[i]:
                if time_from_server_to_client[i] + self.standard_time > self.threshold:
                    self.list_store_model_client[i] = copy.deepcopy(list_state_dict[i])
                    self.server_to_client[i] = (
                        time_from_server_to_client[i] + self.standard_time - self.threshold
                    ) * download_speed[i]
                    self.server_to_client[i] = max(0.001, self.server_to_client[i])
                    self.client_state[i] = False
                    time_from_server_to_client[i] = self.threshold - self.standard_time
                else:
                    self.server_to_client[i] = 0
                    self.client_download_time[i].append(
                        time_from_server_to_client[i] + self.standard_time - begin_time_client_download[i]
                    )
            else:
                if self.server_to_client[i] > 0:
                    if time_from_server_to_client[i] + self.standard_time > self.threshold:
                        self.server_to_client[i] -= download_speed[i] * (self.threshold - self.standard_time)
                        self.server_to_client[i] = max(0.001, self.server_to_client[i])
                        time_from_server_to_client[i] = self.threshold - self.standard_time
                    else:
                        self.server_to_client[i] = 0
                        time_from_server_to_client[i] = self.client_start_work_time[i]
                        self.client_state[i] = True
                        self.client_download_time[i].append(
                            time_from_server_to_client[i] + self.standard_time - begin_time_client_download[i]
                        )
                        list_state_dict[i] = copy.deepcopy(self.list_store_model_client[i])

                if self.size_client_need_upload[i] > 0:
                    time_from_server_to_client[i] = 0

        return list_state_dict, time_from_server_to_client

    def apply_model_and_schedule_to_clients(self, list_state_dict):
        global_lr = None
        if self.server.ip_optimizer_wrapper is not None and self.client_state[0]:
            self.server.ip_optimizer_wrapper.lr_scheduler_step()
            global_lr = self.server.ip_optimizer_wrapper.get_last_lr()

        for i in range(len(self.client_state)):
            if self.client_state[i]:
                client = self.client_list[i]
                client.load_state_dict(list_state_dict[i])
                client.load_rate(self.client_density[i])
                if global_lr is not None:
                    client.load_lr(global_lr)

    def main(self):
        idx = self.train_number[0]
        begin_time_server_merge = 0
        begin_time_client_download = [0] * len(self.client_state)
        standard_client_number = max(1, int(self.args.part[0] * self.args.number_clients))

        while True:
            self.interval_signal = idx % self.interval == 0 and idx != self.previous_idx
            self.previous_idx = idx

            if self.standard_time > self.config.MAX_TIME or idx > self.config.MAX_ROUND or self.server.early_stop:
                self.server.save_metrics()
                return 3

            list_state_dict, density = self.train_client()

            begin_time_client_upload = [0] * len(self.client_state)
            (
                list_state_dict,
                server_up,
                download_speed,
                upload_speed,
                server_close_time,
                server_receive_time,
                begin_time_client_upload,
            ) = self.simulate_client_to_server(density, begin_time_client_upload, list_state_dict)

            self.update_time_tracking(
                idx,
                list_state_dict,
                begin_time_client_upload,
                standard_client_number,
                server_close_time,
                server_receive_time,
                download_speed,
                upload_speed,
            )

            if self.client_state[0]:
                begin_time_server_merge = self.standard_time

            list_state_dict, self.client_density = self.server.main(
                idx,
                list_state_dict,
                self.list_num,
                self.standard_time,
                self.client_density,
                self.train_number,
                self.interval_signal,
                self.average_round_time,
                self.sum_server_upload,
                self.sum_server_download,
                self.sum_time_download,
            )

            list_upload_size, list_client_size, sort_perm = self.prepare_server_to_client_transmission(
                list_state_dict,
                download_speed,
                upload_speed,
                begin_time_server_merge,
                begin_time_client_download,
            )

            time_from_server_to_client = self.simulate_server_to_client_transmission(
                list_upload_size,
                list_client_size,
                sort_perm,
                download_speed,
                server_up,
                standard_client_number,
            )

            list_state_dict, time_from_server_to_client = self.finalize_download_states(
                time_from_server_to_client,
                begin_time_client_download,
                list_state_dict,
                download_speed,
            )

            self.apply_model_and_schedule_to_clients(list_state_dict)
            self.communicate_time_from_server_to_client = [
                self.standard_time + sc for sc in time_from_server_to_client
            ]
            idx = self.train_number[0]
