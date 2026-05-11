import copy

import torch
from timeit import default_timer as timer

from sympy.core.symbol import uniquely_named_symbol
from sympy.printing.tests.test_codeprinter import test_print_Dummy
from torch import unique

from utils.heap_queue import HeapQueue



@torch.no_grad()
def process_layer(layer, layer_prefix, accumulate_weight: dict, dtp):  # 仅处理一层layers
    """
        :param layer: model.layer
        :param layer_prefix:such as ['feauture.0']
        :param accumulate_weight: the acuuumulate_weight of layer, prepare for restoring the model  to a different density
        :param coeff:
        :param dtp:such as 0.3, the model will be pruned to density=0.3 according to the weights
        :return:
        sorted_tba_values.tolist(): the weight value of the pruned neurons in this layer, in decreasing order
        sorted_tba_indices: the idx of sorted_tba_values in original layer
    """
    dtp = 1 - dtp

    w_name = "{}.weight".format(layer_prefix)
    b_name = "{}.bias".format(layer_prefix)
    sqg = accumulate_weight[w_name]
    iu_mask, niu_mask = layer.mask == 1., layer.mask == 0.
    num_iu, num_niu = iu_mask.sum().item(), niu_mask.sum().item()

    if num_iu == 0:
        empty_values = sqg.new_empty((0,))
        empty_indices = torch.empty((0, layer.weight.dim()), dtype=torch.long, device=sqg.device)
        return empty_values, empty_indices

    # Decrease
    max_dec_num = int(dtp * num_iu)
    w_iu = layer.weight[iu_mask]  # use magnitude
    max_dec_num = min(max_dec_num, w_iu.numel() - 1)
    w_thr = torch.sort(torch.abs(w_iu))[0][max_dec_num]
    # tbk_mask = (torch.abs(layer.weight) >= w_thr) * iu_mask
    tba_dec_mask = (torch.abs(layer.weight) < w_thr) * iu_mask

    # Increase
    tba_inc_mask = niu_mask

    # total_sqg = sqg[tbk_mask].sum().item()
    # if b_name in accumulate_weight.keys():
    #     total_sqg += accumulate_weight[b_name].sum().item()
    # total_time = coeff * tbk_mask.sum().item()
    tba_mask = tba_dec_mask + tba_inc_mask
    tba_values, tba_indices = sqg[tba_mask], tba_mask.nonzero()
    sorted_tba_values, sort_perm = torch.sort(tba_values, descending=True)
    sorted_tba_indices = tba_indices[sort_perm]

    layer.prune_by_pct(dtp)
    return sorted_tba_values, sorted_tba_indices #the sqg of the decending sorted and idx





# k-way merge sort
def k_way_merge_sort(tensors):
    if len(tensors) == 1:
        return tensors[0]
    mid = len(tensors) // 2
    left_sorted = k_way_merge_sort(tensors[:mid])
    right_sorted = k_way_merge_sort(tensors[mid:])
    merged = torch.cat((left_sorted, right_sorted))
    return merged[merged[:, 0].argsort()]





def _prunable_weight_keys(model):
    return [f"{prefix}.weight" for prefix in model.prunable_layer_prefixes]


@torch.no_grad()
def _topk_flat_indices(score_dict: dict, weight_keys, k: int):
    flat_scores = []
    for key in weight_keys:
        if key not in score_dict:
            continue
        tensor = score_dict[key]
        if tensor is None:
            continue
        flat_scores.append(torch.abs(tensor).reshape(-1))
    if len(flat_scores) == 0:
        return set(), 0

    merged_scores = torch.cat(flat_scores, dim=0)
    total = int(merged_scores.numel())
    if total == 0:
        return set(), 0
    k = max(1, min(int(k), total))
    topk_idx = torch.topk(merged_scores, k=k, largest=True, sorted=False).indices
    return set(topk_idx.detach().cpu().tolist()), total


@torch.no_grad()
def _build_client_replaced_score_dict(
    global_score_dict: dict,
    client_state_dict: dict,
    client_mask_dict: dict,
    score_mode: str = "wg",
    grad_dict: dict = None,
):
    local_score_dict = {k: v.clone() if torch.is_tensor(v) else v for k, v in global_score_dict.items()}
    if client_state_dict is None or client_mask_dict is None:
        return local_score_dict

    for key, base_score in global_score_dict.items():
        if not key.endswith(".weight"):
            continue
        if key not in client_state_dict or key not in client_mask_dict:
            continue
        if not torch.is_tensor(base_score):
            continue

        mask = client_mask_dict[key]
        client_weight = client_state_dict[key]
        if not torch.is_tensor(mask) or not torch.is_tensor(client_weight):
            continue
        if client_weight.shape != base_score.shape or mask.shape != base_score.shape:
            continue

        device = base_score.device
        mask = mask.to(device=device, dtype=torch.bool)
        client_weight = client_weight.to(device=device)

        mode = (score_mode or "wg").lower()
        if mode.startswith("wg"):
            if grad_dict is not None and key in grad_dict and torch.is_tensor(grad_dict[key]) and grad_dict[key].shape == base_score.shape:
                replace_score = torch.abs(grad_dict[key].to(device=device) * client_weight)
            else:
                replace_score = torch.abs(client_weight)
        elif mode == "g":
            if grad_dict is not None and key in grad_dict and torch.is_tensor(grad_dict[key]) and grad_dict[key].shape == base_score.shape:
                replace_score = torch.abs(grad_dict[key].to(device=device))
            else:
                replace_score = torch.abs(base_score)
        else:
            replace_score = torch.abs(client_weight)

        local_score_dict[key][mask] = replace_score[mask]

    return local_score_dict


@torch.no_grad()
def _compute_topk_overlap_report(
    model,
    global_score_dict: dict,
    client_density: list,
    client_contexts: list,
    score_mode: str = "wg",
    grad_dict: dict = None,
    topk_ratio: float = 0.1,
):
    if client_contexts is None or len(client_contexts) == 0:
        return None

    weight_keys = _prunable_weight_keys(model)
    if len(weight_keys) == 0:
        return None

    client_reports = []
    overlap_values = []
    fixed_topk_targets = [10, 50, 100]
    fixed_topk_reports = {k: [] for k in fixed_topk_targets}
    fixed_topk_means = {}

    for client_id, density in enumerate(client_density):
        if client_id >= len(client_contexts):
            break

        context = client_contexts[client_id] or {}
        local_score_dict = _build_client_replaced_score_dict(
            global_score_dict=global_score_dict,
            client_state_dict=context.get("state_dict"),
            client_mask_dict=context.get("mask"),
            score_mode=score_mode,
            grad_dict=grad_dict,
        )

        total_probe = 0
        for key in weight_keys:
            if key in global_score_dict and torch.is_tensor(global_score_dict[key]):
                total_probe += int(global_score_dict[key].numel())
        if total_probe <= 0:
            continue

        if density is None:
            k = int(total_probe * max(0.0, min(float(topk_ratio), 1.0)))
            density_value = None
        else:
            k = int(total_probe * max(0.0, min(float(density), 1.0)))
            density_value = float(density)
        k = max(1, k)

        global_topk, _ = _topk_flat_indices(global_score_dict, weight_keys, k)
        local_topk, _ = _topk_flat_indices(local_score_dict, weight_keys, k)
        overlap = len(global_topk & local_topk) / max(1, k)

        overlap_values.append(overlap)
        client_reports.append(
            {
                "client_id": client_id,
                "density": density_value,
                "k": int(k),
                "overlap": float(overlap),
            }
        )

        for fixed_k in fixed_topk_targets:
            fixed_k_eff = max(1, min(int(fixed_k), int(total_probe)))
            global_fixed_topk, _ = _topk_flat_indices(global_score_dict, weight_keys, fixed_k_eff)
            local_fixed_topk, _ = _topk_flat_indices(local_score_dict, weight_keys, fixed_k_eff)
            fixed_overlap = len(global_fixed_topk & local_fixed_topk) / max(1, fixed_k_eff)
            fixed_topk_reports[fixed_k].append(
                {
                    "client_id": client_id,
                    "k": int(fixed_k_eff),
                    "overlap": float(fixed_overlap),
                }
            )

    if len(client_reports) == 0:
        return None

    for fixed_k in fixed_topk_targets:
        entries = fixed_topk_reports.get(fixed_k, [])
        if len(entries) == 0:
            continue
        fixed_topk_means[fixed_k] = float(
            sum(item["overlap"] for item in entries) / len(entries)
        )

    return {
        "mean_overlap": float(sum(overlap_values) / len(overlap_values)),
        "client_reports": client_reports,
        "score_mode": str(score_mode),
        "probe_mode": "density",
        "topk_ratio_fallback": float(topk_ratio),
        "fixed_topk_means": fixed_topk_means,
        "fixed_topk_reports": fixed_topk_reports,
    }


@torch.no_grad()
def sub_architecture_search_fast(model,list_coefficient, list_tba_values, list_tba_indices, client_density:list,
                            max_density=None,use_coeff = False):
    """
    Restore the model to different densities according to accumulate_weight

    :param model: server.model, After sub fedavg
    :param list_coefficient: the coefficient of layer, Following the cofficient in Prune_FL, referring to the time used for each layer
    :param list_tba_values:the pruned value of each layer, decreasing sort
    :param list_tba_indices:the list of idx for list_tba_values
    :param client_density:the target density for every client
    :param max_density:

    :return:
    list_state_dict:[[0,1,2,3],[4,5],[6,7],[8],[9,10]]
    model_idx: [[0,1,2],[0],[0,1],[0,1,2,3],[0,1,2,3,4]],but then i need to transform into simluate_server_to_client.client_recv_list

    # """

    device = model.prunable_layers[0].mask.device
    cpu_device = torch.device("cpu")
    sort_time = timer()
    list_n = [0] * len(list_tba_values)
    sqg_index = []
    value_list = []
    index_list = []
    sorted_clientdensity, sorted_clientdensity_indics = torch.sort(
        torch.tensor(client_density, device=device), descending=False)

    list_state_dict = []
    list_sparse_state_dict = []
    list_mask = []
    list_sum_mask = []
    list_threshold = []
    density = []
    sub_model_time = []
    begin_time = timer()
    for index in range(len(list_tba_values)):
        if use_coeff:
            tensor1 = list_tba_values[index] / list_coefficient[index]
        else:
            tensor1 = list_tba_values[index]

        value_list.append(tensor1)
        index_tensor = torch.full_like(tensor1, fill_value=index, device=device)
        index_list.append(index_tensor)

    values = torch.cat(value_list) if len(value_list) > 0 else torch.empty(0, device=device)
    indices = torch.cat(index_list) if len(index_list) > 0 else torch.empty(0, device=device)

    combined = torch.stack((values, indices), dim=1) if values.numel() > 0 else torch.empty((0, 2), device=device)

    if combined.numel() > 0:
        sorted_tensor = combined[combined[:, 0].argsort()]
        sorted_tensor = torch.flip(sorted_tensor, dims=[0])
        index_tensor = sorted_tensor[:, 1]
        sorted_tensor = sorted_tensor[:, 0]
    else:
        index_tensor = torch.empty(0, device=device)
        sorted_tensor = torch.empty(0, device=device)

    sort_time = timer() - sort_time

    total_param_in_use = 0
    total_all_param = 0
    for layer, layer_prefx in zip(model.prunable_layers, model.prunable_layer_prefixes):
        layer_param_in_use = layer.num_weight
        layer_all_param = layer.mask.nelement()
        total_param_in_use += layer_param_in_use
        total_all_param += layer_all_param

    begin_time = timer()
    index = 0
    for i in range(len(sorted_clientdensity)):

        target_density = sorted_clientdensity[i]
        list_n_begin = copy.deepcopy(list_n)
        incre = int(target_density * total_all_param - total_param_in_use)
        if incre < 0:
            incre = 0
        incre_index = index_tensor[index:index + incre]
        for pos in range(len(list_tba_values)):
            list_n[pos] += (incre_index == pos).sum()
        index += incre

        total_param_in_use += incre

        for layer, tba_indices, tba_begin, tba_n in zip(model.prunable_layers, list_tba_indices, list_n_begin, list_n):
            layer.mask[tba_indices[tba_begin:tba_n].t().tolist()] = 1.

        density.append(model.density())

        clean_state_dict = copy.deepcopy(model.state_dict())

        mask_state = {}
        for layer, prefix in zip(model.prunable_layers, model.prunable_layer_prefixes):
            key_w = prefix + ".weight"
            if key_w in clean_state_dict.keys():
                weight = clean_state_dict[key_w]
                w_mask = model.get_mask_by_name(key_w)
                real_weight = (weight * w_mask)
                clean_state_dict[key_w] = real_weight
                mask_state[key_w] = w_mask

        for key, value in clean_state_dict.items():
            clean_state_dict[key] = clean_state_dict[key].cpu()

        if i == 0:
            list_sparse_state_dict.append(copy.deepcopy(clean_state_dict))
        else:
            sd = {}
            for key in model.state_dict().keys():
                sd[key] = (clean_state_dict[key] - last_clean_state_dict[key])

            list_sparse_state_dict.append(sd)
        list_threshold.append(model.get_thresholds())
        last_clean_state_dict = clean_state_dict
        list_state_dict.append(clean_state_dict)
        list_mask.append(copy.deepcopy(mask_state))
        sub_model_time.append(timer() - begin_time)
            
    model_idx = [0] * len(sorted_clientdensity)

    for idx in range(len(model_idx)):
        model_idx[sorted_clientdensity_indics[idx]] = [i for i in range(idx+1)]
    
    
        
    list_thr = []
    client_density = []
    list_sd = []
    list_mk = []
    for idx in range(len(model_idx)):
        client_density.append(density[model_idx[idx][-1]])
        list_sd.append(list_state_dict[model_idx[idx][-1]])
        list_mk.append(list_mask[model_idx[idx][-1]])
        list_thr.append(list_threshold[model_idx[idx][-1]])
        
    model.recover_model()
    return list_sd, model_idx, sort_time, sub_model_time, list_mk, list_sum_mask, list_sparse_state_dict, client_density,list_thr



@torch.no_grad()
def sub_control_fast(model, accumulate_weight_dict: dict, config, client_density:list, min_density,use_coeff = False):
    sum_sqg = 0
    sum_time = config.TIME_CONSTANT
    list_tba_values, list_tba_indices = [], []
    list_coefficient = []

    proc_start = timer()
    comp_coeff_iter = iter(config.COMP_COEFFICIENTS)
    comm_coeff = config.COMM_COEFFICIENT
    for layer, layer_prefix in zip(model.param_layers, model.param_layer_prefixes):
        if layer_prefix in model.prunable_layer_prefixes:
            coeff = comm_coeff + next(comp_coeff_iter)  # 因为对于这个conv2模型是由4个子模块拼接起来的，所以系数只有4个
            sorted_tba_values, sorted_tba_indices = process_layer(layer, layer_prefix, accumulate_weight_dict,
                                                                              min_density)
            if 'transformer' in list(model.state_dict().keys())[0]:
                layer_scale = sorted_tba_values.mean().clamp_min(1e-8)
                coeff = coeff * layer_scale
            
            list_coefficient.append(coeff)
            list_tba_values.append(sorted_tba_values)
            list_tba_indices.append(sorted_tba_indices)
        else:
            w_name = "{}.weight".format(layer_prefix)
            b_name = "{}.bias".format(layer_prefix)

    process_layers_time = timer() - proc_start


    list_state_dict, model_idx, sort_time, sub_model_time, list_mask,list_sum_mask,list_sparse_state_dict,density,list_thr = sub_architecture_search_fast(model, list_coefficient, list_tba_values, list_tba_indices, client_density,use_coeff=use_coeff)
    sub_model_time = [x+process_layers_time+sort_time for x in sub_model_time]

    return list_state_dict, model_idx, sub_model_time, list_mask, list_sum_mask,list_sparse_state_dict,density,list_thr


@torch.no_grad()
def sub_control_client_replace_fast(
    model,
    accumulate_weight_dict: dict,
    config,
    client_density: list,
    min_density,
    use_coeff=False,
    client_contexts=None,
    score_mode: str = "wg",
    grad_dict: dict = None,
    overlap_topk_ratio: float = 0.1,
):
    num_clients = len(client_density)
    client_contexts = client_contexts or [None for _ in range(num_clients)]

    overlap_report = _compute_topk_overlap_report(
        model=model,
        global_score_dict=accumulate_weight_dict,
        client_density=client_density,
        client_contexts=client_contexts,
        score_mode=score_mode,
        grad_dict=grad_dict,
        topk_ratio=overlap_topk_ratio,
    )

    list_state_dict = [None for _ in range(num_clients)]
    list_mask = [None for _ in range(num_clients)]
    list_thr = [None for _ in range(num_clients)]
    sub_model_time = [0.0 for _ in range(num_clients)]

    for client_id in range(num_clients):
        context = client_contexts[client_id] if client_id < len(client_contexts) else None
        context = context or {}

        local_score_dict = _build_client_replaced_score_dict(
            global_score_dict=accumulate_weight_dict,
            client_state_dict=context.get("state_dict"),
            client_mask_dict=context.get("mask"),
            score_mode=score_mode,
            grad_dict=grad_dict,
        )

        local_model = copy.deepcopy(model)
        (local_state_dict,
         _,
         local_sub_model_time,
         local_mask,
         _,
         _,
         _,
         local_thr) = sub_control_fast(
            local_model,
            local_score_dict,
            config,
            [client_density[client_id]],
            min_density,
            use_coeff=use_coeff,
        )

        list_state_dict[client_id] = local_state_dict[0]
        list_mask[client_id] = local_mask[0]
        list_thr[client_id] = local_thr[0]
        sub_model_time[client_id] = float(local_sub_model_time[0]) if len(local_sub_model_time) > 0 else 0.0

    # Keep the same model_idx semantics as baseline implementation.
    device = model.prunable_layers[0].mask.device
    sorted_clientdensity, sorted_clientdensity_indics = torch.sort(
        torch.tensor(client_density, device=device), descending=False
    )
    model_idx = [0] * num_clients
    sorted_client_ids = sorted_clientdensity_indics.tolist()
    for rank, client_id in enumerate(sorted_client_ids):
        model_idx[client_id] = [i for i in range(rank + 1)]

    # Build incremental state dict sequence using sorted client order.
    list_sparse_state_dict = []
    increment_density_hint = []
    prev_state_dict = None
    probe_keys = _prunable_weight_keys(model)
    total_probe = 0
    for key in probe_keys:
        if key in model.state_dict():
            total_probe += int(model.state_dict()[key].numel())
    total_probe = max(1, total_probe)

    for client_id in sorted_client_ids:
        current_state_dict = list_state_dict[client_id]
        if prev_state_dict is None:
            delta_state_dict = copy.deepcopy(current_state_dict)
        else:
            delta_state_dict = {}
            for key in current_state_dict.keys():
                delta_state_dict[key] = current_state_dict[key] - prev_state_dict[key]
        list_sparse_state_dict.append(delta_state_dict)

        nonzero = 0
        for key in probe_keys:
            if key in delta_state_dict and torch.is_tensor(delta_state_dict[key]):
                nonzero += int((delta_state_dict[key] != 0).sum().item())
        increment_density_hint.append(float(nonzero) / float(total_probe))
        prev_state_dict = current_state_dict

    list_sum_mask = []
    return (
        list_state_dict,
        model_idx,
        sub_model_time,
        list_mask,
        list_sum_mask,
        list_sparse_state_dict,
        increment_density_hint,
        list_thr,
        overlap_report,
    )


from utils.functional import disp_num_params

def random_prune(model,client_density):
    device = model.prunable_layers[0].mask.device
    sorted_clientdensity, sorted_clientdensity_indics = torch.sort(
        torch.tensor(client_density, device=device), descending=True)
    prune_model = copy.deepcopy(model)
    current_density = disp_num_params(prune_model)
    list_state_dict = []
    list_mask = []
    end_client_density = []
    for density in sorted_clientdensity:
        prune_model = prune_model.random_prune_by_pct(float(1-density/current_density[0]))
        end_client_density.append(disp_num_params(prune_model))
        current_density = disp_num_params(prune_model)

        mask_state = {}
        clean_state_dict = copy.deepcopy(prune_model.state_dict())
        for layer, prefix in zip(model.prunable_layers, model.prunable_layer_prefixes):
            # works for both layers
            key_w = prefix + ".weight"
            if key_w in clean_state_dict.keys():
                weight = clean_state_dict[key_w]
                w_mask = prune_model.get_mask_by_name(key_w)
                real_weight = (weight * w_mask)
                clean_state_dict[key_w] = real_weight
                mask_state[key_w] = w_mask
        list_state_dict.append(copy.deepcopy(clean_state_dict))
        list_mask.append(copy.deepcopy(mask_state))
    model_idx = [0] * len(sorted_clientdensity)



    for idx in range(len(model_idx)):
        model_idx[sorted_clientdensity_indics[idx]] = [i for i in range(idx + 1)]

    return list_state_dict, model_idx,  list_mask





class ControlModule:
    def __init__(self, model, config, args=None):
        self.model = model
        self.config = config
        self.args = args
        self.accumulate_weight_dict = dict()
        self.old_model = None
        self.g = dict()
        self.g_min = dict()
        self.local_topk_mode = str(getattr(args, "local_topk_mode", "global")).lower() if args is not None else "global"
        if self.local_topk_mode not in {"global", "client_replace"}:
            self.local_topk_mode = "global"
        self.measure_topk_overlap = bool(getattr(args, "measure_topk_overlap", False)) if args is not None else False
        self.overlap_topk_ratio = float(getattr(args, "overlap_topk_ratio", 0.1)) if args is not None else 0.1
        self.last_overlap_report = None
        self.last_increment_density = None

    def use_client_replace_mode(self):
        return self.local_topk_mode == "client_replace"


    @torch.no_grad()
    #i have intended to use the average of the weights of the last ten rounds if the server models as one of the evalution metrics for
    #the importance of each part of the model. however, this function need a lot of storge to store the model of the last ten rounds,
    #so i used an approximation to implement this function, When a new round, i will discard 1/10 of the past information of the model
    # and add 1/10 of latest model in deep learning. where 1/10 is similar to the concept of the learning rate.
    @torch.no_grad()
    def accumulate(self, key, sgrad):
        if key in self.accumulate_weight_dict.keys():
            self.accumulate_weight_dict[key] += sgrad
        else:
            self.accumulate_weight_dict[key] = sgrad

    def accumulate_wg_square(self, old_model=None):
        if self.old_model is not None:
            for key in self.model.state_dict().keys():
                self.g[key] = self.model.state_dict()[key] - old_model[key]
                self.accumulate_weight_dict[key] = torch.abs(self.g[key]*self.model.state_dict()[key])
        else:
            for key in self.model.state_dict().keys():
                self.accumulate_weight_dict[key] = torch.abs(self.model.state_dict()[key])


    def accumulate_w(self, model):
        for key in model.state_dict().keys():
            self.accumulate_weight_dict[key] = torch.abs(model.state_dict()[key]+0)



    def rescale_Im(self, Im_dict, scale=1e10, shift=1.5):
        # 将所有的 Im 值合并到一个列表中
        # Step 1: 找到全局最小和最大值
        all_values = torch.cat([v.view(-1) for v in Im_dict.values()])
        global_min = all_values.min()
        global_max = all_values.max()

        normalized_Im_dict = {}

        # Step 2: 根据全局最大最小值归一化每个 tensor
        for k, v in Im_dict.items():
            normalized_Im_dict[k] = (v - global_min) / (global_max - global_min)

        # Step 3: 根据需要的 scale 和 shift 调整范围
        for k, v in normalized_Im_dict.items():
            normalized_Im_dict[k] = v * scale + shift

        return Im_dict



    @torch.no_grad()
    def accumulate_wg(self, sgrad_to_upload, idx, memory):
        for layer, layer_prefix in zip(self.model.param_layers, self.model.param_layer_prefixes):
            if layer_prefix in self.model.prunable_layer_prefixes:
                w_name = "{}.weight".format(layer_prefix)
                b_name = "{}.bias".format(layer_prefix)

                for key in [w_name, b_name]:
                    if key not in sgrad_to_upload.keys():
                        continue
                    if key not in self.accumulate_weight_dict.keys():
                        self.accumulate_weight_dict[key] = sgrad_to_upload[key] * torch.square(self.model.state_dict()[key])
                    else:
                        mask2 = sgrad_to_upload[key] != 0
                        self.accumulate_weight_dict[key][mask2] = sgrad_to_upload[key][mask2] * torch.square(self.model.state_dict()[key][mask2])

                    mask = self.accumulate_weight_dict[key] == 0
                    mask2 = self.accumulate_weight_dict[key] != 0
                    if len(sgrad_to_upload[key][mask2]) != 0:
                        self.g_min[key] = sgrad_to_upload[key][mask2].min()
                    else:
                        self.g_min[key] = min(self.g_min.values())
                    if len(self.accumulate_weight_dict[key][mask]) != 0:
                        self.accumulate_weight_dict[key][mask] = self.g_min[key] * torch.abs(self.model.state_dict()[key][mask])



    @torch.no_grad()
    def accumulate_g(self,sgrad_to_upload):
        for layer, layer_prefix in zip(self.model.param_layers, self.model.param_layer_prefixes):
            if layer_prefix in self.model.prunable_layer_prefixes:
                w_name = "{}.weight".format(layer_prefix)
                b_name = "{}.bias".format(layer_prefix)

                for key in [w_name,b_name]:
                    if key not in sgrad_to_upload.keys():
                        continue
                    if key not in self.accumulate_weight_dict.keys():
                        self.accumulate_weight_dict[key] = sgrad_to_upload[key]
                        continue
                    mask2 = sgrad_to_upload[key] != 0
                    self.accumulate_weight_dict[key][mask2] = sgrad_to_upload[key][mask2]

                    mask = self.accumulate_weight_dict[key] == 0
                    mask2 = self.accumulate_weight_dict[key] != 0
                    if len(sgrad_to_upload[key][mask2]) != 0:
                        self.g_min[key] = sgrad_to_upload[key][mask2].min()
                    else:
                        self.g_min[key] = min(self.g_min.values())

                    if len(self.accumulate_weight_dict[key][mask]) != 0:
                        self.accumulate_weight_dict[key][mask] = self.g_min[key] * torch.abs(self.model.state_dict()[key][mask])




    @torch.no_grad()
    def sub_adjust_fast(self, client_density: list, use_coff = None, min_density=None, client_contexts=None):
        self.last_overlap_report = None
        score_mode = str(getattr(self.args, "accumulate", "wg")) if self.args is not None else "wg"

        if self.use_client_replace_mode() and client_contexts is not None:
            (
                list_state_dict,
                model_idx,
                sub_model_time,
                list_mask,
                list_sum_mask,
                list_sparse_state_dict,
                density,
                list_thr,
                overlap_report,
            ) = sub_control_client_replace_fast(
                model=self.model,
                accumulate_weight_dict=self.accumulate_weight_dict,
                config=self.config,
                client_density=client_density,
                min_density=min_density,
                use_coeff=use_coff,
                client_contexts=client_contexts,
                score_mode=score_mode,
                grad_dict=self.g,
                overlap_topk_ratio=self.overlap_topk_ratio,
            )
            self.last_overlap_report = overlap_report
            self.last_increment_density = density
            return list_state_dict, model_idx, sub_model_time, list_mask, list_sum_mask, list_sparse_state_dict, density, list_thr

        list_state_dict, model_idx, sub_model_time, list_mask, list_sum_mask, list_sparse_state_dict,density,list_thr = sub_control_fast(
            self.model,
            self.accumulate_weight_dict,
            self.config,
            client_density,
            min_density,
            use_coff,
        )
        self.last_increment_density = density
        if self.measure_topk_overlap and client_contexts is not None:
            self.last_overlap_report = _compute_topk_overlap_report(
                model=self.model,
                global_score_dict=self.accumulate_weight_dict,
                client_density=client_density,
                client_contexts=client_contexts,
                score_mode=score_mode,
                grad_dict=self.g,
                topk_ratio=self.overlap_topk_ratio,
            )
        return list_state_dict, model_idx, sub_model_time, list_mask, list_sum_mask, list_sparse_state_dict,density,list_thr

    @torch.no_grad()
    def partial_model_training(self, client_density: list, use_coff = None, min_density=None):
        min_density = self.config.min_density
        list_state_dict, model_idx, sub_model_time, list_mask, list_sum_mask, list_sparse_state_dict,density,list_thr  = sub_control_fast(self.model, self.accumulate_weight_dict, self.config, client_density, min_density,use_coff)

        return list_mask[0]




import heapq

@torch.no_grad()
def simulate_client_to_server(time, size, upload_speed, server_download_speed):
    # 客户端数量
    n = len(time)
    # 用于存储每个客户端发送文件的状态
    # state[i] = 0 表示第i个客户端还没有开始发送文件
    # state[i] = 1 表示第i个客户端正在发送文件但是还没有发送完成
    # state[i] = 2 表示第i个客户端已经发送完成但是服务器还没有接收完成
    # state[i] = 3 表示第i个客户端的文件已经被服务器接收完成
    state = [0] * n

    # 考虑每个客户端开始发送文件和文件全部上传的时间为上传总流量的变化时间点
    min_time_heap = []
    server_receive_time = [0] * n
    # print(time)
    # print(size)
    # print(upload_speed)
    for i in range(n):
        heapq.heappush(min_time_heap, time[i])
        heapq.heappush(min_time_heap, time[i] + size[i] / upload_speed[i])

    next_time = heapq.heappop(min_time_heap)
    current_time = next_time
    # 存储每个客户端上传到网络的文件大小
    uploaded_size = [0] * n

    next_time = heapq.heappop(min_time_heap)

    client_sequence = []

    # 只有当所有客户端的文件都已经被服务器接收才退出循环
    while True:
        # 存储每个客户端本次循环上传/下载的大小
        cycle_size = [0] * n
        # 遍历每个客户端，计算本次时间段上传文件的总大小
        for i in range(n):
            # 如果第i个客户端已经发送完成并且服务器已经接收完成，则跳过
            if state[i] == 3:
                continue
            # 如果第i个客户端还没有开始发送文件，则判断当前时间是否已经到达发送时间
            if state[i] == 0:
                if current_time == time[i]:
                    state[i] = 1
                    client_sequence.append(i)
            if state[i] == 1:
                cycle_size[i] = min(upload_speed[i] * (next_time - current_time), size[i])
                uploaded_size[i] += cycle_size[i]
                size[i] -= cycle_size[i]
                if size[i] < 1e-9: #if i use == 0, bugs may occur due to errors in high-precision calculation
                    state[i] = 2

        # 模拟服务器下载文件的过程

        max_download_size = server_download_speed * (next_time - current_time)

        for j in client_sequence:
            if max_download_size >= uploaded_size[j]:
                max_download_size -= uploaded_size[j]
                uploaded_size[j] = 0
                if state[j] == 2:
                    state[j] = 3
                    server_receive_time[j] = next_time


            else:
                uploaded_size[j] -= max_download_size

                break

        current_time = next_time

        if len(min_time_heap) > 0:
            next_time = heapq.heappop(min_time_heap)
        else:

            next_time = current_time + max(((sum(uploaded_size) + sum(size)) / server_download_speed), 0.001)

        if sum(state) == n * 3:
            break
        # else:
        #     print(state)
        #     print(size)
        #     print(next_time,current_time)

    return server_receive_time

def determine_density(server_recive_time):
    client_density = []
    return client_density

import heapq

@torch.no_grad()
def simluate_server_to_client(time, client_start_work_time, list_upload_size, server_upload_speed,  sort_perm, client_download_speed):

    time = [0]*10
    client_recv_list = [[0 for _ in range(len(time))] for _ in range(len(time))]


    # index_dict = {}
    # for index, value in enumerate(size):
    #     if value not in index_dict:
    #         index_dict[value] = []
    #     index_dict[value].append(index)
    # unique_sorted_values = sorted(index_dict.keys())
    for file_idx, file_value in enumerate(list_upload_size):
        for client_idx in sort_perm[file_idx]:
            client_recv_list[file_idx][client_idx] = 1
    size = list_upload_size + (len(time)-len(list_upload_size))*[0]



    # for client_idx in range(len(model_idx)):
    #     for file_idx in model_idx[client_idx]:
    #         client_recv_list[file_idx][client_idx] = 1



    # 客户端数目
    n = len(client_download_speed)
    client_receive_time = [-1 for _ in range(n)]
    # 文件数目
    m = len(time)
    import numpy as np
    waiting_recv_number = np.array(client_recv_list).sum(axis=0)

    # 文件
    client_download_size = [[0 for _ in range(n)] for _ in range(m)]
    # 客户端状态
    # 0-未开始接收，1-接收完成
    client_status = [[0 for _ in range(n)] for _ in range(m)]

    # 当前时间
    current_time = 0
    time_increment = 0.05

    time_increment_client_download_bandwidth = [round(i * time_increment, 4) for i in client_download_speed]
    # print(time_increment_client_download_bandwidth)
    # 初始化每个文件的已上传字节数和上传时间点
    file_uploaded_size = [0] * m
    file_uploaded_time = [-1] * m

    # 0:文件尚未上传;1:文件正在上传;2:文件已经成功上传
    server_upload_state = [0] * m

    time_increment_server_current_bandwidth = server_upload_speed * time_increment
    while True:
        current_time = round(current_time + time_increment, 2)
        current_bandwidth = time_increment_server_current_bandwidth
        # 模拟服务器上传情况，以time_increment作为精度模拟服务器文件上传
        for i in range(m):
            if server_upload_state[i] == 2: continue  # 如果第i个文件已经上传，直接进行下一个文件
            if server_upload_state[i] == 0 and current_time >= time[i]:  # 即第i个文件还没上传但是已经到了要上传的时间了
                server_upload_state[i] = 1
            current_number = server_upload_state.count(1)

            if server_upload_state[i] == 1 and size[i] >= 0:  # size[i]是第i个文件还未上传的文件
                if current_bandwidth/current_number >= size[i]:
                    file_uploaded_size[i] += size[i]
                    size[i] = 0
                    server_upload_state[i] = 2
                    file_uploaded_time[i] = current_time

                else:
                    size[i] = size[i] - current_bandwidth/current_number
                    file_uploaded_size[i] += current_bandwidth/current_number



        # 模拟服务器下载情况,以time_increment作为精度模拟服务器文件上传
        client_download_bandwidth = time_increment_client_download_bandwidth.copy()
        for i in range(m):
            for j in range(n):

                # 如果client[j]需要接收文件i
                if current_time <= client_start_work_time[j]:
                    continue
                if client_recv_list[i][j] == 1:
                    # 如果对于客户端j已经没有剩余的带宽了，则跳过客户端j
                    if client_download_bandwidth[j] == 0: continue
                    # 检查对于client[j]是否已经接收i
                    if client_status[i][j] == 1: continue
                    # 如果文件i还未上传
                    if server_upload_state[i] == 0:
                        break
                    # 如果文件已经上传
                    else:
                        # 如果客户端j当前下载的文件i小于服务器已经上传的文件i
                        if client_download_size[i][j] < file_uploaded_size[i]:
                            c = min(file_uploaded_size[i] - client_download_size[i][j], client_download_bandwidth[j])
                            client_download_bandwidth[j] -= c
                            client_download_size[i][j] += c

                        # 如果服务器的文件已经完全上传，并且客户端下载的文件大小与已经上传的大小相同，则代表该客户端已经完全接收了该文件
                        if server_upload_state[i] == 2 and client_download_size[i][j] >= file_uploaded_size[i]:
                            client_status[i][j] = 1

                # if j > 8 and current_time > 1.5 and current_time*100 % 20 == 0:
                #     print(client_download_size)
        # print(client_receive_time)
        client_recved_number = np.array(client_status).sum(axis=0)



        for j in range(n):
            if client_recved_number[j] == waiting_recv_number[j] and client_receive_time[j] == -1:
                client_receive_time[j] = current_time

        if client_recved_number.sum() == waiting_recv_number.sum():
            # print(client_receive_time)
            return client_receive_time
