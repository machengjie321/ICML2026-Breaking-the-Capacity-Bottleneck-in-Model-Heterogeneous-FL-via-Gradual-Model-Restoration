#!/usr/bin/env bash

# CIFAR10 command catalog.
# Commands are intentionally commented out. Remove the leading `#` to run.
#
# Method notes:
# - `pr_fl`: full FedGMR setting used in the main comparison.
# - `gmr`: GMR-only ablation.
# - `asyn`: recovery on, but replace asynchronous scheduling with synchronous scheduling.
# - `mask_fed_avg`: replace mask-aware aggregation with FedAvg-style aggregation, without recovery.
# - `re_mask_fed_avg`: FedAvg-style aggregation, with recovery enabled.
# - `gradient_avg`: replace aggregation with gradient averaging, without recovery.
# - `re_gradient_avg`: gradient averaging, with recovery enabled.

# ========================= FedGMR / pr_fl =========================
# IID:   high=10, medium=25, low=1
# Non-IID: high=15, medium=30, low=1

# python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex pr_fl -num_clients 10 -sample_client high   -patience 10 -bp
# python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex pr_fl -num_clients 10 -sample_client medium -patience 25 -bp
# python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex pr_fl -num_clients 10 -sample_client low    -patience 1  -bp
# python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex pr_fl -num_clients 10 -sample_client high   -patience 15 -niid -bp
# python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex pr_fl -num_clients 10 -sample_client medium -patience 30 -niid -bp
# python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex pr_fl -num_clients 10 -sample_client low    -patience 1  -niid -bp

# ========================= Ablation =========================
# Keep the same patience as the matched `pr_fl` run.
# Main ablation variants: `gmr`, `asyn`, `mask_fed_avg`, `re_mask_fed_avg`, `gradient_avg`, `re_gradient_avg`.

# IID high
# python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex gmr             -num_clients 10 -sample_client high   -patience 10 -bp
# python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex asyn            -num_clients 10 -sample_client high   -patience 10 -bp
# python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex mask_fed_avg    -num_clients 10 -sample_client high   -patience 10 -bp
# python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex re_mask_fed_avg -num_clients 10 -sample_client high   -patience 10 -bp
# python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex gradient_avg    -num_clients 10 -sample_client high   -patience 10 -bp
# python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex re_gradient_avg -num_clients 10 -sample_client high   -patience 10 -bp

# IID medium
# python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex gmr             -num_clients 10 -sample_client medium -patience 25 -bp
# python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex asyn            -num_clients 10 -sample_client medium -patience 25 -bp
# python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex mask_fed_avg    -num_clients 10 -sample_client medium -patience 25 -bp
# python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex re_mask_fed_avg -num_clients 10 -sample_client medium -patience 25 -bp
# python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex gradient_avg    -num_clients 10 -sample_client medium -patience 25 -bp
# python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex re_gradient_avg -num_clients 10 -sample_client medium -patience 25 -bp

# IID low
# python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex gmr             -num_clients 10 -sample_client low    -patience 1 -bp
# python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex asyn            -num_clients 10 -sample_client low    -patience 1 -bp
# python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex mask_fed_avg    -num_clients 10 -sample_client low    -patience 1 -bp
# python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex re_mask_fed_avg -num_clients 10 -sample_client low    -patience 1 -bp
# python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex gradient_avg    -num_clients 10 -sample_client low    -patience 1 -bp
# python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex re_gradient_avg -num_clients 10 -sample_client low    -patience 1 -bp

# Non-IID high
# python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex gmr             -num_clients 10 -sample_client high   -patience 15 -niid -bp
# python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex asyn            -num_clients 10 -sample_client high   -patience 15 -niid -bp
# python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex mask_fed_avg    -num_clients 10 -sample_client high   -patience 15 -niid -bp
# python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex re_mask_fed_avg -num_clients 10 -sample_client high   -patience 15 -niid -bp
# python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex gradient_avg    -num_clients 10 -sample_client high   -patience 15 -niid -bp
# python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex re_gradient_avg -num_clients 10 -sample_client high   -patience 15 -niid -bp

# Non-IID medium
# python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex gmr             -num_clients 10 -sample_client medium -patience 30 -niid -bp
# python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex asyn            -num_clients 10 -sample_client medium -patience 30 -niid -bp
# python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex mask_fed_avg    -num_clients 10 -sample_client medium -patience 30 -niid -bp
# python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex re_mask_fed_avg -num_clients 10 -sample_client medium -patience 30 -niid -bp
# python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex gradient_avg    -num_clients 10 -sample_client medium -patience 30 -niid -bp
# python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex re_gradient_avg -num_clients 10 -sample_client medium -patience 30 -niid -bp

# Non-IID low
# python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex gmr             -num_clients 10 -sample_client low    -patience 1 -niid -bp
# python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex asyn            -num_clients 10 -sample_client low    -patience 1 -niid -bp
# python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex mask_fed_avg    -num_clients 10 -sample_client low    -patience 1 -niid -bp
# python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex re_mask_fed_avg -num_clients 10 -sample_client low    -patience 1 -niid -bp
# python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex gradient_avg    -num_clients 10 -sample_client low    -patience 1 -niid -bp
# python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex re_gradient_avg -num_clients 10 -sample_client low    -patience 1 -niid -bp

# ========================= Baselines =========================
# Baseline patience is not tuned here. Canonical value: 5.

# IID high
# python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex fed_avg   -num_clients 10 -sample_client high   -patience 5 -bp
# python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex fed_asyn  -num_clients 10 -sample_client high   -patience 5 -bp
# python experiments/CIFAR10/Syn_modelhetero.py       -i 25             -ex heterofl  -num_clients 10 -sample_client high   -patience 5 -bp
# python experiments/CIFAR10/Syn_modelhetero.py       -i 25             -ex fedrolex  -num_clients 10 -sample_client high   -patience 5 -bp
# python experiments/CIFAR10/fjord.py                 -i 25             -ex fjord      -num_clients 10 -sample_client high   -patience 5
# python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex fiarse    -num_clients 10 -sample_client high   -patience 5 -bp

# IID medium
# python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex fed_avg   -num_clients 10 -sample_client medium -patience 5 -bp
# python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex fed_asyn  -num_clients 10 -sample_client medium -patience 5 -bp
# python experiments/CIFAR10/Syn_modelhetero.py       -i 25             -ex heterofl  -num_clients 10 -sample_client medium -patience 5 -bp
# python experiments/CIFAR10/Syn_modelhetero.py       -i 25             -ex fedrolex  -num_clients 10 -sample_client medium -patience 5 -bp
# python experiments/CIFAR10/fjord.py                 -i 25             -ex fjord      -num_clients 10 -sample_client medium -patience 5
# python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex fiarse    -num_clients 10 -sample_client medium -patience 5 -bp

# IID low
# python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex fed_avg   -num_clients 10 -sample_client low    -patience 5 -bp
# python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex fed_asyn  -num_clients 10 -sample_client low    -patience 5 -bp
# python experiments/CIFAR10/Syn_modelhetero.py       -i 25             -ex heterofl  -num_clients 10 -sample_client low    -patience 5 -bp
# python experiments/CIFAR10/Syn_modelhetero.py       -i 25             -ex fedrolex  -num_clients 10 -sample_client low    -patience 5 -bp
# python experiments/CIFAR10/fjord.py                 -i 25             -ex fjord      -num_clients 10 -sample_client low    -patience 5
# python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex fiarse    -num_clients 10 -sample_client low    -patience 5 -bp

# Non-IID high
# python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex fed_avg   -num_clients 10 -sample_client high   -patience 5 -niid -bp
# python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex fed_asyn  -num_clients 10 -sample_client high   -patience 5 -niid -bp
# python experiments/CIFAR10/Syn_modelhetero.py       -i 25             -ex heterofl  -num_clients 10 -sample_client high   -patience 5 -niid -bp
# python experiments/CIFAR10/Syn_modelhetero.py       -i 25             -ex fedrolex  -num_clients 10 -sample_client high   -patience 5 -niid -bp
# python experiments/CIFAR10/fjord.py                 -i 25             -ex fjord      -num_clients 10 -sample_client high   -patience 5 -niid
# python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex fiarse    -num_clients 10 -sample_client high   -patience 5 -niid -bp

# Non-IID medium
# python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex fed_avg   -num_clients 10 -sample_client medium -patience 5 -niid -bp
# python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex fed_asyn  -num_clients 10 -sample_client medium -patience 5 -niid -bp
# python experiments/CIFAR10/Syn_modelhetero.py       -i 25             -ex heterofl  -num_clients 10 -sample_client medium -patience 5 -niid -bp
# python experiments/CIFAR10/Syn_modelhetero.py       -i 25             -ex fedrolex  -num_clients 10 -sample_client medium -patience 5 -niid -bp
# python experiments/CIFAR10/fjord.py                 -i 25             -ex fjord      -num_clients 10 -sample_client medium -patience 5 -niid
# python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex fiarse    -num_clients 10 -sample_client medium -patience 5 -niid -bp

# Non-IID low
# python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex fed_avg   -num_clients 10 -sample_client low    -patience 5 -niid -bp
# python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex fed_asyn  -num_clients 10 -sample_client low    -patience 5 -niid -bp
# python experiments/CIFAR10/Syn_modelhetero.py       -i 25             -ex heterofl  -num_clients 10 -sample_client low    -patience 5 -niid -bp
# python experiments/CIFAR10/Syn_modelhetero.py       -i 25             -ex fedrolex  -num_clients 10 -sample_client low    -patience 5 -niid -bp
# python experiments/CIFAR10/fjord.py                 -i 25             -ex fjord      -num_clients 10 -sample_client low    -patience 5 -niid
# python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex fiarse    -num_clients 10 -sample_client low    -patience 5 -niid -bp
