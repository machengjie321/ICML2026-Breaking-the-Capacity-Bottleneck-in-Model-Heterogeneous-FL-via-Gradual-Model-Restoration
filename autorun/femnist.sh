#!/usr/bin/env bash

# FEMNIST command catalog.
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
# IID:   high=5, medium=4, low=3
# Non-IID: high=7, medium=3, low=1

# python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex pr_fl -num_clients 10 -sample_client high   -patience 5 -bp
# python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex pr_fl -num_clients 10 -sample_client medium -patience 4 -bp
# python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex pr_fl -num_clients 10 -sample_client low    -patience 3 -bp
# python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex pr_fl -num_clients 10 -sample_client high   -patience 7 -niid -bp
# python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex pr_fl -num_clients 10 -sample_client medium -patience 3 -niid -bp
# python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex pr_fl -num_clients 10 -sample_client low    -patience 1 -niid -bp

# ========================= Ablation =========================
# Keep the same patience as the matched `pr_fl` run.

# IID high
# python experiments/FEMNIST/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex gmr             -num_clients 10 -sample_client high   -patience 5 -bp
# python experiments/FEMNIST/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex asyn            -num_clients 10 -sample_client high   -patience 5 -bp
# python experiments/FEMNIST/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex mask_fed_avg    -num_clients 10 -sample_client high   -patience 5 -bp
# python experiments/FEMNIST/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex re_mask_fed_avg -num_clients 10 -sample_client high   -patience 5 -bp
# python experiments/FEMNIST/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex gradient_avg    -num_clients 10 -sample_client high   -patience 5 -bp
# python experiments/FEMNIST/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex re_gradient_avg -num_clients 10 -sample_client high   -patience 5 -bp

# IID medium
# python experiments/FEMNIST/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex gmr             -num_clients 10 -sample_client medium -patience 4 -bp
# python experiments/FEMNIST/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex asyn            -num_clients 10 -sample_client medium -patience 4 -bp
# python experiments/FEMNIST/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex mask_fed_avg    -num_clients 10 -sample_client medium -patience 4 -bp
# python experiments/FEMNIST/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex re_mask_fed_avg -num_clients 10 -sample_client medium -patience 4 -bp
# python experiments/FEMNIST/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex gradient_avg    -num_clients 10 -sample_client medium -patience 4 -bp
# python experiments/FEMNIST/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex re_gradient_avg -num_clients 10 -sample_client medium -patience 4 -bp

# IID low
# python experiments/FEMNIST/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex gmr             -num_clients 10 -sample_client low    -patience 3 -bp
# python experiments/FEMNIST/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex asyn            -num_clients 10 -sample_client low    -patience 3 -bp
# python experiments/FEMNIST/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex mask_fed_avg    -num_clients 10 -sample_client low    -patience 3 -bp
# python experiments/FEMNIST/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex re_mask_fed_avg -num_clients 10 -sample_client low    -patience 3 -bp
# python experiments/FEMNIST/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex gradient_avg    -num_clients 10 -sample_client low    -patience 3 -bp
# python experiments/FEMNIST/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex re_gradient_avg -num_clients 10 -sample_client low    -patience 3 -bp

# Non-IID high
# python experiments/FEMNIST/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex gmr             -num_clients 10 -sample_client high   -patience 7 -niid -bp
# python experiments/FEMNIST/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex asyn            -num_clients 10 -sample_client high   -patience 7 -niid -bp
# python experiments/FEMNIST/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex mask_fed_avg    -num_clients 10 -sample_client high   -patience 7 -niid -bp
# python experiments/FEMNIST/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex re_mask_fed_avg -num_clients 10 -sample_client high   -patience 7 -niid -bp
# python experiments/FEMNIST/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex gradient_avg    -num_clients 10 -sample_client high   -patience 7 -niid -bp
# python experiments/FEMNIST/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex re_gradient_avg -num_clients 10 -sample_client high   -patience 7 -niid -bp

# Non-IID medium
# python experiments/FEMNIST/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex gmr             -num_clients 10 -sample_client medium -patience 3 -niid -bp
# python experiments/FEMNIST/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex asyn            -num_clients 10 -sample_client medium -patience 3 -niid -bp
# python experiments/FEMNIST/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex mask_fed_avg    -num_clients 10 -sample_client medium -patience 3 -niid -bp
# python experiments/FEMNIST/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex re_mask_fed_avg -num_clients 10 -sample_client medium -patience 3 -niid -bp
# python experiments/FEMNIST/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex gradient_avg    -num_clients 10 -sample_client medium -patience 3 -niid -bp
# python experiments/FEMNIST/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex re_gradient_avg -num_clients 10 -sample_client medium -patience 3 -niid -bp

# Non-IID low
# python experiments/FEMNIST/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex gmr             -num_clients 10 -sample_client low    -patience 1 -niid -bp
# python experiments/FEMNIST/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex asyn            -num_clients 10 -sample_client low    -patience 1 -niid -bp
# python experiments/FEMNIST/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex mask_fed_avg    -num_clients 10 -sample_client low    -patience 1 -niid -bp
# python experiments/FEMNIST/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex re_mask_fed_avg -num_clients 10 -sample_client low    -patience 1 -niid -bp
# python experiments/FEMNIST/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex gradient_avg    -num_clients 10 -sample_client low    -patience 1 -niid -bp
# python experiments/FEMNIST/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex re_gradient_avg -num_clients 10 -sample_client low    -patience 1 -niid -bp

# ========================= Baselines =========================
# Canonical baseline patience:
# - fed_avg / fed_asyn / heterofl / fjord / fiarse: 5
# - fedrolex: 10

# IID high
# python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex fed_avg   -num_clients 10 -sample_client high   -patience 5  -bp
# python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex fed_asyn  -num_clients 10 -sample_client high   -patience 5  -bp
# python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex heterofl  -num_clients 10 -sample_client high   -patience 5  -bp
# python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex fedrolex  -num_clients 10 -sample_client high   -patience 10 -bp
# python experiments/FEMNIST/fjord.py                 -i 25             -ex fjord      -num_clients 10 -sample_client high   -patience 5
# python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex fiarse    -num_clients 10 -sample_client high   -patience 5  -bp

# IID medium
# python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex fed_avg   -num_clients 10 -sample_client medium -patience 5  -bp
# python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex fed_asyn  -num_clients 10 -sample_client medium -patience 5  -bp
# python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex heterofl  -num_clients 10 -sample_client medium -patience 5  -bp
# python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex fedrolex  -num_clients 10 -sample_client medium -patience 10 -bp
# python experiments/FEMNIST/fjord.py                 -i 25             -ex fjord      -num_clients 10 -sample_client medium -patience 5
# python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex fiarse    -num_clients 10 -sample_client medium -patience 5  -bp

# IID low
# python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex fed_avg   -num_clients 10 -sample_client low    -patience 5  -bp
# python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex fed_asyn  -num_clients 10 -sample_client low    -patience 5  -bp
# python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex heterofl  -num_clients 10 -sample_client low    -patience 5  -bp
# python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex fedrolex  -num_clients 10 -sample_client low    -patience 10 -bp
# python experiments/FEMNIST/fjord.py                 -i 25             -ex fjord      -num_clients 10 -sample_client low    -patience 5
# python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex fiarse    -num_clients 10 -sample_client low    -patience 5  -bp

# Non-IID high
# python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex fed_avg   -num_clients 10 -sample_client high   -patience 5  -niid -bp
# python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex fed_asyn  -num_clients 10 -sample_client high   -patience 5  -niid -bp
# python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex heterofl  -num_clients 10 -sample_client high   -patience 5  -niid -bp
# python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex fedrolex  -num_clients 10 -sample_client high   -patience 10 -niid -bp
# python experiments/FEMNIST/fjord.py                 -i 25             -ex fjord      -num_clients 10 -sample_client high   -patience 5 -niid
# python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex fiarse    -num_clients 10 -sample_client high   -patience 5  -niid -bp

# Non-IID medium
# python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex fed_avg   -num_clients 10 -sample_client medium -patience 5  -niid -bp
# python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex fed_asyn  -num_clients 10 -sample_client medium -patience 5  -niid -bp
# python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex heterofl  -num_clients 10 -sample_client medium -patience 5  -niid -bp
# python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex fedrolex  -num_clients 10 -sample_client medium -patience 10 -niid -bp
# python experiments/FEMNIST/fjord.py                 -i 25             -ex fjord      -num_clients 10 -sample_client medium -patience 5 -niid
# python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex fiarse    -num_clients 10 -sample_client medium -patience 5  -niid -bp

# Non-IID low
# python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex fed_avg   -num_clients 10 -sample_client low    -patience 5  -niid -bp
# python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex fed_asyn  -num_clients 10 -sample_client low    -patience 5  -niid -bp
# python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex heterofl  -num_clients 10 -sample_client low    -patience 5  -niid -bp
# python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex fedrolex  -num_clients 10 -sample_client low    -patience 10 -niid -bp
# python experiments/FEMNIST/fjord.py                 -i 25             -ex fjord      -num_clients 10 -sample_client low    -patience 5 -niid
# python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex fiarse    -num_clients 10 -sample_client low    -patience 5  -niid -bp
