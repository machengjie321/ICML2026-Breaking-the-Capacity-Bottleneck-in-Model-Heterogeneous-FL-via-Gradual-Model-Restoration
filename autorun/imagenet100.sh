#!/usr/bin/env bash

# ImageNet100 command catalog.
# Commands are intentionally commented out. Remove the leading `#` to run.
#
# Method notes:
# - `FedGMR`: full FedGMR setting used in the main comparison.
# - `wo_gmr`: remove GMR while keeping the rest of FedGMR unchanged.
# - `wo_asyn`: keep GMR but replace asynchronous scheduling with synchronous scheduling.
# - `mask_fed_avg`: replace mask-aware aggregation with FedAvg-style aggregation, without recovery.
# - `re_mask_fed_avg`: FedAvg-style aggregation, with recovery enabled.
# - `gradient_avg`: replace aggregation with gradient averaging, without recovery.
# - `re_gradient_avg`: gradient averaging, with recovery enabled.

# ========================= FedGMR =========================
# IID:   high=40, medium=30, low=40
# Non-IID: high=25, medium=30, low=30

# python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ic 2.0 -ex FedGMR -num_clients 10 -sample_client high   -patience 40 -bp
# python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ic 2.0 -ex FedGMR -num_clients 10 -sample_client medium -patience 30 -bp
# python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ic 2.0 -ex FedGMR -num_clients 10 -sample_client low    -patience 40 -bp
# python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ic 2.0 -ex FedGMR -num_clients 10 -sample_client high   -patience 25 -niid -bp
# python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ic 2.0 -ex FedGMR -num_clients 10 -sample_client medium -patience 30 -niid -bp
# python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ic 2.0 -ex FedGMR -num_clients 10 -sample_client low    -patience 30 -niid -bp

# ========================= Ablation =========================
# Keep the same patience as the matched `FedGMR` run.

# IID high
# python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0 -ex wo_gmr             -num_clients 10 -sample_client high   -patience 40 -bp
# python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0 -ex wo_asyn            -num_clients 10 -sample_client high   -patience 40 -bp
# python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0 -ex mask_fed_avg    -num_clients 10 -sample_client high   -patience 40 -bp
# python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0 -ex re_mask_fed_avg -num_clients 10 -sample_client high   -patience 40 -bp
# python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0 -ex gradient_avg    -num_clients 10 -sample_client high   -patience 40 -bp
# python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0 -ex re_gradient_avg -num_clients 10 -sample_client high   -patience 40 -bp

# IID medium
# python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0 -ex wo_gmr             -num_clients 10 -sample_client medium -patience 30 -bp
# python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0 -ex wo_asyn            -num_clients 10 -sample_client medium -patience 30 -bp
# python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0 -ex mask_fed_avg    -num_clients 10 -sample_client medium -patience 30 -bp
# python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0 -ex re_mask_fed_avg -num_clients 10 -sample_client medium -patience 30 -bp
# python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0 -ex gradient_avg    -num_clients 10 -sample_client medium -patience 30 -bp
# python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0 -ex re_gradient_avg -num_clients 10 -sample_client medium -patience 30 -bp

# IID low
# python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0 -ex wo_gmr             -num_clients 10 -sample_client low    -patience 40 -bp
# python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0 -ex wo_asyn            -num_clients 10 -sample_client low    -patience 40 -bp
# python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0 -ex mask_fed_avg    -num_clients 10 -sample_client low    -patience 40 -bp
# python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0 -ex re_mask_fed_avg -num_clients 10 -sample_client low    -patience 40 -bp
# python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0 -ex gradient_avg    -num_clients 10 -sample_client low    -patience 40 -bp
# python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0 -ex re_gradient_avg -num_clients 10 -sample_client low    -patience 40 -bp

# Non-IID high
# python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0 -ex wo_gmr             -num_clients 10 -sample_client high   -patience 25 -niid -bp
# python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0 -ex wo_asyn            -num_clients 10 -sample_client high   -patience 25 -niid -bp
# python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0 -ex mask_fed_avg    -num_clients 10 -sample_client high   -patience 25 -niid -bp
# python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0 -ex re_mask_fed_avg -num_clients 10 -sample_client high   -patience 25 -niid -bp
# python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0 -ex gradient_avg    -num_clients 10 -sample_client high   -patience 25 -niid -bp
# python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0 -ex re_gradient_avg -num_clients 10 -sample_client high   -patience 25 -niid -bp

# Non-IID medium
# python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0 -ex wo_gmr             -num_clients 10 -sample_client medium -patience 30 -niid -bp
# python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0 -ex wo_asyn            -num_clients 10 -sample_client medium -patience 30 -niid -bp
# python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0 -ex mask_fed_avg    -num_clients 10 -sample_client medium -patience 30 -niid -bp
# python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0 -ex re_mask_fed_avg -num_clients 10 -sample_client medium -patience 30 -niid -bp
# python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0 -ex gradient_avg    -num_clients 10 -sample_client medium -patience 30 -niid -bp
# python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0 -ex re_gradient_avg -num_clients 10 -sample_client medium -patience 30 -niid -bp

# Non-IID low
# python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0 -ex wo_gmr             -num_clients 10 -sample_client low    -patience 30 -niid -bp
# python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0 -ex wo_asyn            -num_clients 10 -sample_client low    -patience 30 -niid -bp
# python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0 -ex mask_fed_avg    -num_clients 10 -sample_client low    -patience 30 -niid -bp
# python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0 -ex re_mask_fed_avg -num_clients 10 -sample_client low    -patience 30 -niid -bp
# python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0 -ex gradient_avg    -num_clients 10 -sample_client low    -patience 30 -niid -bp
# python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0 -ex re_gradient_avg -num_clients 10 -sample_client low    -patience 30 -niid -bp

# ========================= Baselines =========================
# Baseline patience is not tuned here. Canonical value: 5.

# IID high
# python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ic 2.0 -ex fed_avg   -num_clients 10 -sample_client high   -patience 5 -bp
# python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ic 2.0 -ex fed_asyn  -num_clients 10 -sample_client high   -patience 5 -bp
# python experiments/ImageNet100/Syn_modelhetero.py       -i 50             -ex heterofl  -num_clients 10 -sample_client high   -patience 5 -bp
# python experiments/ImageNet100/Syn_modelhetero.py       -i 50             -ex fedrolex  -num_clients 10 -sample_client high   -patience 5 -bp
# python experiments/ImageNet100/fjord.py                 -i 50             -ex fjord      -num_clients 10 -sample_client high   -patience 5
# python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ic 2.0 -ex fiarse    -num_clients 10 -sample_client high   -patience 5 -bp

# IID medium
# python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ic 2.0 -ex fed_avg   -num_clients 10 -sample_client medium -patience 5 -bp
# python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ic 2.0 -ex fed_asyn  -num_clients 10 -sample_client medium -patience 5 -bp
# python experiments/ImageNet100/Syn_modelhetero.py       -i 50             -ex heterofl  -num_clients 10 -sample_client medium -patience 5 -bp
# python experiments/ImageNet100/Syn_modelhetero.py       -i 50             -ex fedrolex  -num_clients 10 -sample_client medium -patience 5 -bp
# python experiments/ImageNet100/fjord.py                 -i 50             -ex fjord      -num_clients 10 -sample_client medium -patience 5
# python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ic 2.0 -ex fiarse    -num_clients 10 -sample_client medium -patience 5 -bp

# IID low
# python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ic 2.0 -ex fed_avg   -num_clients 10 -sample_client low    -patience 5 -bp
# python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ic 2.0 -ex fed_asyn  -num_clients 10 -sample_client low    -patience 5 -bp
# python experiments/ImageNet100/Syn_modelhetero.py       -i 50             -ex heterofl  -num_clients 10 -sample_client low    -patience 5 -bp
# python experiments/ImageNet100/Syn_modelhetero.py       -i 50             -ex fedrolex  -num_clients 10 -sample_client low    -patience 5 -bp
# python experiments/ImageNet100/fjord.py                 -i 50             -ex fjord      -num_clients 10 -sample_client low    -patience 5
# python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ic 2.0 -ex fiarse    -num_clients 10 -sample_client low    -patience 5 -bp

# Non-IID high
# python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ic 2.0 -ex fed_avg   -num_clients 10 -sample_client high   -patience 5 -niid -bp
# python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ic 2.0 -ex fed_asyn  -num_clients 10 -sample_client high   -patience 5 -niid -bp
# python experiments/ImageNet100/Syn_modelhetero.py       -i 50             -ex heterofl  -num_clients 10 -sample_client high   -patience 5 -niid -bp
# python experiments/ImageNet100/Syn_modelhetero.py       -i 50             -ex fedrolex  -num_clients 10 -sample_client high   -patience 5 -niid -bp
# python experiments/ImageNet100/fjord.py                 -i 50             -ex fjord      -num_clients 10 -sample_client high   -patience 5 -niid
# python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ic 2.0 -ex fiarse    -num_clients 10 -sample_client high   -patience 5 -niid -bp

# Non-IID medium
# python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ic 2.0 -ex fed_avg   -num_clients 10 -sample_client medium -patience 5 -niid -bp
# python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ic 2.0 -ex fed_asyn  -num_clients 10 -sample_client medium -patience 5 -niid -bp
# python experiments/ImageNet100/Syn_modelhetero.py       -i 50             -ex heterofl  -num_clients 10 -sample_client medium -patience 5 -niid -bp
# python experiments/ImageNet100/Syn_modelhetero.py       -i 50             -ex fedrolex  -num_clients 10 -sample_client medium -patience 5 -niid -bp
# python experiments/ImageNet100/fjord.py                 -i 50             -ex fjord      -num_clients 10 -sample_client medium -patience 5 -niid
# python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ic 2.0 -ex fiarse    -num_clients 10 -sample_client medium -patience 5 -niid -bp

# Non-IID low
# python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ic 2.0 -ex fed_avg   -num_clients 10 -sample_client low    -patience 5 -niid -bp
# python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ic 2.0 -ex fed_asyn  -num_clients 10 -sample_client low    -patience 5 -niid -bp
# python experiments/ImageNet100/Syn_modelhetero.py       -i 50             -ex heterofl  -num_clients 10 -sample_client low    -patience 5 -niid -bp
# python experiments/ImageNet100/Syn_modelhetero.py       -i 50             -ex fedrolex  -num_clients 10 -sample_client low    -patience 5 -niid -bp
# python experiments/ImageNet100/fjord.py                 -i 50             -ex fjord      -num_clients 10 -sample_client low    -patience 5 -niid
# python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ic 2.0 -ex fiarse    -num_clients 10 -sample_client low    -patience 5 -niid -bp
