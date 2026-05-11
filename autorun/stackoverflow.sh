#!/usr/bin/env bash

# StackOverflow command catalog.
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
# IID:   high=15, medium=20, low=8
# Non-IID: high=10, medium=10, low=20

# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex FedGMR -num_clients 10 -sample_client high   -patience 15 -bp
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex FedGMR -num_clients 10 -sample_client medium -patience 20 -bp
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex FedGMR -num_clients 10 -sample_client low    -patience 8  -bp
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex FedGMR -num_clients 10 -sample_client high   -patience 10 -niid -bp
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex FedGMR -num_clients 10 -sample_client medium -patience 10 -niid -bp
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex FedGMR -num_clients 10 -sample_client low    -patience 20 -niid -bp

# ========================= Ablation =========================
# Keep the same patience as the matched `FedGMR` run.
# `gradient_avg`, `re_gradient_avg`, and `re_mask_fed_avg` use `Ablation_Prune_increase_FL_CMD2.py`.

# IID high
# python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py  -i 25 -ic 2.0 -ex wo_gmr             -num_clients 10 -sample_client high   -patience 15 -bp
# python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py  -i 25 -ic 2.0 -ex wo_asyn            -num_clients 10 -sample_client high   -patience 15 -bp
# python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py  -i 25 -ic 2.0 -ex mask_fed_avg    -num_clients 10 -sample_client high   -patience 15 -bp
# python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD2.py -i 25 -ic 2.0 -ex re_mask_fed_avg -num_clients 10 -sample_client high   -patience 15 -bp -re
# python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD2.py -i 25 -ic 2.0 -ex gradient_avg    -num_clients 10 -sample_client high   -patience 15 -bp
# python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD2.py -i 25 -ic 2.0 -ex re_gradient_avg -num_clients 10 -sample_client high   -patience 15 -bp -re

# IID medium
# python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py  -i 25 -ic 2.0 -ex wo_gmr             -num_clients 10 -sample_client medium -patience 20 -bp
# python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py  -i 25 -ic 2.0 -ex wo_asyn            -num_clients 10 -sample_client medium -patience 20 -bp
# python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py  -i 25 -ic 2.0 -ex mask_fed_avg    -num_clients 10 -sample_client medium -patience 20 -bp
# python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD2.py -i 25 -ic 2.0 -ex re_mask_fed_avg -num_clients 10 -sample_client medium -patience 20 -bp -re
# python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD2.py -i 25 -ic 2.0 -ex gradient_avg    -num_clients 10 -sample_client medium -patience 20 -bp
# python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD2.py -i 25 -ic 2.0 -ex re_gradient_avg -num_clients 10 -sample_client medium -patience 20 -bp -re

# IID low
# python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py  -i 25 -ic 2.0 -ex wo_gmr             -num_clients 10 -sample_client low    -patience 8 -bp
# python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py  -i 25 -ic 2.0 -ex wo_asyn            -num_clients 10 -sample_client low    -patience 8 -bp
# python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py  -i 25 -ic 2.0 -ex mask_fed_avg    -num_clients 10 -sample_client low    -patience 8 -bp
# python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD2.py -i 25 -ic 2.0 -ex re_mask_fed_avg -num_clients 10 -sample_client low    -patience 8 -bp -re
# python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD2.py -i 25 -ic 2.0 -ex gradient_avg    -num_clients 10 -sample_client low    -patience 8 -bp
# python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD2.py -i 25 -ic 2.0 -ex re_gradient_avg -num_clients 10 -sample_client low    -patience 8 -bp -re

# Non-IID high
# python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py  -i 25 -ic 2.0 -ex wo_gmr             -num_clients 10 -sample_client high   -patience 10 -niid -bp
# python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py  -i 25 -ic 2.0 -ex wo_asyn            -num_clients 10 -sample_client high   -patience 10 -niid -bp
# python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py  -i 25 -ic 2.0 -ex mask_fed_avg    -num_clients 10 -sample_client high   -patience 10 -niid -bp
# python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD2.py -i 25 -ic 2.0 -ex re_mask_fed_avg -num_clients 10 -sample_client high   -patience 10 -niid -bp -re
# python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD2.py -i 25 -ic 2.0 -ex gradient_avg    -num_clients 10 -sample_client high   -patience 10 -niid -bp
# python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD2.py -i 25 -ic 2.0 -ex re_gradient_avg -num_clients 10 -sample_client high   -patience 10 -niid -bp -re

# Non-IID medium
# python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py  -i 25 -ic 2.0 -ex wo_gmr             -num_clients 10 -sample_client medium -patience 10 -niid -bp
# python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py  -i 25 -ic 2.0 -ex wo_asyn            -num_clients 10 -sample_client medium -patience 10 -niid -bp
# python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py  -i 25 -ic 2.0 -ex mask_fed_avg    -num_clients 10 -sample_client medium -patience 10 -niid -bp
# python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD2.py -i 25 -ic 2.0 -ex re_mask_fed_avg -num_clients 10 -sample_client medium -patience 10 -niid -bp -re
# python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD2.py -i 25 -ic 2.0 -ex gradient_avg    -num_clients 10 -sample_client medium -patience 10 -niid -bp
# python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD2.py -i 25 -ic 2.0 -ex re_gradient_avg -num_clients 10 -sample_client medium -patience 10 -niid -bp -re

# Non-IID low
# python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py  -i 25 -ic 2.0 -ex wo_gmr             -num_clients 10 -sample_client low    -patience 20 -niid -bp
# python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py  -i 25 -ic 2.0 -ex wo_asyn            -num_clients 10 -sample_client low    -patience 20 -niid -bp
# python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py  -i 25 -ic 2.0 -ex mask_fed_avg    -num_clients 10 -sample_client low    -patience 20 -niid -bp
# python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD2.py -i 25 -ic 2.0 -ex re_mask_fed_avg -num_clients 10 -sample_client low    -patience 20 -niid -bp -re
# python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD2.py -i 25 -ic 2.0 -ex gradient_avg    -num_clients 10 -sample_client low    -patience 20 -niid -bp
# python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD2.py -i 25 -ic 2.0 -ex re_gradient_avg -num_clients 10 -sample_client low    -patience 20 -niid -bp -re

# ========================= Baselines =========================
# Baseline patience is not tuned here. Canonical value: 10.

# IID high
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex fed_avg   -num_clients 10 -sample_client high   -patience 10 -bp
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex fed_asyn  -num_clients 10 -sample_client high   -patience 10 -bp
# python experiments/stackoverflow/Syn_modelhetero.py       -i 25             -ex heterofl  -num_clients 10 -sample_client high   -patience 10 -bp
# python experiments/stackoverflow/Syn_modelhetero.py       -i 25             -ex fedrolex  -num_clients 10 -sample_client high   -patience 10 -bp
# python experiments/stackoverflow/fjord.py                 -i 25             -ex fjord      -num_clients 10 -sample_client high   -patience 10
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex fiarse    -num_clients 10 -sample_client high   -patience 10 -bp

# IID medium
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex fed_avg   -num_clients 10 -sample_client medium -patience 10 -bp
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex fed_asyn  -num_clients 10 -sample_client medium -patience 10 -bp
# python experiments/stackoverflow/Syn_modelhetero.py       -i 25             -ex heterofl  -num_clients 10 -sample_client medium -patience 10 -bp
# python experiments/stackoverflow/Syn_modelhetero.py       -i 25             -ex fedrolex  -num_clients 10 -sample_client medium -patience 10 -bp
# python experiments/stackoverflow/fjord.py                 -i 25             -ex fjord      -num_clients 10 -sample_client medium -patience 10
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex fiarse    -num_clients 10 -sample_client medium -patience 10 -bp

# IID low
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex fed_avg   -num_clients 10 -sample_client low    -patience 10 -bp
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex fed_asyn  -num_clients 10 -sample_client low    -patience 10 -bp
# python experiments/stackoverflow/Syn_modelhetero.py       -i 25             -ex heterofl  -num_clients 10 -sample_client low    -patience 10 -bp
# python experiments/stackoverflow/Syn_modelhetero.py       -i 25             -ex fedrolex  -num_clients 10 -sample_client low    -patience 10 -bp
# python experiments/stackoverflow/fjord.py                 -i 25             -ex fjord      -num_clients 10 -sample_client low    -patience 10
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex fiarse    -num_clients 10 -sample_client low    -patience 10 -bp

# Non-IID high
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex fed_avg   -num_clients 10 -sample_client high   -patience 10 -niid -bp
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex fed_asyn  -num_clients 10 -sample_client high   -patience 10 -niid -bp
# python experiments/stackoverflow/Syn_modelhetero.py       -i 25             -ex heterofl  -num_clients 10 -sample_client high   -patience 10 -niid -bp
# python experiments/stackoverflow/Syn_modelhetero.py       -i 25             -ex fedrolex  -num_clients 10 -sample_client high   -patience 10 -niid -bp
# python experiments/stackoverflow/fjord.py                 -i 25             -ex fjord      -num_clients 10 -sample_client high   -patience 10 -niid
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex fiarse    -num_clients 10 -sample_client high   -patience 10 -niid -bp

# Non-IID medium
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex fed_avg   -num_clients 10 -sample_client medium -patience 10 -niid -bp
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex fed_asyn  -num_clients 10 -sample_client medium -patience 10 -niid -bp
# python experiments/stackoverflow/Syn_modelhetero.py       -i 25             -ex heterofl  -num_clients 10 -sample_client medium -patience 10 -niid -bp
# python experiments/stackoverflow/Syn_modelhetero.py       -i 25             -ex fedrolex  -num_clients 10 -sample_client medium -patience 10 -niid -bp
# python experiments/stackoverflow/fjord.py                 -i 25             -ex fjord      -num_clients 10 -sample_client medium -patience 10 -niid
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex fiarse    -num_clients 10 -sample_client medium -patience 10 -niid -bp

# Non-IID low
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex fed_avg   -num_clients 10 -sample_client low    -patience 10 -niid -bp
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex fed_asyn  -num_clients 10 -sample_client low    -patience 10 -niid -bp
# python experiments/stackoverflow/Syn_modelhetero.py       -i 25             -ex heterofl  -num_clients 10 -sample_client low    -patience 10 -niid -bp
# python experiments/stackoverflow/Syn_modelhetero.py       -i 25             -ex fedrolex  -num_clients 10 -sample_client low    -patience 10 -niid -bp
# python experiments/stackoverflow/fjord.py                 -i 25             -ex fjord      -num_clients 10 -sample_client low    -patience 10 -niid
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex fiarse    -num_clients 10 -sample_client low    -patience 10 -niid -bp
