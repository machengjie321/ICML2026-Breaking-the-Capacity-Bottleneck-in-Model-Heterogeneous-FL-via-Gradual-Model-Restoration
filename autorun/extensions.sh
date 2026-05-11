#!/usr/bin/env bash

# Additional command groups:
# 1. fixed-time restoration
# 2. applying GMR to other MHFL methods
#
# Remove the leading `#` from the command you want to run.

# ========================= Fixed-time restoration =========================

# FEMNIST: total virtual time = 70000s
# python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex FedGMR     -num_clients 10 -sample_client high -patience 7 -niid -bp --recover_trigger_mode time --recover_time_total 70000  --recover_time_points 0.2,0.4,0.6,0.8,1.0 --recover_time_ladder 0.05,0.1,0.2,0.5,1.0
# python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex gmr_fiarse -num_clients 10 -sample_client high -patience 7 -niid -bp --recover_trigger_mode time --recover_time_total 70000  --recover_time_points 0.2,0.4,0.6,0.8,1.0 --recover_time_ladder 0.05,0.1,0.2,0.5,1.0

# CIFAR10: total virtual time = 220000s
# python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 25 -ex FedGMR     -num_clients 10 -sample_client high -patience 15 -niid -bp --recover_trigger_mode time --recover_time_total 220000 --recover_time_points 0.2,0.4,0.6,0.8,1.0 --recover_time_ladder 0.05,0.1,0.2,0.5,1.0
# python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 25 -ex gmr_fiarse -num_clients 10 -sample_client high -patience 15 -niid -bp --recover_trigger_mode time --recover_time_total 220000 --recover_time_points 0.2,0.4,0.6,0.8,1.0 --recover_time_ladder 0.05,0.1,0.2,0.5,1.0

# ImageNet100: total virtual time = 250000s
# python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ex FedGMR     -num_clients 10 -sample_client high -patience 25 -niid -bp --recover_trigger_mode time --recover_time_total 250000 --recover_time_points 0.2,0.4,0.6,0.8,1.0 --recover_time_ladder 0.05,0.1,0.2,0.5,1.0
# python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ex gmr_fiarse -num_clients 10 -sample_client high -patience 25 -niid -bp --recover_trigger_mode time --recover_time_total 250000 --recover_time_points 0.2,0.4,0.6,0.8,1.0 --recover_time_ladder 0.05,0.1,0.2,0.5,1.0

# StackOverflow: total virtual time = 70000s
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ex FedGMR     -num_clients 10 -sample_client high -patience 10 -niid -bp --recover_trigger_mode time --recover_time_total 70000  --recover_time_points 0.2,0.4,0.6,0.8,1.0 --recover_time_ladder 0.05,0.1,0.2,0.5,1.0
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ex gmr_fiarse -num_clients 10 -sample_client high -patience 10 -niid -bp --recover_trigger_mode time --recover_time_total 70000  --recover_time_points 0.2,0.4,0.6,0.8,1.0 --recover_time_ladder 0.05,0.1,0.2,0.5,1.0

# ========================= Apply GMR to other MHFL methods =========================

# ---- CIFAR10 ----
# HeteroFL / FedRolex
# python experiments/CIFAR10/Syn_modelhetero.py -i 25 -ex gmr_heterofl    -num_clients 10 -sample_client high -patience 15 -niid -bp
# python experiments/CIFAR10/Syn_modelhetero.py -i 25 -ex abgmr_heterofl  -num_clients 10 -sample_client high -patience 15 -niid -bp
# python experiments/CIFAR10/Syn_modelhetero.py -i 25 -ex gmr_fedrolex    -num_clients 10 -sample_client high -patience 15 -niid -bp
# python experiments/CIFAR10/Syn_modelhetero.py -i 25 -ex abgmr_fedrolex  -num_clients 10 -sample_client high -patience 15 -niid -bp
# FIARSE
# python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 25 -ex gmr_fiarse    -num_clients 10 -sample_client high -patience 15 -niid -bp
# python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 25 -ex abgmr_fiarse  -num_clients 10 -sample_client high -patience 15 -niid -bp
# Fjord
# python experiments/CIFAR10/fjord.py -i 25 -ex gmr_fjord   -num_clients 10 -sample_client high -patience 15 -niid
# python experiments/CIFAR10/fjord.py -i 25 -ex abgmr_fjord -num_clients 10 -sample_client high -patience 15 -niid

# ---- FEMNIST ----
# python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex gmr_heterofl    -num_clients 10 -sample_client high -patience 7 -niid -bp
# python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex abgmr_heterofl  -num_clients 10 -sample_client high -patience 7 -niid -bp
# python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex gmr_fedrolex    -num_clients 10 -sample_client high -patience 7 -niid -bp
# python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex abgmr_fedrolex  -num_clients 10 -sample_client high -patience 7 -niid -bp
# python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex gmr_fiarse      -num_clients 10 -sample_client high -patience 7 -niid -bp
# python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex abgmr_fiarse    -num_clients 10 -sample_client high -patience 7 -niid -bp
# python experiments/FEMNIST/fjord.py                 -i 25 -ex gmr_fjord       -num_clients 10 -sample_client high -patience 7 -niid
# python experiments/FEMNIST/fjord.py                 -i 25 -ex abgmr_fjord     -num_clients 10 -sample_client high -patience 7 -niid

# ---- ImageNet100 ----
# python experiments/ImageNet100/Syn_modelhetero.py -i 50 -ex gmr_heterofl    -num_clients 10 -sample_client high -patience 25 -niid -bp
# python experiments/ImageNet100/Syn_modelhetero.py -i 50 -ex abgmr_heterofl  -num_clients 10 -sample_client high -patience 25 -niid -bp
# python experiments/ImageNet100/Syn_modelhetero.py -i 50 -ex gmr_fedrolex    -num_clients 10 -sample_client high -patience 25 -niid -bp
# python experiments/ImageNet100/Syn_modelhetero.py -i 50 -ex abgmr_fedrolex  -num_clients 10 -sample_client high -patience 25 -niid -bp
# python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ex gmr_fiarse    -num_clients 10 -sample_client high -patience 25 -niid -bp
# python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ex abgmr_fiarse  -num_clients 10 -sample_client high -patience 25 -niid -bp
# python experiments/ImageNet100/fjord.py -i 50 -ex gmr_fjord   -num_clients 10 -sample_client high -patience 25 -niid
# python experiments/ImageNet100/fjord.py -i 50 -ex abgmr_fjord -num_clients 10 -sample_client high -patience 25 -niid

# ---- StackOverflow ----
# python experiments/stackoverflow/Syn_modelhetero.py -i 25 -ex gmr_heterofl    -num_clients 10 -sample_client high -patience 10 -niid -bp
# python experiments/stackoverflow/Syn_modelhetero.py -i 25 -ex abgmr_heterofl  -num_clients 10 -sample_client high -patience 10 -niid -bp
# python experiments/stackoverflow/Syn_modelhetero.py -i 25 -ex gmr_fedrolex    -num_clients 10 -sample_client high -patience 10 -niid -bp
# python experiments/stackoverflow/Syn_modelhetero.py -i 25 -ex abgmr_fedrolex  -num_clients 10 -sample_client high -patience 10 -niid -bp
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ex gmr_fiarse    -num_clients 10 -sample_client high -patience 10 -niid -bp
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ex abgmr_fiarse  -num_clients 10 -sample_client high -patience 10 -niid -bp
# python experiments/stackoverflow/fjord.py -i 25 -ex gmr_fjord   -num_clients 10 -sample_client high -patience 10 -niid
# python experiments/stackoverflow/fjord.py -i 25 -ex abgmr_fjord -num_clients 10 -sample_client high -patience 10 -niid
