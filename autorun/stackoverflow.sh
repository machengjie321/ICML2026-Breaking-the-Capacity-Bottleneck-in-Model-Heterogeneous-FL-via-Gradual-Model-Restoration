#!/usr/bin/env bash


# ========================= Time-trigger GMR (StackOverflow) =========================
# total virtual time: 70000s
# nohup python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ex pr_fl -num_clients 10 -sample_client high -patience 7 -niid -bp --recover_trigger_mode time --recover_time_total 70000 --recover_time_points 0.2,0.4,0.6,0.8,1.0 --recover_time_ladder 0.05,0.1,0.2,0.5,1.0 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
# nohup python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ex gmr_fiarse -num_clients 10 -sample_client high -patience 7 -niid -bp --recover_trigger_mode time --recover_time_total 70000 --recover_time_points 0.2,0.4,0.6,0.8,1.0 --recover_time_ladder 0.05,0.1,0.2,0.5,1.0 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
