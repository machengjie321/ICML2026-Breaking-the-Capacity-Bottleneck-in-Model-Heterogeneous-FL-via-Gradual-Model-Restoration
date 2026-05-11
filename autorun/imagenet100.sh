#nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ex fed_avg -ac wg -num_clients 10 -sample_client medium -patience 10 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
#nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ex fed_asyn -ac wg -num_clients 10 -sample_client medium -patience 10 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
#nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ex heterofl -ac wg -num_clients 10 -sample_client medium -patience 10 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
#nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ex fed_avg -ac wg -num_clients 10 -sample_client medium -patience 10 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
#nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ex fed_asyn -ac wg -num_clients 10 -sample_client medium -patience 10 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
nohup python experiments/ImageNet100/Split_model.py -i 50 -ex heterofl -ac wg -num_clients 10 -sample_client medium -patience 10 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
#
python experiments/ImageNet100/Split_model.py -i 50 -ex heterofl -ac wg -num_clients 10 -sample_client medium -patience 10
#
#nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ex gmr -ac wg -num_clients 10 -sample_client medium -patience 10 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
#nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ex gmr -ac wg -num_clients 10 -sample_client high -patience 10 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
#nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ex gmr -ac wg -num_clients 10 -sample_client low -patience 10 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
#nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ex gmr -ac wg -num_clients 10 -sample_client medium -patience 10 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
#nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ex gmr -ac wg -num_clients 10 -sample_client high -patience 10 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
#nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ex gmr -ac wg -num_clients 10 -sample_client low -patience 10 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &

#nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ex pr_fl -ac wg -num_clients 10 -sample_client medium -patience 3 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
#nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ex pr_fl -ac wg -num_clients 10 -sample_client medium -patience 5 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
#nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ex pr_fl -ac wg -num_clients 10 -sample_client medium -patience 10 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
#nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ex pr_fl -ac wg -num_clients 10 -sample_client medium -patience 7 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &

#nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ex pr_fl -ac wg -num_clients 10 -sample_client low -patience 5 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
#nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ex pr_fl -ac wg -num_clients 10 -sample_client high -patience 5 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
#nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ex pr_fl -ac wg -num_clients 10 -sample_client medium -patience 5 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
#
#nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ex fed_avg -ac wg -num_clients 10 -sample_client low -patience 10 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
#nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ex fed_asyn -ac wg -num_clients 10 -sample_client low -patience 10 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
#nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ex heterofl -ac wg -num_clients 10 -sample_client low -patience 10 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
#nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ex fed_avg -ac wg -num_clients 10 -sample_client high -patience 10 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
#nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ex fed_asyn -ac wg -num_clients 10 -sample_client high -patience 10 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
#nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ex heterofl -ac wg -num_clients 10 -sample_client high -patience 10 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
#
#nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ex pr_fl -ac wg -num_clients 10 -sample_client medium -patience 1 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
#nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ex pr_fl -ac wg -num_clients 10 -sample_client medium -patience 15 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
#
#nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ex pr_fl -ac wg -num_clients 10 -sample_client medium -patience 20 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&

#nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ex asyn -ac wg -num_clients 10 -sample_client low -patience 5 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ex buff -ac wg -num_clients 10 -sample_client low -patience 5 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ex mask_fed_avg -ac wg -num_clients 10 -sample_client low -patience 5 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ex asyn -ac wg -num_clients 10 -sample_client medium -patience 5 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ex buff -ac wg -num_clients 10 -sample_client medium -patience 5 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ex mask_fed_avg -ac wg -num_clients 10 -sample_client medium -patience 5 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ex asyn -ac wg -num_clients 10 -sample_client medium -patience 5 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ex buff -ac wg -num_clients 10 -sample_client medium -patience 5 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ex mask_fed_avg -ac wg -num_clients 10 -sample_client medium -patience 5 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
#nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ex asyn -ac wg -num_clients 10 -sample_client high -patience 5 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
#nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ex buff -ac wg -num_clients 10 -sample_client high -patience 5 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
#nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ex mask_fed_avg -ac wg -num_clients 10 -sample_client high -patience 5 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ex heterofl -ac wg -num_clients 10 -sample_client low -patience 10 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ex heterofl -ac wg -num_clients 10 -sample_client high  -patience 10 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ex heterofl -ac wg -num_clients 10 -sample_client medium -patience 10 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&

nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ex heterofl -ac wg -num_clients 10 -sample_client low -patience 10 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ex heterofl -ac wg -num_clients 10 -sample_client high  -patience 10 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ex heterofl -ac wg -num_clients 10 -sample_client medium -patience 10 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&


# ========================= Time-trigger GMR (ImageNet100) =========================
# total virtual time: 250000s
# nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ex pr_fl -num_clients 10 -sample_client high -patience 7 -niid -bp --recover_trigger_mode time --recover_time_total 250000 --recover_time_points 0.2,0.4,0.6,0.8,1.0 --recover_time_ladder 0.05,0.1,0.2,0.5,1.0 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
# nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ex gmr_fiarse -num_clients 10 -sample_client high -patience 7 -niid -bp --recover_trigger_mode time --recover_time_total 250000 --recover_time_points 0.2,0.4,0.6,0.8,1.0 --recover_time_ladder 0.05,0.1,0.2,0.5,1.0 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
