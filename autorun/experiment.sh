# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client low -patience 30 -bp
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client low -patience 40 -bp
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client low -patience 30 -bp -niid
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client low -patience 40 -bp -niid
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client medium -patience 30 -bp
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client medium -patience 40 -bp
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client medium -patience 30 -bp -niid
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client medium -patience 40 -bp -niid
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client high -patience 30 -bp -niid
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client high -patience 40 -bp -niid
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client high -patience 30 -bp
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client high -patience 40 -bp



# # nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ex gmr -num_clients 10 -sample_client low -patience 10 -niid  -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# # nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ex gmr -num_clients 10 -sample_client medium -patience 10 -niid -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# # nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ex gmr -num_clients 10 -sample_client high -patience 10 -niid  -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# # nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ex gmr -num_clients 10 -sample_client medium -patience 10 -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# # nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ex gmr -num_clients 10 -sample_client high -patience 10 -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# # nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ex gmr -num_clients 10 -sample_client low -patience 10  -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&


# # nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ex fiarse -num_clients 10 -sample_client high -patience 10  -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# # nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ex fiarse -num_clients 10 -sample_client medium -patience 10 -niid -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# # nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ex fiarse -num_clients 10 -sample_client low -patience 10  -niid  -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# # nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ex fiarse -num_clients 10 -sample_client high -patience 10 -niid  -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# # nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ex fiarse -num_clients 10 -sample_client low -patience 10  -bp   > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# # nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ex fiarse -num_clients 10 -sample_client medium -patience 10  -bp   > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&

# # nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ex pr_fl -num_clients 10 -sample_client high -patience 10 -niid  -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# # nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ex pr_fl -num_clients 10 -sample_client low -patience 10 -niid  -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# # nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ex pr_fl -num_clients 10 -sample_client medium -patience 10 -niid -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# # nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ex pr_fl -num_clients 10 -sample_client high -patience 10 -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# # nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ex pr_fl -num_clients 10 -sample_client low -patience 10 -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# # nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ex pr_fl -num_clients 10 -sample_client medium -patience 10 -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&

# # nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ex asyn -num_clients 10 -sample_client high -patience 10 -niid -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# # nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ex asyn -num_clients 10 -sample_client low -patience 10 -niid -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# # nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ex asyn -num_clients 10 -sample_client medium -patience 10 -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# # nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ex asyn -num_clients 10 -sample_client low -patience 10 -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# # nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ex asyn -num_clients 10 -sample_client high -patience 10 -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# # nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ex asyn -num_clients 10 -sample_client medium -patience 10 -niid -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# # nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ex fiarse -num_clients 10 -sample_client high -patience 10 -niid  -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&



# # python experiments/ImageNet100/fjord.py -i 50 -ex fjord  -num_clients 10 -sample_client high -patience 5 -niid
# # python experiments/ImageNet100/fjord.py -i 50 -ex fjord  -num_clients 10 -sample_client medium -patience 5 -niid 




# # nohup python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 50 -ex asyn -num_clients 10 -sample_client high -patience 5 -niid -bp -clip > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# # nohup python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 50 -ex asyn -num_clients 10 -sample_client low -patience 5 -niid -bp -clip > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# # nohup python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 50 -ex asyn -num_clients 10 -sample_client medium -patience 5 -bp -clip > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# # nohup python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 50 -ex asyn -num_clients 10 -sample_client low -patience 5 -bp  -clip > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# # nohup python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 50 -ex asyn -num_clients 10 -sample_client high -patience 5 -bp -clip   > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# # nohup python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 50 -ex asyn -num_clients 10 -sample_client medium -patience 5 -niid -bp -clip > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&

# # nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ex fed_asyn -num_clients 10 -sample_client high -patience 5 -niid -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# # nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ex fed_asyn -num_clients 10 -sample_client low -patience 5 -niid -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# # nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ex fed_asyn -num_clients 10 -sample_client medium -patience 5 -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# # nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ex fed_asyn -num_clients 10 -sample_client low -patience 5 -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&

# # nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ex fed_asyn -num_clients 10 -sample_client low -patience 5 -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# # nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ex fed_asyn -num_clients 10 -sample_client high -patience 5 -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# # nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ex fed_asyn -num_clients 10 -sample_client medium -patience 5 -niid -bp> "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&

# # nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ex pr_fl -num_clients 10 -sample_client medium -patience 5 -bp -clip  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# # nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ex pr_fl -num_clients 10 -sample_client low -patience 5 -bp  -clip > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# # nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ex pr_fl -num_clients 10 -sample_client high -patience 5 -niid  -bp -clip  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# # nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ex pr_fl -num_clients 10 -sample_client low -patience 5 -niid  -bp -clip  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# # nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ex pr_fl -num_clients 10 -sample_client medium -patience 5 -niid -bp  -clip  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# # nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ex pr_fl -num_clients 10 -sample_client high -patience 5 -bp -clip   > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&


# # nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ex pr_fl -num_clients 10 -sample_client low -patience 5 -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# # nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ex pr_fl -num_clients 10 -sample_client high -patience 5 -niid  -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# # nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ex pr_fl -num_clients 10 -sample_client low -patience 5 -niid  -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# # nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ex pr_fl -num_clients 10 -sample_client medium -patience 5 -niid -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# # nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ex pr_fl -num_clients 10 -sample_client high -patience 5 -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&

# # nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client medium -patience 5 -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# # nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client low -patience 5 -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# # nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client high -patience 5 -niid  -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# # nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client low -patience 5 -niid  -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&


# # nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client medium -patience 5 -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# # nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client low -patience 5 -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&

# # nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client high -patience 5 -niid  -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# # nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client low -patience 5 -niid  -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# # nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client medium -patience 5 -niid -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# # nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client high -patience 5 -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&

# # nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client medium -patience 5 -niid -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# # nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client high -patience 5 -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&









# # nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex asyn -num_clients 10 -sample_client medium -patience 5 -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# # nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex asyn -num_clients 10 -sample_client low -patience 5 -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# # nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex asyn -num_clients 10 -sample_client high -patience 5 -niid  -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# # nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex asyn -num_clients 10 -sample_client low -patience 5 -niid  -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# # nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex asyn -num_clients 10 -sample_client medium -patience 5 -niid -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# # nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex asyn -num_clients 10 -sample_client high -patience 5 -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&


# # nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client medium -patience 1 -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# # nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client low -patience 1 -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# # nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client high -patience 1 -niid  -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# # nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client low -patience 1 -niid  -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# # nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client medium -patience 1 -niid -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# # nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client high -patience 1 -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&


# # nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client medium -patience 10 -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# # nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client low -patience 10 -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# # nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client high -patience 10 -niid  -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# # nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client low -patience 10 -niid  -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# # nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client medium -patience 10 -niid -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# # nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client high -patience 10 -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&

# # nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client medium -patience 15 -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# # nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client low -patience 15 -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# # nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client high -patience 15 -niid  -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# # nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client low -patience 15 -niid  -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# # nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client medium -patience 15 -niid -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# # nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client high -patience 15 -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&


# # nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client medium -patience 2 -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# # nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client low -patience 2 -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# # nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client high -patience 2 -niid  -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# # nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client low -patience 2 -niid  -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# # nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client medium -patience 2 -niid -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# # nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client high -patience 2 -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&

# # nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client medium -patience 7 -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# # nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client low -patience 7 -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# # nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client high -patience 7 -niid  -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# # nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client low -patience 7 -niid  -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# # nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client medium -patience 7 -niid -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# # nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client high -patience 7 -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&



# nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex mask_fed_avg -num_clients 10 -sample_client medium -patience 5 -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex mask_fed_avg -num_clients 10 -sample_client low -patience 5 -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex mask_fed_avg -num_clients 10 -sample_client high -patience 5 -niid  -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex mask_fed_avg -num_clients 10 -sample_client low -patience 5 -niid  -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex mask_fed_avg -num_clients 10 -sample_client medium -patience 5 -niid -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex mask_fed_avg -num_clients 10 -sample_client high -patience 5 -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&

# nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex gradient_avg -num_clients 10 -sample_client medium -patience 5 -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex gradient_avg -num_clients 10 -sample_client low -patience 5 -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex gradient_avg -num_clients 10 -sample_client high -patience 5 -niid  -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# # nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex gradient_avg -num_clients 10 -sample_client low -patience 5 -niid  -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# # nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex gradient_avg -num_clients 10 -sample_client medium -patience 5 -niid -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# # nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex gradient_avg -num_clients 10 -sample_client high -patience 5 -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&

# nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex mask_fed_avg -num_clients 10 -sample_client high -patience 5 -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
# nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex gradient_avg -num_clients 10 -sample_client medium -patience 5 -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
# nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex gradient_avg -num_clients 10 -sample_client low -patience 5 -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
# nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex gradient_avg -num_clients 10 -sample_client high -patience 5 -niid  -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &




# nohup python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex gradient_avg -num_clients 10 -sample_client medium -patience 5 -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex gradient_avg -num_clients 10 -sample_client low -patience 5 -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex gradient_avg -num_clients 10 -sample_client high -patience 5 -niid  -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex gradient_avg -num_clients 10 -sample_client low -patience 5 -niid  -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex gradient_avg -num_clients 10 -sample_client medium -patience 5 -niid -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex gradient_avg -num_clients 10 -sample_client high -patience 5 -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&




# nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client medium -patience 5 -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
# nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex gradient_avg -num_clients 10 -sample_client medium -patience 5 -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
# nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex mask_fed_avg -num_clients 10 -sample_client medium -patience 5 -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
# nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex gradient_avg -num_clients 10 -sample_client medium -patience 5 -bp -re > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
# nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex mask_fed_avg -num_clients 10 -sample_client medium -patience 5 -bp  -re> "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
# nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex gmr -num_clients 10 -sample_client medium -patience 5 -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &

# nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client high -patience 5 -bp -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
# nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex gradient_avg -num_clients 10 -sample_client high -patience 5 -bp -niid  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
# nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex mask_fed_avg -num_clients 10 -sample_client high -patience 5 -bp -niid  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
# nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex gradient_avg -num_clients 10 -sample_client high -patience 5 -bp -niid -re > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
# nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex mask_fed_avg -num_clients 10 -sample_client high -patience 5 -bp -niid  -re> "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
# nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex gmr -num_clients 10 -sample_client high -patience 5 -bp -niid  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &

# # nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client low -patience 5 -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
# # nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex gradient_avg -num_clients 10 -sample_client low -patience 5 -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
# # nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex mask_fed_avg -num_clients 10 -sample_client low -patience 5 -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
# nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex gradient_avg -num_clients 10 -sample_client low -patience 5 -bp -re > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
# nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex mask_fed_avg -num_clients 10 -sample_client low -patience 5 -bp  -re> "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
# # nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex gmr -num_clients 10 -sample_client low -patience 5 -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &

# nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client low -patience 5 -bp -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
# # nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex gradient_avg -num_clients 10 -sample_client low -patience 5 -bp -niid  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
# nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex mask_fed_avg -num_clients 10 -sample_client low -patience 5 -bp -niid  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
# nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex gradient_avg -num_clients 10 -sample_client low -patience 5 -bp -niid -re > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
# nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex mask_fed_avg -num_clients 10 -sample_client low -patience 5 -bp -niid  -re> "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
# nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex gmr -num_clients 10 -sample_client low -patience 5 -bp -niid  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &

# nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client high -patience 5 -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
# nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex gradient_avg -num_clients 10 -sample_client high -patience 5 -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
# nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex mask_fed_avg -num_clients 10 -sample_client high -patience 5 -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
# nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex gradient_avg -num_clients 10 -sample_client high -patience 5 -bp -re > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
# nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex mask_fed_avg -num_clients 10 -sample_client high -patience 5 -bp  -re> "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
# nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex gmr -num_clients 10 -sample_client high -patience 5 -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &

# nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client medium -patience 5 -bp -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
# nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex gradient_avg -num_clients 10 -sample_client medium -patience 5 -bp -niid  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
# nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex mask_fed_avg -num_clients 10 -sample_client medium -patience 5 -bp -niid  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
# nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex gradient_avg -num_clients 10 -sample_client medium -patience 5 -bp -niid -re > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
# nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex mask_fed_avg -num_clients 10 -sample_client medium -patience 5 -bp -niid  -re> "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
# nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex gmr -num_clients 10 -sample_client medium -patience 5 -bp -niid  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &

# nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex gmr -num_clients 10 -sample_client medium -patience 5 -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
# nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex gradient_avg -num_clients 10 -sample_client medium -patience 5 -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &





# nohup python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex buff -num_clients 10 -sample_client medium -patience 5 -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex buff -num_clients 10 -sample_client low -patience 5 -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex buff -num_clients 10 -sample_client high -patience 5 -niid  -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex buff -num_clients 10 -sample_client low -patience 5 -niid  -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex buff -num_clients 10 -sample_client medium -patience 5 -niid -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex buff -num_clients 10 -sample_client high -patience 5 -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&

# nohup python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex ims -num_clients 10 -sample_client medium -patience 5 -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex ims -num_clients 10 -sample_client low -patience 5 -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex ims -num_clients 10 -sample_client high -patience 5 -niid  -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex ims -num_clients 10 -sample_client low -patience 5 -niid  -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex ims -num_clients 10 -sample_client medium -patience 5 -niid -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex ims -num_clients 10 -sample_client high -patience 5 -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&


# nohup python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex pure -num_clients 10 -sample_client medium -patience 5 -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex pure -num_clients 10 -sample_client low -patience 5 -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex pure -num_clients 10 -sample_client high -patience 5 -niid  -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex pure -num_clients 10 -sample_client low -patience 5 -niid  -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex pure -num_clients 10 -sample_client medium -patience 5 -niid -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex pure -num_clients 10 -sample_client high -patience 5 -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&


# nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ex buff -num_clients 10 -sample_client low -patience 5 -niid  -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ex buff -num_clients 10 -sample_client medium -patience 5 -niid -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ex buff -num_clients 10 -sample_client high -patience 5 -niid  -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ex buff -num_clients 10 -sample_client medium -patience 5 -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ex buff -num_clients 10 -sample_client high -patience 5 -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ex buff -num_clients 10 -sample_client low -patience 5  -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&

# nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ex pure -num_clients 10 -sample_client low -patience 5 -niid  -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ex pure -num_clients 10 -sample_client medium -patience 5 -niid -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ex pure -num_clients 10 -sample_client high -patience 5 -niid  -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ex pure -num_clients 10 -sample_client medium -patience 5 -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ex pure -num_clients 10 -sample_client high -patience 5 -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ex pure -num_clients 10 -sample_client low -patience 5  -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&

# nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ex ims -num_clients 10 -sample_client low -patience 5 -niid  -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ex ims -num_clients 10 -sample_client medium -patience 5 -niid -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ex ims -num_clients 10 -sample_client high -patience 5 -niid  -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ex ims -num_clients 10 -sample_client medium -patience 5 -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ex ims -num_clients 10 -sample_client high -patience 5 -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ex ims -num_clients 10 -sample_client low -patience 5  -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&



# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex fiarse -num_clients 10 -sample_client low -patience 5 -bp
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex fed_asyn -num_clients 10 -sample_client low -patience 5 -bp
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex fedavg -num_clients 10 -sample_client low -patience 5 -bp
# python experiments/stackoverflow/Syn_modelhetero.py -i 25 -ex fedrolex  -num_clients 10 -sample_client low  -patience 5 
# python experiments/stackoverflow/Syn_modelhetero.py -i 25 -ex heterofl  -num_clients 10 -sample_client low  -patience 5 
# python experiments/stackoverflow/fjord.py -i 25 -ex fjord  -num_clients 10 -sample_client low  -patience 5 

# # python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client medium -patience 5 -bp
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex fiarse -num_clients 10 -sample_client medium -patience 5 -bp
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex fed_asyn -num_clients 10 -sample_client medium -patience 5 -bp
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex fed_avg -num_clients 10 -sample_client medium -patience 5 -bp
# python experiments/stackoverflow/Syn_modelhetero.py -i 25 -ex fedrolex  -num_clients 10 -sample_client medium  -patience 5
# python experiments/stackoverflow/Syn_modelhetero.py -i 25 -ex heterofl  -num_clients 10 -sample_client medium  -patience 5 
# python experiments/stackoverflow/fjord.py -i 25 -ex fjord  -num_clients 10 -sample_client medium  -patience 5 

# nohup python ./experiments/stackoverflow/PMT.py -na > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &

# # python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client high -patience 5 -bp
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex fiarse -num_clients 10 -sample_client high -patience 5 -bp
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex fed_asyn -num_clients 10 -sample_client high -patience 5 -bp
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex fed_avg -num_clients 10 -sample_client high -patience 5 -bp
# python experiments/stackoverflow/Syn_modelhetero.py -i 25 -ex fedrolex  -num_clients 10 -sample_client high  -patience 5
# python experiments/stackoverflow/Syn_modelhetero.py -i 25 -ex heterofl  -num_clients 10 -sample_client high  -patience 5 
# python experiments/stackoverflow/fjord.py -i 25 -ex fjord  -num_clients 10 -sample_client high  -patience 5 

# # python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client low -patience 5 -bp -niid
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex fiarse -num_clients 10 -sample_client low -patience 5 -bp -niid
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex fed_asyn -num_clients 10 -sample_client low -patience 5 -bp -niid
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex fed_avg -num_clients 10 -sample_client low -patience 5 -bp -niid
# python experiments/stackoverflow/Syn_modelhetero.py -i 25 -ex fedrolex  -num_clients 10 -sample_client low  -patience 5 -niid
# python experiments/stackoverflow/Syn_modelhetero.py -i 25 -ex heterofl  -num_clients 10 -sample_client low  -patience 5 -niid 
# python experiments/stackoverflow/fjord.py -i 25 -ex fjord  -num_clients 10 -sample_client low  -patience 5 -niid

# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client medium -patience 5 -bp -niid
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex fiarse -num_clients 10 -sample_client medium -patience 5 -bp -niid
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex fed_asyn -num_clients 10 -sample_client medium -patience 5 -bp -niid
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex fed_avg -num_clients 10 -sample_client medium -patience 5 -bp -niid
# python experiments/stackoverflow/Syn_modelhetero.py -i 25 -ex fedrolex  -num_clients 10 -sample_client medium  -patience 5 -niid
# python experiments/stackoverflow/Syn_modelhetero.py -i 25 -ex heterofl  -num_clients 10 -sample_client medium  -patience 5 -niid
# python experiments/stackoverflow/fjord.py -i 25 -ex fjord  -num_clients 10 -sample_client medium  -patience 5 -niid

# # python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client high -patience 5 -bp -niid
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex fiarse -num_clients 10 -sample_client high -patience 5 -bp -niid
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex fed_asyn -num_clients 10 -sample_client high -patience 5 -bp -niid
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex fed_avg -num_clients 10 -sample_client high -patience 5 -bp -niid
# python experiments/stackoverflow/Syn_modelhetero.py -i 25 -ex fedrolex  -num_clients 10 -sample_client high  -patience 5 -niid
# python experiments/stackoverflow/Syn_modelhetero.py -i 25 -ex heterofl  -num_clients 10 -sample_client high  -patience 5 -niid
# python experiments/stackoverflow/fjord.py -i 25 -ex fjord  -num_clients 10 -sample_client high  -patience 5 -niid



#IID – low
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client low -patience 5 -bp
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex fiarse -num_clients 10 -sample_client low -patience 5 -bp
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex fed_asyn -num_clients 10 -sample_client low -patience 5 -bp
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex fed_avg -num_clients 10 -sample_client low -patience 5 -bp
# python experiments/stackoverflow/Syn_modelhetero.py -i 25 -ex fedrolex  -num_clients 10 -sample_client low  -patience 5
# python experiments/stackoverflow/Syn_modelhetero.py -i 25 -ex heterofl  -num_clients 10 -sample_client low  -patience 5
# python experiments/stackoverflow/fjord.py -i 25 -ex fjord  -num_clients 10 -sample_client low  -patience 5

# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py -i 25 -ex buff -num_clients 10 -sample_client low -patience 5  -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py -i 25 -ex pure -num_clients 10 -sample_client low -patience 5  -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py -i 25 -ex ims -num_clients 10 -sample_client low -patience 5  -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex mask_fed_avg -num_clients 10 -sample_client low -patience 5 -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex gradient_avg -num_clients 10 -sample_client low -patience 5 -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &

# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py -i 25 -ex gmr -num_clients 10 -sample_client low -patience 10  -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex asyn -num_clients 10 -sample_client low -patience 5 -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &

# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex re_mask_fed_avg -num_clients 10 -sample_client low -patience 5 -bp -re > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex re_gradient_avg -num_clients 10 -sample_client low -patience 5 -bp -re > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &

# # IID + high
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client high -patience 5 -bp
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex fiarse -num_clients 10 -sample_client high -patience 5 -bp
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex fed_asyn -num_clients 10 -sample_client high -patience 5 -bp
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex fed_avg -num_clients 10 -sample_client high -patience 5 -bp

# python experiments/stackoverflow/Syn_modelhetero.py -i 25 -ex fedrolex  -num_clients 10 -sample_client high  -patience 5
# python experiments/stackoverflow/Syn_modelhetero.py -i 25 -ex heterofl  -num_clients 10 -sample_client high  -patience 5
# python experiments/stackoverflow/fjord.py -i 25 -ex fjord  -num_clients 10 -sample_client high  -patience 5

# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py -i 25 -ex buff -num_clients 10 -sample_client high -patience 5 -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py -i 25 -ex pure -num_clients 10 -sample_client high -patience 5 -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py -i 25 -ex ims -num_clients 10 -sample_client high -patience 5 -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &

# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD2.py -i 25 -ex gmr -num_clients 10 -sample_client high -patience 10 -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD2.py -i 25 -ic 2.0  -ex asyn -num_clients 10 -sample_client high -patience 5 -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &

# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex re_mask_fed_avg -num_clients 10 -sample_client high -patience 5 -bp -re > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD2.py -i 25 -ic 2.0  -ex re_gradient_avg -num_clients 10 -sample_client high -patience 5 -bp -re  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &

# # #Non-IID low
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client low -patience 5 -bp -niid
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex fiarse -num_clients 10 -sample_client low -patience 5 -bp -niid
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex fed_asyn -num_clients 10 -sample_client low -patience 5 -bp -niid
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex fed_avg -num_clients 10 -sample_client low -patience 5 -bp -niid

# python experiments/stackoverflow/Syn_modelhetero.py -i 25 -ex fedrolex  -num_clients 10 -sample_client low  -patience 5 -niid
# python experiments/stackoverflow/Syn_modelhetero.py -i 25 -ex heterofl  -num_clients 10 -sample_client low  -patience 5 -niid
# python experiments/stackoverflow/fjord.py -i 25 -ex fjord  -num_clients 10 -sample_client low  -patience 5 -niid

# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py -i 25 -ex buff -num_clients 10 -sample_client low -patience 5 -niid  -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py -i 25 -ex pure -num_clients 10 -sample_client low -patience 5 -niid -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py -i 25 -ex ims -num_clients 10 -sample_client low -patience 5 -niid -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &

# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex mask_fed_avg -num_clients 10 -sample_client low -patience 5 -niid  -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex gradient_avg -num_clients 10 -sample_client low -patience 5 -niid  -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &

# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD2.py -i 25 -ex gmr -num_clients 10 -sample_client low -patience 10 -niid  -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD2.py -i 25 -ic 2.0  -ex asyn -num_clients 10 -sample_client low -patience 5 -niid  -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &

# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex re_mask_fed_avg -num_clients 10 -sample_client low -patience 5 -niid  -bp -re > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD2.py -i 25 -ic 2.0  -ex re_gradient_avg -num_clients 10 -sample_client low -patience 5 -niid  -bp -re > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &

# #Non-IID high
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client high -patience 5 -bp -niid
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex fiarse -num_clients 10 -sample_client high -patience 5 -bp -niid
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex fed_asyn -num_clients 10 -sample_client high -patience 5 -bp -niid
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex fed_avg -num_clients 10 -sample_client high -patience 5 -bp -niid

# python experiments/stackoverflow/Syn_modelhetero.py -i 25 -ex fedrolex  -num_clients 10 -sample_client high  -patience 5 -niid
# python experiments/stackoverflow/Syn_modelhetero.py -i 25 -ex heterofl  -num_clients 10 -sample_client high  -patience 5 -niid
# python experiments/stackoverflow/fjord.py -i 25 -ex fjord  -num_clients 10 -sample_client high  -patience 5 -niid

# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py -i 25 -ex buff -num_clients 10 -sample_client high -patience 5 -niid  -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py -i 25 -ex pure -num_clients 10 -sample_client high -patience 5 -niid -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py -i 25 -ex ims -num_clients 10 -sample_client high -patience 5 -niid -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &

# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD2.py -i 25 -ex gmr -num_clients 10 -sample_client high -patience 10 -niid  -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD2.py -i 25 -ic 2.0  -ex asyn -num_clients 10 -sample_client high -patience 5 -niid  -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &

# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex re_mask_fed_avg -num_clients 10 -sample_client high -patience 5 -niid  -bp -re > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD2.py -i 25 -ic 2.0  -ex re_gradient_avg -num_clients 10 -sample_client high -patience 5 -niid -bp -re  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &



# # Non-IID medium

# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client medium -patience 5 -bp -niid
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex fiarse -num_clients 10 -sample_client medium -patience 5 -bp -niid
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex fed_asyn -num_clients 10 -sample_client medium -patience 5 -bp -niid
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex fed_avg -num_clients 10 -sample_client medium -patience 5 -bp -niid

# python experiments/stackoverflow/Syn_modelhetero.py -i 25 -ex fedrolex  -num_clients 10 -sample_client medium  -patience 5 -niid
# python experiments/stackoverflow/Syn_modelhetero.py -i 25 -ex heterofl  -num_clients 10 -sample_client medium  -patience 5 -niid
# python experiments/stackoverflow/fjord.py -i 25 -ex fjord  -num_clients 10 -sample_client medium  -patience 5 -niid

# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py -i 25 -ex buff -num_clients 10 -sample_client medium -patience 5 -niid -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py -i 25 -ex pure -num_clients 10 -sample_client medium -patience 5 -niid -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py -i 25 -ex ims -num_clients 10 -sample_client medium -patience 5 -niid -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &

# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex mask_fed_avg -num_clients 10 -sample_client medium -patience 5 -niid -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex gradient_avg -num_clients 10 -sample_client medium -patience 5 -niid -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &

# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD2.py -i 25 -ex gmr -num_clients 10 -sample_client medium -patience 10 -niid -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD2.py -i 25 -ic 2.0  -ex asyn -num_clients 10 -sample_client medium -patience 5 -niid -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &

# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex re_mask_fed_avg -num_clients 10 -sample_client medium -patience 5 -niid -bp -re > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD2.py -i 25 -ic 2.0  -ex re_gradient_avg -num_clients 10 -sample_client medium -patience 5 -niid -bp -re  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &

# # IID medium
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client medium -patience 5 -bp
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex fiarse -num_clients 10 -sample_client medium -patience 5 -bp
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex fed_asyn -num_clients 10 -sample_client medium -patience 5 -bp
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex fed_avg -num_clients 10 -sample_client medium -patience 5 -bp

# python experiments/stackoverflow/Syn_modelhetero.py -i 25 -ex fedrolex  -num_clients 10 -sample_client medium  -patience 5
# python experiments/stackoverflow/Syn_modelhetero.py -i 25 -ex heterofl  -num_clients 10 -sample_client medium  -patience 5
# python experiments/stackoverflow/fjord.py -i 25 -ex fjord  -num_clients 10 -sample_client medium  -patience 5

# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py -i 25 -ex buff -num_clients 10 -sample_client medium -patience 5 -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py -i 25 -ex pure -num_clients 10 -sample_client medium -patience 5 -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py -i 25 -ex ims -num_clients 10 -sample_client medium -patience 5 -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &

# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex mask_fed_avg -num_clients 10 -sample_client medium -patience 5 -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex gradient_avg -num_clients 10 -sample_client medium -patience 5 -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &

# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD2.py -i 25 -ex gmr -num_clients 10 -sample_client medium -patience 10 -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD2.py -i 25 -ic 2.0  -ex asyn -num_clients 10 -sample_client medium -patience 5 -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &

# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex re_mask_fed_avg -num_clients 10 -sample_client medium -patience 5 -bp -re > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD2.py -i 25 -ic 2.0  -ex re_gradient_avg -num_clients 10 -sample_client medium -patience 5 -bp -re  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &


# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client low -patience 5 -bp
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex fiarse -num_clients 10 -sample_client low -patience 5 -bp
# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py -i 25 -ex gmr -num_clients 10 -sample_client low -patience 10  -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &

# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD2.py -i 25 -ex gmr -num_clients 10 -sample_client high -patience 10 -niid  -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client high -patience 5 -bp -niid
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex fiarse -num_clients 10 -sample_client high -patience 5 -bp -niid


# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client low -patience 10 -bp

# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client high -patience 10 -bp -niid

# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py -i 25 -ex gmr -num_clients 10 -sample_client low -patience 5  -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &

# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD2.py -i 25 -ex gmr -num_clients 10 -sample_client high -patience 5 -niid  -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &


# python experiments/FEMNIST/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client low -patience 5 -niid  -bp

# nohup python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex pure2 -num_clients 10 -sample_client medium -patience 5 -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex pure2 -num_clients 10 -sample_client low -patience 5 -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex pure2 -num_clients 10 -sample_client high -patience 5 -niid  -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex pure2 -num_clients 10 -sample_client medium -patience 5 -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&

# nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ex pure2 -num_clients 10 -sample_client low -patience 5 -niid  -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ex pure2 -num_clients 10 -sample_client medium -patience 5 -niid -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ex pure2 -num_clients 10 -sample_client high -patience 5 -niid  -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ex pure2 -num_clients 10 -sample_client medium -patience 5 -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ex pure2 -num_clients 10 -sample_client high -patience 5 -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/ImageNet100/Ablation_Prune_increase_FL_CMD.py -i 50 -ex pure2 -num_clients 10 -sample_client low -patience 5  -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&


# nohup python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex pure2 -num_clients 10 -sample_client low -patience 5 -niid  -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex pure2 -num_clients 10 -sample_client medium -patience 5 -niid -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex pure2 -num_clients 10 -sample_client high -patience 5 -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&

# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client high -patience 8 -bp -niid
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client high -patience 4 -bp -niid
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client high -patience 2 -bp -niid
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client high -patience 10 -bp -niid
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client high -patience 15 -bp -niid
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client high -patience 20 -bp -niid
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client high -patience 6 -bp -niid
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client high -patience 5 -bp -niid

# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD2.py -i 25 -ex gmr -num_clients 10 -sample_client high -patience 5 -niid  -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &



# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client medium -patience 8 -bp -niid
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client medium -patience 4 -bp -niid
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client medium -patience 2 -bp -niid
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client medium -patience 10 -bp -niid
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client medium -patience 15 -bp -niid
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client medium -patience 20 -bp -niid
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client medium -patience 6 -bp -niid
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client medium -patience 5 -bp -niid

# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD2.py -i 25 -ex gmr -num_clients 10 -sample_client medium -patience 5 -niid  -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &

# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client low -patience 8 -bp -niid
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client low -patience 4 -bp -niid
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client low -patience 2 -bp -niid
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client low -patience 10 -bp -niid
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client low -patience 15 -bp -niid
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client low -patience 20 -bp -niid
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client low -patience 6 -bp -niid
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client low -patience 5 -bp -niid

# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD2.py -i 25 -ex gmr -num_clients 10 -sample_client low -patience 5 -niid  -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &

# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client high -patience 8 -bp
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client high -patience 4 -bp
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client high -patience 2 -bp
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client high -patience 10 -bp
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client high -patience 15 -bp
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client high -patience 20 -bp
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client high -patience 6 -bp
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client high -patience 5 -bp

# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD2.py -i 25 -ex gmr -num_clients 10 -sample_client high -patience 5  -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &



# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client medium -patience 8 -bp
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client medium -patience 4 -bp
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client medium -patience 2 -bp
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client medium -patience 10 -bp
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client medium -patience 15 -bp
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client medium -patience 20 -bp
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client medium -patience 6 -bp
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client medium -patience 5 -bp

# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD2.py -i 25 -ex gmr -num_clients 10 -sample_client medium -patience 5  -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &

# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client low -patience 8 -bp
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client low -patience 4 -bp
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client low -patience 2 -bp
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client low -patience 10 -bp
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client low -patience 15 -bp
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client low -patience 20 -bp
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client low -patience 6 -bp
# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client low -patience 5 -bp

# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD2.py -i 25 -ex gmr -num_clients 10 -sample_client low -patience 5  -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &



# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD2.py -i 25 -ex gmr -num_clients 10 -sample_client low -patience 10 -niid  -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD2.py -i 25 -ex gmr -num_clients 10 -sample_client medium -patience 10 -niid -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD2.py -i 25 -ex gmr -num_clients 10 -sample_client high -patience 10 -niid  -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD2.py -i 25 -ex gmr -num_clients 10 -sample_client medium -patience 10 -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD2.py -i 25 -ex gmr -num_clients 10 -sample_client high -patience 10 -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD2.py -i 25 -ex gmr -num_clients 10 -sample_client low -patience 10  -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&

# python experiments/stackoverflow/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex fiarse -num_clients 10 -sample_client high -patience 5 -bp
# nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client high -patience 20 -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&


# # --- buff 组 ---
# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py -i 25 -ex buff -num_clients 10 -sample_client low -patience 20 -niid -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py -i 25 -ex buff -num_clients 10 -sample_client medium -patience 10 -niid -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py -i 25 -ex buff -num_clients 10 -sample_client high -patience 10 -niid -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py -i 25 -ex buff -num_clients 10 -sample_client medium -patience 20 -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py -i 25 -ex buff -num_clients 10 -sample_client high -patience 15 -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py -i 25 -ex buff -num_clients 10 -sample_client low -patience 8 -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&

# # --- pure 组 ---
# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py -i 25 -ex pure -num_clients 10 -sample_client low -patience 20 -niid -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py -i 25 -ex pure -num_clients 10 -sample_client medium -patience 10 -niid -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py -i 25 -ex pure -num_clients 10 -sample_client high -patience 10 -niid -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py -i 25 -ex pure -num_clients 10 -sample_client medium -patience 20 -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py -i 25 -ex pure -num_clients 10 -sample_client high -patience 15 -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py -i 25 -ex pure -num_clients 10 -sample_client low -patience 8 -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&

# # --- pure2 组 ---
# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py -i 25 -ex pure2 -num_clients 10 -sample_client low -patience 20 -niid -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py -i 25 -ex pure2 -num_clients 10 -sample_client medium -patience 10 -niid -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py -i 25 -ex pure2 -num_clients 10 -sample_client high -patience 10 -niid -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py -i 25 -ex pure2 -num_clients 10 -sample_client medium -patience 20 -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py -i 25 -ex pure2 -num_clients 10 -sample_client high -patience 15 -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py -i 25 -ex pure2 -num_clients 10 -sample_client low -patience 8 -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&

# # --- ims 组 ---
# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py -i 25 -ex ims -num_clients 10 -sample_client low -patience 20 -niid -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py -i 25 -ex ims -num_clients 10 -sample_client medium -patience 10 -niid -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py -i 25 -ex ims -num_clients 10 -sample_client high -patience 10 -niid -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py -i 25 -ex ims -num_clients 10 -sample_client medium -patience 20 -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py -i 25 -ex ims -num_clients 10 -sample_client high -patience 15 -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py -i 25 -ex ims -num_clients 10 -sample_client low -patience 8 -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&


# # --- asyn 组 ---
# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex asyn -num_clients 10 -sample_client medium -patience 20 -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex asyn -num_clients 10 -sample_client low -patience 8 -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex asyn -num_clients 10 -sample_client high -patience 10 -niid -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex asyn -num_clients 10 -sample_client low -patience 20 -niid -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex asyn -num_clients 10 -sample_client medium -patience 10 -niid -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex asyn -num_clients 10 -sample_client high -patience 15 -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&

# # --- re_mask_fed_avg 组 ---
# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex re_mask_fed_avg -num_clients 10 -sample_client medium -patience 20 -bp -re > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex re_mask_fed_avg -num_clients 10 -sample_client low -patience 8 -bp -re > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex re_mask_fed_avg -num_clients 10 -sample_client high -patience 10 -niid -bp -re > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex re_mask_fed_avg -num_clients 10 -sample_client low -patience 20 -niid -bp -re > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex re_mask_fed_avg -num_clients 10 -sample_client medium -patience 10 -niid -bp -re > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex re_mask_fed_avg -num_clients 10 -sample_client high -patience 15 -bp -re > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&

# # --- mask_fed_avg 组 ---
# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex mask_fed_avg -num_clients 10 -sample_client medium -patience 20 -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex mask_fed_avg -num_clients 10 -sample_client low -patience 8 -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex mask_fed_avg -num_clients 10 -sample_client high -patience 10 -niid -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex mask_fed_avg -num_clients 10 -sample_client low -patience 20 -niid -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex mask_fed_avg -num_clients 10 -sample_client medium -patience 10 -niid -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex mask_fed_avg -num_clients 10 -sample_client high -patience 15 -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&

# # --- re_gradient_avg 组 ---
# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD2.py -i 25 -ic 2.0 -ex re_gradient_avg -num_clients 10 -sample_client medium -patience 20 -bp -re > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD2.py -i 25 -ic 2.0 -ex re_gradient_avg -num_clients 10 -sample_client low -patience 8 -bp -re > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD2.py -i 25 -ic 2.0 -ex re_gradient_avg -num_clients 10 -sample_client high -patience 10 -niid -bp -re > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD2.py -i 25 -ic 2.0 -ex re_gradient_avg -num_clients 10 -sample_client low -patience 20 -niid -bp -re > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD2.py -i 25 -ic 2.0 -ex re_gradient_avg -num_clients 10 -sample_client medium -patience 10 -niid -bp -re > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD2.py -i 25 -ic 2.0 -ex re_gradient_avg -num_clients 10 -sample_client high -patience 15 -bp -re > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1


# # --- gradient_avg 组 ---
# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex gradient_avg -num_clients 10 -sample_client medium -patience 20 -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex gradient_avg -num_clients 10 -sample_client low -patience 8 -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex gradient_avg -num_clients 10 -sample_client high -patience 10 -niid -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex gradient_avg -num_clients 10 -sample_client low -patience 20 -niid -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex gradient_avg -num_clients 10 -sample_client medium -patience 10 -niid -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex gradient_avg -num_clients 10 -sample_client high -patience 15 -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&



# --- asyn 组 ---
nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex asyn -num_clients 10 -sample_client medium -patience 20 -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex asyn -num_clients 10 -sample_client low -patience 8 -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex asyn -num_clients 10 -sample_client high -patience 10 -niid -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex asyn -num_clients 10 -sample_client low -patience 20 -niid -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex asyn -num_clients 10 -sample_client medium -patience 10 -niid -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
nohup python experiments/stackoverflow/Ablation_Prune_increase_FL_CMD.py -i 25 -ic 2.0 -ex asyn -num_clients 10 -sample_client high -patience 15 -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
