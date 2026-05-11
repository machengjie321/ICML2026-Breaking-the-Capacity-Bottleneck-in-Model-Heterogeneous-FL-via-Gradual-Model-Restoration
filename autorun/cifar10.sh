#python experiments/CelebA/Prune_increase_FL_CMD.py -i 25 -ex pr_fl -ac wg -num_clients 10 -sample_client medium -patience 5 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
#python experiments/CelebA/Prune_increase_FL_CMD.py -i 25 -ex fed_asyn -ac wg -num_clients 10 -sample_client medium -patience 5 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1


#python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex pr_fl -ac wg -num_clients 10 -sample_client medium -patience 20 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
#python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex fed_asyn -ac wg -num_clients 10 -sample_client medium -patience 20 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1

#nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ex pr_fl -ac wg -num_clients 10 -sample_client medium -patience 20 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ex fed_asyn -ac wg -num_clients 10 -sample_client medium -patience 5 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&

nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ex pr_fl -ac wg -num_clients 10 -sample_client medium -patience 5 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ex fed_asyn -ac wg -num_clients 10 -sample_client medium -patience 5 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
#nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ex heterofl -ac wg -num_clients 10 -sample_client medium -patience 5  -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
#nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ex Asyn -ac wg -num_clients 10 -sample_client medium -patience 5  -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
#
#nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ex heterofl -ac wg -num_clients 10 -sample_client medium -patience 5  -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
#nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ex Asyn -ac wg -num_clients 10 -sample_client medium -patience 5  -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
#
#nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ex heterofl -ac wg -num_clients 10 -sample_client medium -patience 5  -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
#nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ex Asyn -ac wg -num_clients 10 -sample_client medium -patience 5  -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&

#nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ex heterofl -ac wg -num_clients 10 -sample_client low -patience 5 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
#nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ex heterofl -ac wg -num_clients 10 -sample_client high -patience 5 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
#nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ex heterofl -ac wg -num_clients 10 -sample_client medium -patience 5 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
#nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ex heterofl -ac wg -num_clients 10 -sample_client low -patience 5 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
#nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ex heterofl -ac wg -num_clients 10 -sample_client high -patience 5 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
#nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ex heterofl -ac wg -num_clients 10 -sample_client medium -patience 5 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&

#nohup python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 50 -ex gmr -ac wg -num_clients 10 -sample_client low -patience 5 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
#nohup python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 50 -ex gmr -ac wg -num_clients 10 -sample_client high -patience 5 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
#nohup python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 50 -ex gmr -ac wg -num_clients 10 -sample_client medium -patience 5 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
#nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex heterofl -ac wg -num_clients 10 -sample_client low -patience 10 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
#nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex heterofl -ac wg -num_clients 10 -sample_client medium -patience 10 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
#nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex heterofl -ac wg -num_clients 10 -sample_client high -patience 10 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
#python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex heterofl -ac wg -num_clients 10 -sample_client medium -patience 10 -niid && python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex heterofl -ac wg -num_clients 10 -sample_client high -patience 10 -niid && python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex heterofl -ac wg -num_clients 10 -sample_client medium -patience 10
#
##nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex heterofl -ac wg -num_clients 10 -sample_client low -patience 10 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
#nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex heterofl -ac wg -num_clients 10 -sample_client medium -patience 10 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
##nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex heterofl -ac wg -num_clients 10 -sample_client high -patience 10 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
#python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex heterofl -ac wg -num_clients 10 -sample_client medium -patience 10 -niid
#python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex heterofl -ac wg -num_clients 10 -sample_client high -patience 10 -niid
#python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex heterofl -ac wg -num_clients 10 -sample_client medium -patience 10

#nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ex pr_fl -ac wg -num_clients 10 -sample_client medium -patience 5  -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
#nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ex pr_fl -ac wg -num_clients 10 -sample_client medium -patience 10  -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
#nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ex pr_fl -ac wg -num_clients 10 -sample_client medium -patience 30  -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
#nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ex pr_fl -ac wg -num_clients 10 -sample_client medium -patience 10 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
#nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ex pr_fl -ac wg -num_clients 10 -sample_client medium -patience 20 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&

#nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex heterofl -ac wg -num_clients 10 -sample_client low -patience 10 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
#nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex heterofl -ac wg -num_clients 10 -sample_client medium -patience 10 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
#nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex heterofl -ac wg -num_clients 10 -sample_client high -patience 10 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
#
##nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex heterofl -ac wg -num_clients 10 -sample_client low -patience 10 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
#nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex heterofl -ac wg -num_clients 10 -sample_client medium -patience 10 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
##nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex heterofl -ac wg -num_clients 10 -sample_client high -patience 10 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &




#nohup python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 50 -ex gmr -ac wg -num_clients 10 -sample_client medium -patience 5 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ex pr_fl -ac wg -num_clients 10 -sample_client medium -patience 10 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# # nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ex pr_fl -ac wg -num_clients 10 -sample_client medium -patience 20 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ex pr_fl -ac wg -num_clients 10 -sample_client medium -patience 30 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# # nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ex pr_fl -ac wg -num_clients 10 -sample_client medium -patience 40 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ex pr_fl -ac wg -num_clients 10 -sample_client medium -patience 50 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ex pr_fl -ac wg -num_clients 10 -sample_client medium -patience 60 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
#nohup python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 50 -ex gmr -ac wg -num_clients 10 -sample_client high -patience 5 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ex pr_fl -ac wg -num_clients 10 -sample_client high -patience 10 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# # nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ex pr_fl -ac wg -num_clients 10 -sample_client high -patience 20 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ex pr_fl -ac wg -num_clients 10 -sample_client high -patience 30 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# # nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ex pr_fl -ac wg -num_clients 10 -sample_client high -patience 40 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ex pr_fl -ac wg -num_clients 10 -sample_client high -patience 50 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ex pr_fl -ac wg -num_clients 10 -sample_client high -patience 60 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 50 -ex gmr -ac wg -num_clients 10 -sample_client low -patience 5 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ex pr_fl -ac wg -num_clients 10 -sample_client low -patience 10 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# # nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ex pr_fl -ac wg -num_clients 10 -sample_client low -patience 20 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ex pr_fl -ac wg -num_clients 10 -sample_client low -patience 30 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# # nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ex pr_fl -ac wg -num_clients 10 -sample_client low -patience 40 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ex pr_fl -ac wg -num_clients 10 -sample_client low -patience 50 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ex pr_fl -ac wg -num_clients 10 -sample_client low -patience 60 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&

#nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ex fed_asyn -ac wg -num_clients 10 -sample_client low -patience 5 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
#nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ex fed_asyn -ac wg -num_clients 10 -sample_client high -patience 5 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
#nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ex fed_asyn -ac wg -num_clients 10 -sample_client medium -patience 5 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &


#nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ex fed_avg -ac wg -num_clients 10 -sample_client low -patience 5 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
#nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ex fed_avg -ac wg -num_clients 10 -sample_client high -patience 5 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
#nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ex fed_avg -ac wg -num_clients 10 -sample_client medium -patience 5 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
#

##patience for GMR
##hypermeter
##nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ex pr_fl -ac wg -num_clients 10 -sample_client medium -patience 5 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
##nohup python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 50 -ex mask_fed_avg -ac wg -num_clients 10 -sample_client low -patience 5 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
#
##nohup python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 50 -ex mask_fed_avg -ac wg -num_clients 10 -sample_client high -patience 5 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
##nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ex pr_fl -ac wg -num_clients 10 -sample_client low -patience 5 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
##nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ex pr_fl -ac wg -num_clients 10 -sample_client high -patience 5 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
#nohup python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 50 -ex mask_fed_avg -ac wg -num_clients 10 -sample_client medium -patience 5 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
#
#
#nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ex pr_fl -ac wg -num_clients 10 -sample_client high -patience 2 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
#nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ex pr_fl -ac wg -num_clients 10 -sample_client high -patience 5 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
#nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ex pr_fl -ac wg -num_clients 10 -sample_client high -patience 10 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
#nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ex pr_fl -ac wg -num_clients 10 -sample_client low -patience 2 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
#nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ex pr_fl -ac wg -num_clients 10 -sample_client low -patience 5 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
#nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ex pr_fl -ac wg -num_clients 10 -sample_client low -patience 10 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
#nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 500 -ex pr_fl -ac wg -num_clients 10 -sample_client medium -patience 2 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
##nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ex pr_fl -ac wg -num_clients 10 -sample_client medium -patience 5 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
#nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ex pr_fl -ac wg -num_clients 10 -sample_client medium -patience 10 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
#

## ablation experiment
#nohup python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 50 -ex gmr -ac wg -num_clients 10 -sample_client medium -patience 5 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
#nohup python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 50 -ex asyn -ac wg -num_clients 10 -sample_client medium -patience 5 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
#nohup python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 50 -ex buff -ac wg -num_clients 10 -sample_client medium -patience 5 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
##nohup python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 50 -ex mask_fed_avg -ac wg -num_clients 10 -sample_client medium -patience 5 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
#nohup python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 50 -ex ims -ac wg -num_clients 10 -sample_client low -patience 5 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
#nohup python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 50 -ex gmr -ac wg -num_clients 10 -sample_client low -patience 5 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
#nohup python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 50 -ex asyn -ac wg -num_clients 10 -sample_client low -patience 5 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
#nohup python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 50 -ex buff -ac wg -num_clients 10 -sample_client low -patience 5 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
##nohup python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 50 -ex mask_fed_avg -ac wg -num_clients 10 -sample_client low -patience 5 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
#nohup python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 50 -ex ims -ac wg -num_clients 10 -sample_client high -patience 5 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
#nohup python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 50 -ex gmr -ac wg -num_clients 10 -sample_client high -patience 5 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
#nohup python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 50 -ex asyn -ac wg -num_clients 10 -sample_client high -patience 5 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
#nohup python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 50 -ex buff -ac wg -num_clients 10 -sample_client high -patience 5 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
##nohup python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 50 -ex mask_fed_avg -ac wg -num_clients 10 -sample_client high -patience 5 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
#
##BaseLine
#nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ex fed_asyn -ac wg -num_clients 10 -sample_client medium -patience 5 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
#nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ex fed_avg -ac wg -num_clients 10 -sample_client medium -patience 5 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
#nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ex heterofl -ac wg -num_clients 10 -sample_client medium -patience 5 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
#nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ex fed_avg -ac wg -num_clients 10 -sample_client high -patience 5 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
#nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ex fed_asyn -ac wg -num_clients 10 -sample_client high -patience 5 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
#nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ex heterofl -ac wg -num_clients 10 -sample_client high -patience 5 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
#nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ex fed_avg -ac wg -num_clients 10 -sample_client low -patience 5 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
#nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ex fed_asyn -ac wg -num_clients 10 -sample_client low -patience 5 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
#nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ex heterofl -ac wg -num_clients 10 -sample_client low -patience 5 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
#
#
##BaseLine
#nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ex pr_fl -ac wg -num_clients 10 -sample_client medium -patience 5 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
#nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ex fed_avg -ac wg -num_clients 10 -sample_client medium -patience 5 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
#nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ex fed_asyn -ac wg -num_clients 10 -sample_client medium -patience 5 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
#nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ex heterofl -ac wg -num_clients 10 -sample_client medium -patience 5 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
#nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ex pr_fl -ac wg -num_clients 10 -sample_client high -patience 5 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
#nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ex fed_avg -ac wg -num_clients 10 -sample_client high -patience 5 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
#nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ex fed_asyn -ac wg -num_clients 10 -sample_client high -patience 5 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
#nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ex heterofl -ac wg -num_clients 10 -sample_client high -patience 5 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
#nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ex pr_fl -ac wg -num_clients 10 -sample_client low -patience 5 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
#nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ex fed_avg -ac wg -num_clients 10 -sample_client low -patience 5 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
#nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ex fed_asyn -ac wg -num_clients 10 -sample_client low -patience 5 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
#nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ex heterofl -ac wg -num_clients 10 -sample_client low -patience 5 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
#
#
#
## server_up ims
#nohup python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 50 -ex pr_fl -ac wg -num_clients 10 -sample_client medium -patience 5 -niid -server_up 40.0 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
#nohup python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 50 -ex ims -ac wg -num_clients 10 -sample_client medium -patience 5 -niid -server_up 40.0 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
#nohup python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 50 -ex pr_fl -ac wg -num_clients 10 -sample_client medium -patience 5 -niid -server_up 20.0 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
#nohup python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 50 -ex ims -ac wg -num_clients 10 -sample_client medium -patience 5 -niid -server_up 20.0 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
#nohup python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 50 -ex pr_fl -ac wg -num_clients 10 -sample_client medium -patience 5 -niid -server_up 5.0 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
#nohup python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 50 -ex ims -ac wg -num_clients 10 -sample_client medium -patience 5 -niid -server_up 5.0 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
#nohup python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 50 -ex pr_fl -ac wg -num_clients 10 -sample_client medium -patience 5 -niid -server_up 2.5 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
#nohup python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 50 -ex ims -ac wg -num_clients 10 -sample_client medium -patience 5 -niid -server_up 2.5 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
#nohup python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 50 -ex pr_fl -ac wg -num_clients 10 -sample_client medium -patience 5 -niid -server_up 10.0 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
#nohup python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 50 -ex ims -ac wg -num_clients 10 -sample_client medium -patience 5 -niid -server_up 10.0 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
#nohup python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 50 -ex pr_fl -ac wg -num_clients 10 -sample_client medium -patience 5 -niid -server_up 60.0 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
#nohup python experiments/CIFAR10/Ablation_Prune_increase_FL_CMD.py -i 50 -ex ims -ac wg -num_clients 10 -sample_client medium -patience 5 -niid -server_up 60.0 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&


# ========================= Time-trigger GMR (CIFAR10) =========================
# total virtual time: 220000s
# nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ex pr_fl -num_clients 10 -sample_client high -patience 7 -niid -bp --recover_trigger_mode time --recover_time_total 220000 --recover_time_points 0.2,0.4,0.6,0.8,1.0 --recover_time_ladder 0.05,0.1,0.2,0.5,1.0 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
# nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ex gmr_fiarse -num_clients 10 -sample_client high -patience 7 -niid -bp --recover_trigger_mode time --recover_time_total 220000 --recover_time_points 0.2,0.4,0.6,0.8,1.0 --recover_time_ladder 0.05,0.1,0.2,0.5,1.0 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
