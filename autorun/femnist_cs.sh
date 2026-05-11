# source setenv.sh
# # save_for_split
# python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex pr_fl -ac wg -num_clients 10 -sample_client medium -patience 10 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex fed_asyn -ac wg -num_clients 10 -sample_client medium -patience 10 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex fed_asyn -ac wg -num_clients 10 -sample_client high -patience 10 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&


# nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ex pr_fl -ac wg -num_clients 10 -sample_client medium -patience 20 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ex fed_asyn -ac wg -num_clients 10 -sample_client medium -patience 5 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&

# nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ex pr_fl -ac wg -num_clients 10 -sample_client medium -patience 5 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/ImageNet100/Prune_increase_FL_CMD.py -i 50 -ex fed_asyn -ac wg -num_clients 10 -sample_client medium -patience 5 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# #python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex pr_fl -ac wg -num_clients 10 -sample_client medium -patience 20 -niid
# #
# #python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex heterofl -ac wg -num_clients 10 -sample_client medium -patience 10 -niid &&
# #python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex heterofl -ac wg -num_clients 10 -sample_client high -patience 10 -niid
# #
# #python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex heterofl -ac wg -num_clients 10 -sample_client low -patience 10 -niid &&
# #python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex heterofl -ac wg -num_clients 10 -sample_client medium -patience 10
# #
# #python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex fed_asyn -ac wg -num_clients 10 -sample_client low -patience 10 -niid &&
# #python experiments/FEMNIST/Ablation_Prune_increase_FL_CMD.py -i 25 -ex gmr -ac wg -num_clients 10 -sample_client low -patience 10 -niid
# #python experiments/FEMNIST/Ablation_Prune_increase_FL_CMD2.py -i 25 -ex ims -ac wg -num_clients 10 -sample_client medium -patience 10 -niid
# #

# #python3 experiments/FEMNIST/PMT.py -na -ni -s 0 -e conventional && python3 experiments/FEMNIST/PMT.py -a -i -s 0 -e adaptive && python3 experiments/FEMNIST/iterative.py -s 0 -e iterative && python3 experiments/FEMNIST/online.py -s 0 -e online && python3 experiments/FEMNIST/reinitialize.py -m r -s 0 -e reinit && python3 experiments/FEMNIST/reinitialize.py -m rr -s 0 -e random_reinit && python3 experiments/FEMNIST/snip.py -s 0 -e snip
# #
# #conda activate FL
# #cd /home/mcj/chengjie/PR-FL
# #source setenv.sh     #setup the path
# #python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ex pr_fl -ac wg -num_clients 10 -sample_client high -patience 15 -niid
# #
# #conda activate FL
# #cd /home/mcj/chengjie/PR-FL
# #nohup jupyter lab --ip=0.0.0.0 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
# #source setenv.sh
# #python experiments/FEMNIST/Ablation_Prune_increase_FL_CMD2.py -i 25 -ex ims -ac wg -num_clients 10 -sample_client medium -patience 10 -niid
# #
# #nohup python experiments/FEMNIST/Ablation_Prune_increase_FL_CMD2.py -i 25 -ex pr_fl -ac wg -num_clients 10 -sample_client medium -patience 10 -niid -awt 1 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# #nohup python experiments/FEMNIST/Ablation_Prune_increase_FL_CMD2.py -i 25 -ex pr_fl -ac wg -num_clients 10 -sample_client medium -patience 10 -niid -awt 2 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# #nohup python experiments/FEMNIST/Ablation_Prune_increase_FL_CMD2.py -i 25 -ex pr_fl -ac wg -num_clients 10 -sample_client medium -patience 10 -niid -awt 4 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# #nohup python experiments/FEMNIST/Ablation_Prune_increase_FL_CMD2.py -i 25 -ex pr_fl -ac wg -num_clients 10 -sample_client medium -patience 10 -niid -awt 0.1 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# #
# #
# ## ablation experiment
# #
# #nohup python experiments/FEMNIST/Ablation_Prune_increase_FL_CMD.py -i 25 -ex gmr -ac wg -num_clients 10 -sample_client medium -patience 10 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# #nohup python experiments/FEMNIST/Ablation_Prune_increase_FL_CMD.py -i 25 -ex asyn -ac wg -num_clients 10 -sample_client medium -patience 10 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# #nohup python experiments/FEMNIST/Ablation_Prune_increase_FL_CMD.py -i 25 -ex buff -ac wg -num_clients 10 -sample_client medium -patience 10 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# #nohup python experiments/FEMNIST/Ablation_Prune_increase_FL_CMD.py -i 25 -ex mask_fed_avg -ac wg -num_clients 10 -sample_client medium -patience 10 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# #nohup python experiments/FEMNIST/Ablation_Prune_increase_FL_CMD.py -i 25 -ex ims -ac wg -num_clients 10 -sample_client low -patience 10 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# #nohup python experiments/FEMNIST/Ablation_Prune_increase_FL_CMD.py -i 25 -ex gmr -ac wg -num_clients 10 -sample_client low -patience 10 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# #nohup python experiments/FEMNIST/Ablation_Prune_increase_FL_CMD.py -i 25 -ex asyn -ac wg -num_clients 10 -sample_client low -patience 10 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# #nohup python experiments/FEMNIST/Ablation_Prune_increase_FL_CMD.py -i 25 -ex buff -ac wg -num_clients 10 -sample_client low -patience 10 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# #nohup python experiments/FEMNIST/Ablation_Prune_increase_FL_CMD.py -i 25 -ex mask_fed_avg -ac wg -num_clients 10 -sample_client low -patience 10 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# #nohup python experiments/FEMNIST/Ablation_Prune_increase_FL_CMD.py -i 25 -ex ims -ac wg -num_clients 10 -sample_client high -patience 10 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# #nohup python experiments/FEMNIST/Ablation_Prune_increase_FL_CMD.py -i 25 -ex gmr -ac wg -num_clients 10 -sample_client high -patience 10 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# #nohup python experiments/FEMNIST/Ablation_Prune_increase_FL_CMD.py -i 25 -ex asyn -ac wg -num_clients 10 -sample_client high -patience 10 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# #nohup python experiments/FEMNIST/Ablation_Prune_increase_FL_CMD.py -i 25 -ex buff -ac wg -num_clients 10 -sample_client high -patience 10 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# #nohup python experiments/FEMNIST/Ablation_Prune_increase_FL_CMD.py -i 25 -ex mask_fed_avg -ac wg -num_clients 10 -sample_client high -patience 10 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# ## server_up ims
# #nohup python experiments/FEMNIST/Ablation_Prune_increase_FL_CMD.py -i 25 -ex pr_fl -ac wg -num_clients 10 -sample_client medium -patience 10 -niid -server_up 40.0 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# #nohup python experiments/FEMNIST/Ablation_Prune_increase_FL_CMD.py -i 25 -ex ims -ac wg -num_clients 10 -sample_client medium -patience 10 -niid -server_up 40.0 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# #nohup python experiments/FEMNIST/Ablation_Prune_increase_FL_CMD.py -i 25 -ex pr_fl -ac wg -num_clients 10 -sample_client medium -patience 10 -niid -server_up 20.0 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# #nohup python experiments/FEMNIST/Ablation_Prune_increase_FL_CMD.py -i 25 -ex ims -ac wg -num_clients 10 -sample_client medium -patience 10 -niid -server_up 20.0 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# #nohup python experiments/FEMNIST/Ablation_Prune_increase_FL_CMD.py -i 25 -ex pr_fl -ac wg -num_clients 10 -sample_client medium -patience 10 -niid -server_up 5.0 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# #nohup python experiments/FEMNIST/Ablation_Prune_increase_FL_CMD.py -i 25 -ex ims -ac wg -num_clients 10 -sample_client medium -patience 10 -niid -server_up 5.0 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# #nohup python experiments/FEMNIST/Ablation_Prune_increase_FL_CMD.py -i 25 -ex pr_fl -ac wg -num_clients 10 -sample_client medium -patience 10 -niid -server_up 2.5 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# #nohup python experiments/FEMNIST/Ablation_Prune_increase_FL_CMD.py -i 25 -ex ims -ac wg -num_clients 10 -sample_client medium -patience 10 -niid -server_up 2.5 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# #nohup python experiments/FEMNIST/Ablation_Prune_increase_FL_CMD.py -i 25 -ex pr_fl -ac wg -num_clients 10 -sample_client medium -patience 10 -niid -server_up 10.0 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# #nohup python experiments/FEMNIST/Ablation_Prune_increase_FL_CMD.py -i 25 -ex ims -ac wg -num_clients 10 -sample_client medium -patience 10 -niid -server_up 10.0 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
# #nohup python experiments/FEMNIST/Ablation_Prune_increase_FL_CMD.py -i 25 -ex pr_fl -ac wg -num_clients 10 -sample_client medium -patience 10 -niid -server_up 60.0 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# #nohup python experiments/FEMNIST/Ablation_Prune_increase_FL_CMD.py -i 25 -ex ims -ac wg -num_clients 10 -sample_client medium -patience 10 -niid -server_up 60.0 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# #
# ## patience for GMR
# ##hypermeter
# #nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex pr_fl -ac wg -num_clients 10 -sample_client high -patience 5 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# #nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex pr_fl -ac wg -num_clients 10 -sample_client high -patience 10 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
# #nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex pr_fl -ac wg -num_clients 10 -sample_client high -patience 15 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# #
# #nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex pr_fl -ac wg -num_clients 10 -sample_client low -patience 5 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# #nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex pr_fl -ac wg -num_clients 10 -sample_client low -patience 10 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# #nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex pr_fl -ac wg -num_clients 10 -sample_client low -patience 15 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# #
# #nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex pr_fl -ac wg -num_clients 10 -sample_client medium -patience 5 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# #nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex pr_fl -ac wg -num_clients 10 -sample_client medium -patience 10 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# #nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex pr_fl -ac wg -num_clients 10 -sample_client medium -patience 15 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# #
# #
# #
# ##BaseLine
# ##experiment
# #nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex pr_fl -ac wg -num_clients 10 -sample_client medium -patience 10 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# #nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex fed_avg -ac wg -num_clients 10 -sample_client medium -patience 10 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# #nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex fed_asyn -ac wg -num_clients 10 -sample_client medium -patience 10 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# #nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex heterofl -ac wg -num_clients 10 -sample_client medium -patience 10 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# #nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex pr_fl -ac wg -num_clients 10 -sample_client high -patience 10 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# #nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex fed_avg -ac wg -num_clients 10 -sample_client high -patience 10 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# #nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex fed_asyn -ac wg -num_clients 10 -sample_client high -patience 10 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# #nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex heterofl -ac wg -num_clients 10 -sample_client high -patience 10 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# #nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex pr_fl -ac wg -num_clients 10 -sample_client low -patience 10 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# #nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex fed_avg -ac wg -num_clients 10 -sample_client low -patience 10 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# #nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex fed_asyn -ac wg -num_clients 10 -sample_client low -patience 10 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# #nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex heterofl -ac wg -num_clients 10 -sample_client low -patience 10 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# #
# #nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex fed_asyn -ac wg -num_clients 10 -sample_client medium -patience 10 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# #nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex fed_avg -ac wg -num_clients 10 -sample_client medium -patience 10 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# #nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex heterofl -ac wg -num_clients 10 -sample_client medium -patience 10 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# #nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex fed_avg -ac wg -num_clients 10 -sample_client high -patience 10 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# #nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex fed_asyn -ac wg -num_clients 10 -sample_client high -patience 10 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# #nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex heterofl -ac wg -num_clients 10 -sample_client high -patience 10 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# #nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex fed_avg -ac wg -num_clients 10 -sample_client low -patience 10 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# #nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex fed_asyn -ac wg -num_clients 10 -sample_client low -patience 10 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# #nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex heterofl -ac wg -num_clients 10 -sample_client low -patience 10 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
# #
# #
# #
# #
# #
# ##有无Re, 有无Asyn， 两种聚合方式，有无res
# ## Ablation experiment
# #nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex fedprox -ac wg -num_clients 10 -sample_client medium -patience 5 -niid -mu 1 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# #nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex fedprox -ac wg -num_clients 10 -sample_client medium -patience 5 -niid -mu 2 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# #nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex fedprox -ac wg -num_clients 10 -sample_client medium -patience 5 -niid -mu 0.1 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# #nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex fedprox -ac wg -num_clients 10 -sample_client medium -patience 5 -niid -mu 0.5 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# #nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex fedprox -ac wg -num_clients 10 -sample_client medium -patience 5 -niid -mu 0.05 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# #nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex fedprox -ac wg -num_clients 10 -sample_client medium -patience 5 -niid -mu 0.08 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# #nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex fedprox -ac wg -num_clients 10 -sample_client medium -patience 5 -niid -mu 0.03 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# #nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex fedprox -ac wg -num_clients 10 -sample_client medium -patience 5 -niid -mu 0.01 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# #
# #
# ##nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex fedprox -ac wg -num_clients 10 -sample_client medium -patience 5 -mu 2 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# #nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex fedprox -ac wg -num_clients 10 -sample_client medium -patience 5 -mu 1 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# #nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex fedprox -ac wg -num_clients 10 -sample_client medium -patience 5 -mu 0.5 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# #nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex fedprox -ac wg -num_clients 10 -sample_client medium -patience 5 -mu 0.1 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# #nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex fedprox -ac wg -num_clients 10 -sample_client medium -patience 5 -mu 0.005 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# #nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex fedprox -ac wg -num_clients 10 -sample_client medium -patience 5 -mu 0.001 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# #nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex fedprox -ac wg -num_clients 10 -sample_client medium -patience 5 -mu 0.008 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# #nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex fedprox -ac wg -num_clients 10 -sample_client medium -patience 5 -mu 0.003> "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# #nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex fedprox -ac wg -num_clients 10 -sample_client medium -patience 5 -mu 0.05 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# #nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex fedprox -ac wg -num_clients 10 -sample_client medium -patience 5 -mu 0.08 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# #nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex fedprox -ac wg -num_clients 10 -sample_client medium -patience 5 -mu 0.03 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# #nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex fedprox -ac wg -num_clients 10 -sample_client medium -patience 5 -mu 0.01 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
# #
# #nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex pr_fl -ac wg -num_clients 10 -sample_client medium -patience 10 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# #nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex pr_fl -ac wg -num_clients 10 -sample_client medium -patience 10 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# #
# #nohup python experiments/FEMNIST/Ablation_Prune_increase_FL_CMD.py -i 25 -ex ims -ac wg -num_clients 10 -sample_client medium -patience 10 -niid > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# #nohup python experiments/FEMNIST/Ablation_Prune_increase_FL_CMD.py -i 25 -ex ims -ac wg -num_clients 10 -sample_client medium -patience 10 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# #nohup python experiments/FEMNIST/Ablation_Prune_increase_FL_CMD.py -i 25 -ex ims -ac wg -num_clients 10 -sample_client medium -patience 10 -niid -server_up 40.0 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# #nohup python experiments/FEMNIST/Ablation_Prune_increase_FL_CMD.py -i 25 -ex ims -ac wg -num_clients 10 -sample_client medium -patience 10 -niid -server_up 20.0 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# #nohup python experiments/FEMNIST/Ablation_Prune_increase_FL_CMD.py -i 25 -ex ims -ac wg -num_clients 10 -sample_client medium -patience 10 -niid -server_up 5.0 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# #nohup python experiments/FEMNIST/Ablation_Prune_increase_FL_CMD.py -i 25 -ex ims -ac wg -num_clients 10 -sample_client medium -patience 10 -niid -server_up 2.5 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
# #
# #nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex fedprox -ac wg -num_clients 10 -sample_client medium -patience 10 -mu 0.005 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
# #nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex fedprox -ac wg -num_clients 10 -sample_client medium -patience 10 -niid -mu 0.1 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
# #nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex fedprox -ac wg -num_clients 10 -sample_client low -patience 10 -niid -mu 0.5 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# #nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex fedprox -ac wg -num_clients 10 -sample_client high -patience 10 -niid -mu 0.5 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# #nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex fedprox -ac wg -num_clients 10 -sample_client medium -patience 10 -mu 0.005 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# #nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex fedprox -ac wg -num_clients 10 -sample_client low -patience 10 -mu 0.005 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# #nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex fedprox -ac wg -num_clients 10 -sample_client high -patience 10 -mu 0.005 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
# #
# #
# #
# #
# #
# #
# #
# #nohup python experiments/FEMNIST/Ablation_Prune_increase_FL_CMD.py -i 25 -ex asyn -ac wg -num_clients 10 -sample_client medium -patience 10 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# #nohup python experiments/FEMNIST/Ablation_Prune_increase_FL_CMD.py -i 25 -ex buff -ac wg -num_clients 10 -sample_client medium -patience 10 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# #nohup python experiments/FEMNIST/Ablation_Prune_increase_FL_CMD.py -i 25 -ex mask_fed_avg -ac wg -num_clients 10 -sample_client medium -patience 10 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# #
# ##nohup python experiments/FEMNIST/Ablation_Prune_increase_FL_CMD.py -i 25 -ex gmr -ac wg -num_clients 10 -sample_client high -patience 10 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# ##nohup python experiments/FEMNIST/Ablation_Prune_increase_FL_CMD.py -i 25 -ex ims -ac wg -num_clients 10 -sample_client high -patience 10 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# ##nohup python experiments/FEMNIST/Ablation_Prune_increase_FL_CMD.py -i 25 -ex asyn -ac wg -num_clients 10 -sample_client high -patience 10 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# ##nohup python experiments/FEMNIST/Ablation_Prune_increase_FL_CMD.py -i 25 -ex buff -ac wg -num_clients 10 -sample_client high -patience 10 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &
# ##nohup python experiments/FEMNIST/Ablation_Prune_increase_FL_CMD.py -i 25 -ex mask_fed_avg -ac wg -num_clients 10 -sample_client high -patience 10 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# #
# ##nohup python experiments/FEMNIST/Ablation_Prune_increase_FL_CMD.py -i 25 -ex gmr -ac wg -num_clients 10 -sample_client low -patience 10 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# ##nohup python experiments/FEMNIST/Ablation_Prune_increase_FL_CMD.py -i 25 -ex ims -ac wg -num_clients 10 -sample_client low -patience 10 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# ##nohup python experiments/FEMNIST/Ablation_Prune_increase_FL_CMD.py -i 25 -ex asyn -ac wg -num_clients 10 -sample_client low -patience 10 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# ##nohup python experiments/FEMNIST/Ablation_Prune_increase_FL_CMD.py -i 25 -ex buff -ac wg -num_clients 10 -sample_client low -patience 10 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# ##nohup python experiments/FEMNIST/Ablation_Prune_increase_FL_CMD.py -i 25 -ex mask_fed_avg -ac wg -num_clients 10 -sample_client low -patience 10 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex pr_fl -ac wg -num_clients 10 -sample_client high -patience 5 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# #nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex pr_fl -ac wg -num_clients 10 -sample_client high -patience 10 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# #nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex pr_fl -ac wg -num_clients 10 -sample_client high -patience 15 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# #nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex pr_fl -ac wg -num_clients 10 -sample_client low -patience 5 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# #nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex pr_fl -ac wg -num_clients 10 -sample_client low -patience 10 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# #nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex pr_fl -ac wg -num_clients 10 -sample_client low -patience 15 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# #nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex pr_fl -ac wg -num_clients 10 -sample_client medium -patience 5 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# #nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex pr_fl -ac wg -num_clients 10 -sample_client medium -patience 10 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# #nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex pr_fl -ac wg -num_clients 10 -sample_client medium -patience 15 > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# #
# #
# #
# #
# #
# #
# #
# #python experiments/FEMNIST/PGT.py -i 25 -ex fed_avg -ac wg -num_clients 10 -sample_client low -patience 10 -niid
# #
# #
# #python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ex fed_avg -ac wg -num_clients 10 -sample_client low -patience 10 -niid
# #
# #source setenv.sh
# #python experiments/FEMNIST/Ablation_Prune_increase_FL_CMD2.py -i 25 -ex ims -ac wg -num_clients 10 -sample_client medium -patience 10 -niid

# nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client medium -patience 35 -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client low -patience 35 -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client high -patience 35 -niid  -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client low -patience 35 -niid  -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client medium -patience 35 -niid -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client high -patience 35 -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&

# nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client medium -patience 40 -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client low -patience 40 -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client high -patience 40 -niid  -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client low -patience 40 -niid  -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client medium -patience 40 -niid -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client high -patience 40 -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&

# # nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client medium -patience 45 -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# # nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client low -patience 45 -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# # nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client high -patience 45 -niid  -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# # nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client low -patience 45 -niid  -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# # nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client medium -patience 45 -niid -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# # nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client high -patience 45 -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&

# nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client medium -patience 50 -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client low -patience 50 -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client high -patience 50 -niid  -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client low -patience 50 -niid  -bp > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client medium -patience 50 -niid -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/CIFAR10/Prune_increase_FL_CMD.py -i 50 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client high -patience 50 -bp   > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&

# nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client high -patience 4 -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client high -patience 6 -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client medium -patience 2 -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client medium -patience 4 -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client medium -patience 6 -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client medium -patience 8 -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client medium -patience 9 -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client high -patience 4 -niid  -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client high -patience 6 -niid  -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client high -patience 8 -niid  -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client medium -patience 2 -niid -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client medium -patience 4 -niid -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&


# nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client high -patience 2 -niid  -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client high -patience 9 -niid  -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
# nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client high -patience 12 -niid  -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&

nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client high -patience 11 -niid  -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&

nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client high -patience 13 -niid  -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&

nohup python experiments/FEMNIST/Prune_increase_FL_CMD.py -i 25 -ic 2.0  -ex pr_fl -num_clients 10 -sample_client high -patience 14 -niid  -bp  > "log_$(date +'%Y%m%d%H%M%S%3N').log" 2>&1 &&
