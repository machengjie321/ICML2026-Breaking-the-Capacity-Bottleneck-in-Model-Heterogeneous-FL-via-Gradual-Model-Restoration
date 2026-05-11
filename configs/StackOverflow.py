BASE_EMB = 96
BASE_HID  = 384
NUM_HEADS = 4
HEAD_DIM  = BASE_EMB // NUM_HEADS

EXP_NAME = "stackoverflow"


NUM_TRAIN_DATA = 299968
NUM_TEST_DATA = 31571
NUM_USERS = 6000

acc_sign = 0.03


download_speed = 20
upload_speed = 5

min_density = 0.02
weight_decay = 1e-4
modementum =  0.9
INIT_LR = 2e-3

NUM_CLIENTS = 10     # Set the number of client
NUM_LOCAL_UPDATES = 10
CLIENT_BATCH_SIZE = 16


ADJ_INTERVAL = 10
EVAL_DISP_INTERVAL = 20

IP_MAX_ROUNDS = 5
IP_ADJ_INTERVAL = ADJ_INTERVAL
IP_DATA_BATCH = 10
IP_THR = 0.1

save_for_split = [0.83]


# Conv2
DENSE_TIME = 2.2486449218005875  # 10 times
SPARSE_ALL_TIME = 2.898095353249955  # all params are in, but sparse form
SPARSE_TIME = 1.2492789765
COMP_COEFFICIENTS_SINGLE = [0., 7.098850e-6, 1.927325e-7, 1.782308e-7]
COMP_COEFFICIENTS = [
  5.0,      # pos embedding（不重要 → 数值大）
  0.5,      # token embedding（非常重要 → 数值小）

  # Layer 0 MHA (4 layers)
  1.0, 1.0, 1.0, 1.0,

  # Layer 0 FFN
  0.6667, 0.6667,

  # Layer 1 MHA
  1.0, 1.0, 1.0, 1.0,

  # Layer 1 FFN
  0.6667, 0.6667,

  # Layer 2 MHA
  1.0, 1.0, 1.0, 1.0,

  # Layer 2 FFN
  0.6667, 0.6667,

  # decoder.linear1
  0.5,

  # decoder.linear2 (最重要)
  0.3333
]
COMM_COEFFICIENT = 0
C_COMP = SPARSE_TIME * NUM_LOCAL_UPDATES
C_COMM = 0.
TIME_CONSTANT = C_COMP + C_COMM

TO_SPARSE_THR = 0.9

MAX_INC_DIFF = None
MAX_DEC_DIFF = 0.3

ADJ_THR_FACTOR = 1.5
# ADJ_THR_ACC = ADJ_THR_FACTOR / NUM_CLASSES
ADJ_HALF_LIFE = 7000



MAX_ROUND_CONVENTIONAL_FL=1000
MAX_ROUND_ADAPTIVE=10000
test_num = 2
train_num = 2
# test_num = 0
# train_num = 0
patience = 25
asyn_interval = 0.5
# Iterative pruning config
NUM_ITERATIVE_PRUNING = 20

# Online algorithm config
MAX_NUM_UPLOAD = 5
model_size = 25.9811
#
MAX_TIME_PMT = 15000
PMT_ACC=[0.2,0.4,0.6,0.70,0.75,0.80, 0.82,0.83, 0.835, 0.84,0.843,0.846,0.848,0.85,0.855]
MAX_ROUND = 13001
# MAX_TIME = 210000
MAX_TIME = 3000000
# MAX_ROUND = 1000
# MAX_TIME = 20000

model_size = 17.26