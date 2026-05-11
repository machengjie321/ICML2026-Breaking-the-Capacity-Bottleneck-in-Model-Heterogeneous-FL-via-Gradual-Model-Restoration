import os
import sys

# 计算项目根目录：/mnt/data/mcj/AGMR-main
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
# 等价于手写：PROJECT_ROOT = "/mnt/data/mcj/AGMR-main"

# 把项目根目录加到 sys.path 里
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# （可选）切换工作目录到项目根，方便相对路径读取文件
os.chdir(PROJECT_ROOT)

from configs.StackOverflow import *
import configs.StackOverflow as config
from bases.vision.load import get_data_loader
from bases.nn.models.bp_transformer import transformer
import torch
from bases.optim.optimizer_wrapper import OptimizerWrapper
import torch.optim.lr_scheduler as lr_scheduler
from bases.optim.optimizer import SGD


# print('test_prune_0.5')
for target_density in [1]:
    model = transformer(1)
    test_loader = get_data_loader(EXP_NAME, data_type="val", num_workers=test_num, pin_memory=True)
    device = torch.device("cuda:"+str(1) if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.prune_by_pct(1-target_density)
    print(model.density())
    loss,acc = model.evaluate(test_loader)

    ds = get_data_loader(EXP_NAME, data_type="train", batch_size=CLIENT_BATCH_SIZE, shuffle=False,
                                        num_workers=config.train_num, pin_memory=True)

    import torch.optim.lr_scheduler as lr_scheduler

    optimizer = SGD(
        model.parameters(),
        lr=INIT_LR,
        momentum=modementum,
        weight_decay=weight_decay
    )
    optimizer_wrapper = OptimizerWrapper(model, optimizer,clip=True)



device = next(model.parameters()).device

from tqdm import tqdm
import matplotlib.pyplot as plt

EPOCHS = 10
PRINT_EVERY = 1000


train_losses = []
val_losses = []
val_top5_list = []


for epoch in range(1, EPOCHS + 1):

    model.train()
    total_loss = 0
    total_steps = 0

    progress = tqdm(ds, desc=f"Epoch {epoch}/{EPOCHS}")

    for step, (inputs, labels) in enumerate(progress):

        inputs = inputs.to(device)
        labels = labels.to(device)

        list_grad, loss = optimizer_wrapper.step(inputs, labels)

        total_loss += loss.item()
        total_steps += 1

        # 实时在进度条上显示当前 loss
        progress.set_postfix({"train_loss": loss.item()})

        # 每 N step 做一次验证
        if step % PRINT_EVERY == 0:
            val_loss, val_top5 = model.evaluate(test_loader)
            val_losses.append(val_loss)
            val_top5_list.append(val_top5)

            progress.write(f"[Val] loss={val_loss:.4f}, top5={val_top5:.4f}")

    # epoch 结束
    avg_train_loss = total_loss / total_steps
    train_losses.append(avg_train_loss)

    val_loss, val_top5 = model.evaluate(test_loader)
    val_losses.append(val_loss)
    val_top5_list.append(val_top5)

    print(f"=== Epoch {epoch} END | train_loss={avg_train_loss:.4f}, val_loss={val_loss:.4f}, top5={val_top5:.4f} ===")



