import os
import sys
import json
from tqdm import tqdm

# ========== 设置项目路径 ==========
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from configs.StackOverflow import *
import configs.StackOverflow as config
from bases.vision.load import get_data_loader
from bases.nn.models.bp_transformer import transformer
import torch
from bases.optim.optimizer_wrapper import OptimizerWrapper
from bases.optim.optimizer import SGD


# ========== 准备保存结果 ==========
save_path = os.path.join(PROJECT_ROOT, "structure_prune_results.json")
final_results = {}

# ========== 主循环 ==========
for target_density in [1.0, 0.05, 0.2, 0.1, 0.05]:

    print(f"\n====== Density = {target_density} ======")

    # -------- 模型加载 --------
    # model = transformer(target_density)
    model = transformer(1.0)
    model.prune_by_pct(1-target_density)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # -------- Test Loader --------
    test_loader = get_data_loader(
        EXP_NAME,
        data_type="val",
        num_workers=test_num,
        pin_memory=True
    )

    # -------- Train Loader --------
    train_loader = get_data_loader(
        EXP_NAME,
        data_type="train",
        batch_size=CLIENT_BATCH_SIZE,
        shuffle=False,
        num_workers=config.train_num,
        pin_memory=True
    )

    # -------- Optimizer --------
    optimizer = SGD(
        model.parameters(),
        lr=INIT_LR,
        momentum=modementum,
        weight_decay=weight_decay
    )
    optimizer_wrapper = OptimizerWrapper(model, optimizer, clip=True)

    # -------- 训练 --------
    for i in tqdm(range(int(400/target_density)), desc=f"Training density {target_density}"):
        inputs, labels = train_loader.get_next_batch()
        list_grad, loss = optimizer_wrapper.step(inputs.to(device), labels.to(device))

        if i % 200 == 0:
            loss_val, acc_val = model.evaluate(test_loader)
            print(f"[Iter {i}] loss={loss_val:.4f}, acc={acc_val:.4f}")

    # -------- 最终评估 --------
    final_loss, final_acc = model.evaluate(test_loader)
    print(f"Final Result @ density={target_density}: loss={final_loss:.4f}, acc={final_acc:.4f}")

    # -------- 保存该 density 的结果 --------
    final_results[str(target_density)] = {
        "loss": float(final_loss),
        "acc": float(final_acc)
    }


# ========== 写入 JSON ==========
with open(save_path, "w") as f:
    json.dump(final_results, f, indent=4)

print("\n==============================")
print("All density results saved to:")
print(save_path)
print("==============================\n")
