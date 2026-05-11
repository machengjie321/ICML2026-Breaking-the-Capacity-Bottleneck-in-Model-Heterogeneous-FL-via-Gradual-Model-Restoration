import os


# os.chdir(r'D:\PR-FL')
# project_path = r"D:\PR-FL"
# import sys
# if project_path not in sys.path:
#     sys.path.append(project_path)
    
import torch
from bases.fl.simulation_real.Partial_Model_Training import AdaptiveServer, AdaptiveClient, AdaptiveFL, parse_args
from bases.optim.optimizer import SGD, MaskedSGD
from torch.optim import lr_scheduler
from bases.optim.optimizer_wrapper import OptimizerWrapper
from bases.vision.load import get_data_loader
from control.sub_algorithm import ControlModule

from bases.nn.models.resnet import resnet18
from configs.imagenet100 import *
import configs.imagenet100 as config

from utils.save_load import mkdir_save, load
from bases.vision.FLsampler import FLSampler
from utils.functional import compute_same_params_ratio

class INAdaptiveServer(AdaptiveServer):
    def init_test_loader(self):
        self.test_loader = get_data_loader(EXP_NAME, data_type="val", batch_size=200, num_workers=0, pin_memory=True)

    def init_clients(self):
        rand_perm = torch.randperm(NUM_TRAIN_DATA).tolist()
        indices = []
        len_slice = NUM_TRAIN_DATA // num_slices

        for i in range(num_slices):
            indices.append(rand_perm[i * len_slice: (i + 1) * len_slice])

        models = [self.model for _ in range(NUM_CLIENTS)]
        self.indices = indices
        return models, indices

    def init_control(self):
        self.control = ControlModule(self.model, config=config)

    def init_ip_config(self):
        pass

    def save_exp_config(self):
        pass


class INAdaptiveClient(AdaptiveClient):
    def init_optimizer(self):
        self.optimizer = SGD(self.model.parameters(), lr=INIT_LR, momentum=0, weight_decay=0)
        self.optimizer_scheduler = lr_scheduler.StepLR(self.optimizer, step_size=STEP_SIZE,
                                                       gamma=0.5 ** (STEP_SIZE / LR_HALF_LIFE))
        self.optimizer_wrapper = OptimizerWrapper(self.model, self.optimizer, self.optimizer_scheduler)

    def init_train_loader(self, tl):
        self.train_loader = tl

import torch
import numpy as np
import random
def set_seed(seed):
    torch.manual_seed(seed)  # PyTorch CPU
    np.random.seed(seed)  # NumPy
    random.seed(seed)  # Python 内置随机数
    torch.cuda.manual_seed(seed)  # PyTorch GPU（如果使用）
    torch.cuda.manual_seed_all(seed)  # 多 GPU
    torch.backends.cudnn.deterministic = True  # 确保 CUDA 计算可复现
    torch.backends.cudnn.benchmark = False  # 关闭自动优化（保证结果一致）


if __name__ == "__main__":

    args = parse_args()
    import os
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import datasets, transforms, models
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    if not args.use_adaptive:
        args.experiment_name = 'PMT'
        set_seed(args.seed)
        args.MaskSGD = False
        num_users = 100
        num_slices = num_users if args.client_selection else NUM_CLIENTS
        density_list = [1.0]

        
        test_loader = get_data_loader(EXP_NAME, data_type="val", batch_size=200, num_workers=2, pin_memory=True)
        server = INAdaptiveServer(args, config, resnet18(NUM_CLASSES))
        list_models, list_indices = server.init_clients()
        sampler = FLSampler(list_indices, 3000, NUM_LOCAL_UPDATES * CLIENT_BATCH_SIZE, args.client_selection,
                                num_slices)
        train_loader = get_data_loader(EXP_NAME, data_type="train", batch_size=128, shuffle=False,
                                           num_workers=2, pin_memory=True)
        # print(train_loader.dataset.class_to_idx)
        # 1. 超参数配置
        num_classes = 100
        batch_size = 64
        num_epochs = 50
        lr = 0.01
        image_size = 224

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 4. 模型定义（冻结除 fc 外所有参数）
        model = models.resnet18(pretrained=True)

        for param in model.parameters():
            param.requires_grad = False

        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model.to(device)

        # 5. 损失函数 & 优化器（只优化 fc）
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.fc.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

        # 6. 训练 + 验证循环
        for epoch in range(num_epochs):
            model.train()
            running_loss, correct, total = 0.0, 0, 0

            for inputs, labels in tqdm(train_loader, desc=f"[Train] Epoch {epoch+1}/{num_epochs}"):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            train_loss = running_loss / total
            train_acc = 100. * correct / total

            # 验证
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for inputs, labels in tqdm(test_loader, desc="[Eval]"):
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

            val_acc = 100. * correct / total
            scheduler.step()

            print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        # 7. 保存模型
        torch.save(model.state_dict(), "resnet18_imagenet100_fc_only.pth")




