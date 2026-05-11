
import lmdb
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import numpy as np

class LMDBDataset(Dataset):
    def __init__(self, lmdb_path, transform=None, is_test=False):
        """
        Args:
            lmdb_path (str): LMDB 文件的路径。
            transform (callable, optional): 图像的变换操作。
            is_test (bool): 是否为测试集（无标签）。
        """
        self.lmdb_path = lmdb_path
        self.transform = transform
        self.is_test = is_test
        self.targets = []  # 用于存储标签

        # 打开 LMDB 数据库，读取所有键值
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False)
        with self.env.begin() as txn:
            self.keys = [key.decode() for key, _ in txn.cursor() if key.decode().startswith("image")]
            if not is_test:
                for key in self.keys:
                    label_key = key.replace("image", "label").encode()
                    label = np.frombuffer(txn.get(label_key), dtype=np.int64)[0]
                    self.targets.append(label)  # 将标签存储到 targets 列表中

    def __getitem__(self, index):
        """
        根据索引返回图像和标签。
        """
        image_key = f"image-{index:05d}".encode()
        label_key = f"label-{index:05d}".encode()

        # 读取图像
        with self.env.begin() as txn:
            buffer = txn.get(image_key)
            image = cv2.imdecode(np.frombuffer(buffer, np.uint8), cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 如果有标签且不是测试集，读取标签
            if not self.is_test:
                label = np.frombuffer(txn.get(label_key), dtype=np.int64)[0]
            else:
                label = -1  # 测试集无标签

        # 应用变换
        if self.transform:
            image = self.transform(image)

        return (image, label) if not self.is_test else image

    def __len__(self):
        """返回数据集的长度"""
        return len(self.keys)

