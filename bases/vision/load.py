import os
from os.path import join
import torch
import torchvision
import torchvision.transforms as transforms
from bases.vision.data_loader import DataLoader
from bases.vision.build_stackoverflow_simple import ensure_stackoverflow_preprocessed
from bases.vision.transforms import Flatten, OneHot, DataToTensor
from configs import femnist, celeba, cifar10, imagenet100, mnist, StackOverflow
from bases.vision.datasets import FEMNIST, CelebA, TinyImageNet,ImageNet100,StackOverflowLM,collate_fn_stackoverflow,StackOverflowClientDataset
from bases.vision.get_lmdb import LMDBDataset

__all__ = ["get_data", "get_data_loader"]


def get_config_by_name(name: str):
    if name.lower() == "femnist":
        return femnist
    elif name.lower() == "celeba":
        return celeba
    elif name.lower() == "cifar10":
        return cifar10
    elif name.lower() in ["tinyimagenet", "tiny-imagenet-200"]:
        return TinyImageNet
    elif name.lower() == "mnist":
        return mnist
    elif name.lower() in ["imagenet", "imagenet100"]:
        return ImageNet100
    elif name.lower() == 'stackoverflow':
        return StackOverflow
    else:
        raise ValueError("{} is not supported.".format(name))


def get_data(name: str, data_type, transform=None, target_transform=None, user_list=None):
    dataset = get_config_by_name(name)
    print(dataset)

    if dataset == femnist:
        assert data_type in ["train", "test"]
        if transform is None:
            transform = transforms.Compose([transforms.ToTensor()])
        if target_transform is None:
            target_transform = transforms.Compose(
                [DataToTensor(dtype=torch.long), OneHot(dataset.NUM_CLASSES, to_float=True)])

        return FEMNIST(root=join("datasets", "FEMNIST"), train=data_type == "train", download=True, transform=transform,
                       target_transform=target_transform, user_list=user_list)
    if dataset == mnist:
        assert data_type in ["train", "test"]
        transform = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(),
             torchvision.transforms.Normalize((0.1307,), (0.3081,))]
        )
        train_transform = transforms.Compose([
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomRotation((-10, 10)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])
        test_transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.1307,), (0.3081,))])

        train_folder = join("datasets", "MNIST", "train")
        test_folder = join("datasets", "MNIST", "test")
        dl = True
        if os.path.isdir(train_folder) and os.path.isdir(test_folder):
            dl = False

        from torchvision import datasets
        if data_type == 'train':

            data = datasets.MNIST(root=train_folder,
                                    train=True,
                                    transform=train_transform,
                                    download=dl)
        else:
            data = datasets.MNIST(root=test_folder,
                                   train=False,
                                   transform=test_transform,
                                   download=dl)
        return data

    elif dataset == celeba:
        assert data_type in ["train", "test"]


        if transform is None:
            transform = transforms.Compose([transforms.Resize((84, 84)),
                                            transforms.ToTensor()])
        if target_transform is None:
            target_transform = transforms.Compose([DataToTensor(dtype=torch.long)])

        return CelebA(root=join("datasets", "CelebA"), train=data_type == "train", download=True, transform=transform,
                      target_transform=target_transform, user_list=user_list)

    elif dataset == cifar10:
        assert data_type in ["train", "test"]
        if transform is None:
            mean = [0.4914, 0.4822, 0.4465]
            std = [0.2023, 0.1994, 0.2010]
            if data_type == "train":
                transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                                transforms.RandomCrop(32, 4),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean, std)])
            else:
                transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize(mean, std)])
        if target_transform is None:
            target_transform = transforms.Compose([DataToTensor(dtype=torch.long),
                                                   OneHot(dataset.NUM_CLASSES, to_float=True)])
        import contextlib
        with contextlib.redirect_stdout(None):
            data = torchvision.datasets.CIFAR10(root=join("datasets", "CIFAR10"), train=data_type == "train", download=True,
                                         transform=transform, target_transform=target_transform)
        return data

    elif dataset == TinyImageNet:

        assert data_type in ["train", "test", "val"]


        # 设置默认的 transform
        if transform is None:

            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]

            if data_type == "train":
                transform = transforms.Compose([
                    transforms.RandomCrop(64, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            else:
                transform = transforms.Compose([
                    transforms.Resize((64, 64)),  # 直接调整图像大小到目标尺寸
                    transforms.ToTensor(),  # 转为张量
                    transforms.Normalize(mean, std)  # 归一化
                ])

        # 返回 LMDB 数据集

        return TinyImageNet(root=join("datasets", "TinyImageNet"), data_type = data_type, transform=transform,
                           target_transform=None,download=True)


    elif dataset == ImageNet100:

        assert data_type in ["train", "test", "val"]

        # # 设置默认的 transform
        # if transform is None:
        #     if data_type == "train":
        #         transform = transforms.Compose([
        #             transforms.RandomResizedCrop(168),  # 如果显存允许，也可以使用 168
        #             transforms.RandomHorizontalFlip(),
        #             transforms.RandomRotation(15),
        #             transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        #             transforms.RandomGrayscale(p=0.1),
        #             transforms.ToTensor(),
        #             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        #         ])
        #     else:
        #         transform = transforms.Compose([
        #             transforms.Resize((168, 168)),
        #             transforms.ToTensor(),
        #             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        #         ])
        #         elif dataset == imagenet100:

        if transform is None:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            if data_type == "train":
                transform = transforms.Compose([transforms.RandomResizedCrop(168),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean, std)])
            else:
                transform = transforms.Compose([transforms.Resize(168),
                                                transforms.CenterCrop(imagenet100.IMAGE_SIZE),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean, std)])

        # 返回 LMDB 数据集

        return ImageNet100(root=join("datasets", "ImageNet100"), data_type = data_type, transform=transform,
                           target_transform=None,download=True)

    elif dataset == StackOverflow:
        assert data_type in ["train",  "val","vocab"]
        root = "./datasets/stackoverflow"

        if data_type == "train":
            return StackOverflowLM(root, split="train", download=True)
        elif data_type == "val":
            return StackOverflowClientDataset(root, 21, 24, download=True)
    
        elif data_type == "vocab":
            ensure_stackoverflow_preprocessed(save_dir=root)
            return torch.load(os.path.join(root, "stackoverflow_vocab.pt"))

    else:
        raise ValueError("{} dataset is not supported.".format(name))



def get_data_loader(name: str, data_type, batch_size=None, shuffle: bool = False, sampler=None, transform=None,
                    target_transform=None, subset_indices=None, num_workers=8, pin_memory=True, user_list=None):
    assert data_type in ["train", "val", "test"]
    if data_type == "train":
        assert batch_size is not None, "Batch size for training data is required"

    data = get_data(name, data_type=data_type, transform=transform, target_transform=target_transform,
                    user_list=user_list)

    if data_type == "test" or data_type == "val":
        assert sampler is None, "Cannot shuffle when using sampler"

        if name == "TinyImageNet": n = 2
        elif name == "ImageNet100": n = 1
        elif name == "CelebA": n = 1
        elif name == "FEMNIST": n = 1
        elif name == "CIFAR10": n = 1
        else: n = 1
        from torch.utils.data import SubsetRandomSampler
         # 计算要抽样的子集大小（假设是原数据集大小的 1/10）
        subset_size = len(data) // n
        # 生成随机的子集索引
        indices = torch.randperm(len(data))[:subset_size]
        # 使用 SubsetRandomSampler 创建新的数据加载器
        sampler = SubsetRandomSampler(indices)



    if subset_indices is not None:
        data = torch.utils.data.Subset(data, subset_indices)
    if data_type != "train" and batch_size is None:
        batch_size = len(data)
    if name.lower() == 'stackoverflow':
        if data_type == 'train':
            return DataLoader(data, batch_size=batch_size, shuffle=shuffle, sampler=sampler, num_workers=num_workers,
                      pin_memory=pin_memory, collate_fn=collate_fn_stackoverflow)
        else:
            return DataLoader(
                data,
                batch_size=1,        # 每个 item 已经是 batch
                shuffle=False,       # 不应shuffle，因为它是 language model stream
                collate_fn=lambda x: x[0],   # 直接返回 dict
                num_workers=num_workers,
                pin_memory=pin_memory,
                
            )    
    else:
        return DataLoader(data, batch_size=batch_size, shuffle=shuffle, sampler=sampler, num_workers=num_workers,
                      pin_memory=pin_memory)
