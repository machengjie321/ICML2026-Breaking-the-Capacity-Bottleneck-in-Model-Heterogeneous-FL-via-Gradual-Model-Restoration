import fnmatch
import json
import os
import numpy as np
import shutil
import subprocess
import warnings
from shutil import move
import cv2
import lmdb
import torch
from PIL import Image
from torchvision import transforms
from torchvision.datasets import VisionDataset, ImageFolder
from torchvision.datasets.folder import default_loader
from tqdm import tqdm

from bases.vision.build_stackoverflow_simple import ensure_stackoverflow_preprocessed
from configs.femnist import IMG_DIM


def _run_shell(cmd, cwd=None):
    result = subprocess.run(cmd, cwd=cwd, shell=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {cmd}")


def _replace_path(src, dst):
    if not os.path.exists(src):
        raise FileNotFoundError(f"Expected path does not exist: {src}")
    if os.path.exists(dst):
        if os.path.isdir(dst) and not os.path.islink(dst):
            shutil.rmtree(dst)
        else:
            os.remove(dst)
    shutil.move(src, dst)


def _remove_path(path):
    if not os.path.exists(path):
        return
    if os.path.isdir(path) and not os.path.islink(path):
        shutil.rmtree(path)
    else:
        os.remove(path)

class FEMNIST(VisionDataset):
    """
    classes: 10 digits, 26 lower cases, 26 upper cases.
    We use torch.save, torch.load in this dataset
    """

    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, user_list: list = None):
        super(FEMNIST, self).__init__(root, transform=transform, target_transform=target_transform)
        """
        0 <= any user in user_list < total_users
        """
        self.train = train
        self.user_list = user_list

        if download:
            self.download()
        elif self._raw_exists() and not self._processed_exists():
            self.process()

        if not self._check_exists():
            raise FileNotFoundError(
                "FEMNIST dataset not found. Use download=True to auto-download it, or place the LEAF-generated "
                f"raw train/test JSON files under {self.raw_folder} and rerun."
            )

        self.total_num_users = torch.load(os.path.join(self.processed_folder, "num_users.pt"),weights_only=True)

        if self.user_list is not None:
            self.num_users = len(self.user_list)
        else:
            self.user_list = [i for i in range(self.total_num_users)]
            self.num_users = self.total_num_users

        if self.train:
            self.data, self.targets = self.load(train=True)
        else:
            self.data, self.targets = self.load(train=False)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # Needs 0~255, uint8 scale
        img = Image.fromarray(np.uint8(255 * (1 - img.numpy())), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    @property
    def raw_folder(self):
        return os.path.join(self.root, "raw")

    @property
    def all_data_folder(self):
        return os.path.join(self.root, "femnist", "data", "raw_data")

    @property
    def processed_folder(self):
        return os.path.join(self.root, "processed")

    @property
    def leaf_repo_folder(self):
        return os.path.join(self.root, "github_repo")

    def _raw_exists(self):
        return (
            os.path.isdir(os.path.join(self.raw_folder, "train")) and
            os.path.isdir(os.path.join(self.raw_folder, "test"))
        )

    def _processed_exists(self):
        return os.path.isfile(os.path.join(self.processed_folder, "num_users.pt"))

    def _check_exists(self):
        return self._processed_exists()

    def download(self):
        if self._processed_exists():
            return

        if self._raw_exists():
            self.process()
            return

        os.makedirs(self.root, exist_ok=True)
        os.makedirs(self.raw_folder, exist_ok=True)

        if not os.path.isdir(self.leaf_repo_folder):
            _run_shell(f"git clone https://github.com/TalwalkarLab/leaf.git {self.leaf_repo_folder}")

        femnist_dir = os.path.join(self.leaf_repo_folder, "data", "femnist")
        if not os.path.isdir(femnist_dir):
            raise RuntimeError(f"LEAF FEMNIST directory not found under {self.leaf_repo_folder}")

        _run_shell("./preprocess.sh -s niid --sf 0.05 -k 0 -t sample", cwd=femnist_dir)

        generated_train = os.path.join(femnist_dir, "data", "train")
        generated_test = os.path.join(femnist_dir, "data", "test")
        if not (os.path.isdir(generated_train) and os.path.isdir(generated_test)):
            raise RuntimeError("FEMNIST preprocessing did not produce train/test folders as expected.")

        _replace_path(generated_train, os.path.join(self.raw_folder, "train"))
        _replace_path(generated_test, os.path.join(self.raw_folder, "test"))

        self.process()

    def process(self):
        print("Processing data...")

        if not self._raw_exists():
            raise FileNotFoundError(
                f"FEMNIST raw data not found under {self.raw_folder}. "
                "Download first or place the LEAF-generated train/test folders there."
            )

        os.makedirs(self.processed_folder, exist_ok=True)
        for fname in os.listdir(self.processed_folder):
            if fnmatch.fnmatch(fname, "train_*.pt") or fnmatch.fnmatch(fname, "test_*.pt") or fname == "num_users.pt":
                _remove_path(os.path.join(self.processed_folder, fname))

        total_users_train = 0
        list_train_f = [f for f in os.listdir(os.path.join(self.raw_folder, "train")) if
                        fnmatch.fnmatch(f, "*.json")]
        list_train_f.sort(key=lambda fname: int(fname[9:-28]))

        for filename in list_train_f:
            with open(os.path.join(self.raw_folder, "train", filename)) as file:
                data = json.load(file)
                for user_name, val in data["user_data"].items():
                    # key: user name
                    # val: dict {x: x_data, y: y_data}
                    x = torch.tensor(val["x"]).reshape((-1, *IMG_DIM))
                    y = torch.tensor(val["y"])

                    torch.save((x, y), os.path.join(self.processed_folder, "train_{}.pt".format(total_users_train)))
                    total_users_train += 1

        total_users_test = 0
        list_test_f = [f for f in os.listdir(os.path.join(self.raw_folder, "test")) if fnmatch.fnmatch(f, "*.json")]
        list_test_f.sort(key=lambda fname: int(fname[9:-27]))

        for filename in list_test_f:
            with open(os.path.join(self.raw_folder, "test", filename)) as file:
                data = json.load(file)
                for user_name, val in data["user_data"].items():
                    # key: user name
                    # val: dict {x: x_data, y: y_data}
                    x = torch.tensor(val["x"]).reshape((-1, *IMG_DIM))
                    y = torch.tensor(val["y"])

                    torch.save((x, y), os.path.join(self.processed_folder, "test_{}.pt").format(total_users_test))
                    total_users_test += 1

        assert total_users_train == total_users_test
        torch.save(total_users_train, os.path.join(self.processed_folder, "num_users.pt"))
        print("Done. {} users processed.".format(total_users_train))

    def load(self, train):
        if train:
            prf = "train"
        else:
            prf = "test"

        data_list, label_list = [], []
        for user_id in self.user_list:
            x, y = torch.load(
                os.path.join(self.processed_folder, "{}_{}.pt".format(prf, user_id)),
                weights_only=True  # 仅加载权重，避免 FutureWarning
            )
            data_list.append(x)
            label_list.append(y)
        return torch.cat(data_list, dim=0), torch.cat(label_list, dim=0)


class CelebA(VisionDataset):
    """
    The Leaf CelebA dataset. See "https://github.com/TalwalkarLab/leaf/tree/master/data/celeba" for details.
    We use torch.save, torch.load in this dataset.
    """

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, user_list: list = None):
        super(CelebA, self).__init__(root, transform=transform, target_transform=target_transform)
        self.train = train
        self.user_list = user_list
        self.loader = default_loader

        if download:
            self.download()
        elif self._raw_exists() and not self._processed_exists():
            self.process()

        if not self._check_exists():
            raise FileNotFoundError(
                "CelebA dataset not found. Place the required raw files under "
                f"{self.raw_folder} and rerun, or use download=True to trigger auto-processing once the raw files exist."
            )

        self.total_num_users = torch.load(os.path.join(self.processed_folder, "num_users.pt"))

        if self.user_list is not None:
            self.num_users = len(self.user_list)
        else:
            self.user_list = list(range(self.total_num_users))
            self.num_users = self.total_num_users

        if self.train:
            self.img_paths, self.labels = self.load(train=True)

        else:
            self.img_paths, self.labels = self.load(train=False)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        path, target = self.img_paths[index], self.labels[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.img_paths)

    @property
    def raw_folder(self):
        return os.path.join(self.root, "raw")

    @property
    def processed_folder(self):
        return os.path.join(self.root, "processed")

    @property
    def leaf_github_folder(self):
        return os.path.join(self.root, "github_repo")

    def _raw_exists(self):
        return (
            os.path.isdir(os.path.join(self.raw_folder, "img_align_celeba")) and
            os.path.isfile(os.path.join(self.raw_folder, "identity_CelebA.txt")) and
            os.path.isfile(os.path.join(self.raw_folder, "list_attr_celeba.txt"))
        )

    def _processed_exists(self):
        return (
            os.path.isfile(os.path.join(self.processed_folder, "num_users.pt")) and
            os.path.isfile(os.path.join(self.processed_folder, "train_meta.pt")) and
            os.path.isfile(os.path.join(self.processed_folder, "test_meta.pt"))
        )

    def _check_exists(self):
        return os.path.isdir(os.path.join(self.raw_folder, "img_align_celeba")) and self._processed_exists()

    def download(self):
        if self._check_exists():
            return

        if self._raw_exists():
            self.process()
            return

        os.makedirs(self.raw_folder, exist_ok=True)
        raise RuntimeError(
            "Automatic download is not supported for CelebA in this project. "
            f"Please place img_align_celeba, identity_CelebA.txt, and list_attr_celeba.txt under {self.raw_folder}, "
            "then rerun the same command and preprocessing will continue automatically."
        )

    def process(self):
        print("Processing data...")

        if not self._raw_exists():
            raise FileNotFoundError(
                f"CelebA raw files are incomplete under {self.raw_folder}. "
                "Expected img_align_celeba, identity_CelebA.txt, and list_attr_celeba.txt."
            )

        os.makedirs(self.processed_folder, exist_ok=True)

        root = self.root
        if not os.path.exists(self.leaf_github_folder):
            _run_shell(f"git clone https://github.com/TalwalkarLab/leaf.git {root}/github_repo")

        celeba_dir = os.path.join(root, "github_repo", "data", "celeba")
        leaf_raw_dir = os.path.join(celeba_dir, "data", "raw")
        os.makedirs(leaf_raw_dir, exist_ok=True)

        processed_train_dir = os.path.join(self.processed_folder, "train")
        processed_test_dir = os.path.join(self.processed_folder, "test")
        if not (os.path.isdir(processed_train_dir) and os.path.isdir(processed_test_dir)):
            _replace_path(os.path.join(root, "raw", "img_align_celeba"), os.path.join(leaf_raw_dir, "img_align_celeba"))
            _replace_path(os.path.join(root, "raw", "identity_CelebA.txt"), os.path.join(leaf_raw_dir, "identity_CelebA.txt"))
            _replace_path(os.path.join(root, "raw", "list_attr_celeba.txt"), os.path.join(leaf_raw_dir, "list_attr_celeba.txt"))

            _run_shell("./preprocess.sh -s niid --sf 1. -k 0 -t sample", cwd=celeba_dir)

            _replace_path(os.path.join(leaf_raw_dir, "img_align_celeba"), os.path.join(root, "raw", "img_align_celeba"))
            _replace_path(os.path.join(leaf_raw_dir, "identity_CelebA.txt"), os.path.join(root, "raw", "identity_CelebA.txt"))
            _replace_path(os.path.join(leaf_raw_dir, "list_attr_celeba.txt"), os.path.join(root, "raw", "list_attr_celeba.txt"))

            _replace_path(os.path.join(celeba_dir, "data", "train"), processed_train_dir)
            _replace_path(os.path.join(celeba_dir, "data", "test"), processed_test_dir)

        for meta_name in ("num_users.pt", "train_meta.pt", "test_meta.pt"):
            _remove_path(os.path.join(self.processed_folder, meta_name))

        # train data
        total_users_train = 0
        list_train_f = [f for f in os.listdir(os.path.join(self.processed_folder, "train")) if
                        fnmatch.fnmatch(f, "*.json")]
        assert len(list_train_f) == 1
        filename = list_train_f[0]

        renamed_users_train = []
        with open(os.path.join(self.processed_folder, "train", filename)) as file:
            data = json.load(file)
            sorted_user_name = sorted(data["user_data"].keys())
            for user_name in sorted_user_name:
                val = data["user_data"][user_name]
                renamed_users_train.append(val)

                total_users_train += 1

        # test data
        total_users_test = 0
        list_test_f = [f for f in os.listdir(os.path.join(self.processed_folder, "test")) if
                       fnmatch.fnmatch(f, "*.json")]
        assert len(list_test_f) == 1
        filename = list_test_f[0]

        renamed_users_test = []
        with open(os.path.join(self.processed_folder, "test", filename)) as file:
            data = json.load(file)
            sorted_user_name = sorted(data["user_data"].keys())
            for user_name in sorted_user_name:
                val = data["user_data"][user_name]
                renamed_users_test.append(val)

                total_users_test += 1

        assert total_users_train == total_users_test
        torch.save(total_users_train, os.path.join(self.processed_folder, "num_users.pt"))
        torch.save(renamed_users_train, os.path.join(self.processed_folder, "train_meta.pt"))
        torch.save(renamed_users_test, os.path.join(self.processed_folder, "test_meta.pt"))
        print("Done. {} users processed.".format(total_users_train))

    def load(self, train):
        if train:
            prf = "train"
        else:
            prf = "test"

        meta_data = torch.load(os.path.join(self.processed_folder, "{}_meta.pt".format(prf)))

        path_list, label_list = [], []
        for user_id in self.user_list:
            x, y = meta_data[user_id]["x"], meta_data[user_id]["y"]
            path_list.extend([os.path.join(self.raw_folder, "img_align_celeba", p) for p in x])
            label_list.extend(y)
        return path_list, label_list


class ImageNet100(VisionDataset):
    def __init__(self, root, data_type, transform=None, target_transform=None, download=False):
        super().__init__(root, transform=transform, target_transform=target_transform)
        assert data_type in ["train", "test", "val"], "data_type must be 'train', 'test', or 'val'."

        self.root = root
        self.data_type = "val" if data_type == "test" else data_type
        self.lmdb_train_path = os.path.join(self.root, "imagenet100_train.lmdb")
        self.lmdb_val_path = os.path.join(self.root, "imagenet100_val.lmdb")
        self.raw_root = os.path.join(self.root, "ILSVRC")

        if download:
            self.download()
        elif self._raw_exists() and not self._processed_exists():
            self.process()

        if not self._check_exists():
            raise FileNotFoundError(
                "ImageNet100 dataset not found. Place the raw ILSVRC files under "
                f"{self.raw_root} and rerun, or use download=True to trigger processing once the raw files exist."
            )

        if self.data_type == "train":
            path_to_data = self.lmdb_train_path
        else:
            path_to_data = self.lmdb_val_path

        self.dataset = TinyImageNetDataset(path_to_data, transform=transform)
        self.targets = self.dataset.labels

    @property
    def train_file(self):
        return os.path.join(self.raw_root, "ImageSets", "CLS-LOC", "train_cls.txt")

    @property
    def val_solution_file(self):
        return os.path.join(self.raw_root, "LOC_val_solution.csv")

    @property
    def mapping_file(self):
        return os.path.join(self.raw_root, "LOC_synset_mapping.txt")

    @property
    def train_image_dir(self):
        return os.path.join(self.raw_root, "Data", "CLS-LOC", "train")

    @property
    def val_image_dir(self):
        return os.path.join(self.raw_root, "Data", "CLS-LOC", "val")

    def _processed_exists(self):
        return os.path.exists(self.lmdb_train_path) and os.path.exists(self.lmdb_val_path)

    def _raw_exists(self):
        return all(
            os.path.exists(path)
            for path in (
                self.train_file,
                self.val_solution_file,
                self.mapping_file,
                self.train_image_dir,
                self.val_image_dir,
            )
        )

    def download(self):
        if self._processed_exists():
            return

        if self._raw_exists():
            self.process()
            return

        os.makedirs(self.root, exist_ok=True)
        raise RuntimeError(
            "Automatic download is not supported for ImageNet100. "
            "Please download the ILSVRC files manually, place them under "
            f"{self.raw_root}, then rerun the same command and LMDB conversion will continue automatically."
        )

    def process(self):
        if self._processed_exists():
            return
        if not self._raw_exists():
            raise FileNotFoundError(
                "ImageNet100 raw files are incomplete. Expected "
                f"{self.train_file}, {self.val_solution_file}, {self.mapping_file}, "
                f"{self.train_image_dir}, and {self.val_image_dir}."
            )

        image_size = (168, 168)
        top_100_wnids = []
        with open(self.mapping_file, 'r') as f:
            for idx, line in enumerate(f):
                if idx >= 100:
                    break
                wnid = line.strip().split(' ')[0]
                top_100_wnids.append(wnid)

        train_image_list = []
        for wnid in top_100_wnids:
            class_dir = os.path.join(self.train_image_dir, wnid)
            if not os.path.exists(class_dir):
                print(f"Warning: class directory does not exist: {class_dir}")
                continue

            for image_name in os.listdir(class_dir):
                if image_name.endswith('.JPEG'):
                    image_path = os.path.join(class_dir, image_name)
                    train_image_list.append((image_path, wnid))

        val_image_list = []
        with open(self.val_solution_file, 'r') as f:
            lines = f.readlines()[1:]
            for line in lines:
                parts = line.strip().split(',', 1)
                if len(parts) < 2:
                    continue

                image_id = parts[0]
                prediction_string = parts[1]
                wnid = prediction_string.split()[0]
                if wnid in top_100_wnids:
                    image_path = os.path.join(self.val_image_dir, f"{image_id}.JPEG")
                    val_image_list.append((image_path, wnid))

        self.convert_to_lmdb(top_100_wnids, train_image_list, self.lmdb_train_path, image_size)
        self.convert_to_lmdb(top_100_wnids, val_image_list, self.lmdb_val_path, image_size)

    def convert_to_lmdb(self, top_100_wnids, image_list, lmdb_path, image_size):
        """
        Convert image-label pairs into LMDB and map WordNet IDs to integer labels.
        """
        _remove_path(lmdb_path)
        lmdb_dir = os.path.dirname(lmdb_path)
        if not os.path.exists(lmdb_dir):
            os.makedirs(lmdb_dir)
        label_mapping = {wnid: idx for idx, wnid in enumerate(top_100_wnids)}
        map_size = len(image_list) * 3 * image_size[0] * image_size[1] * 2
        env = lmdb.open(lmdb_path, map_size=map_size)

        with env.begin(write=True) as txn:
            for idx, (image_path, wnid) in enumerate(tqdm(image_list, desc=f"Writing {os.path.basename(lmdb_path)}")):
                if not os.path.exists(image_path):
                    print(f"Warning: image not found {image_path}")
                    continue

                image = cv2.imread(image_path)
                if image is None:
                    print(f"Error: unable to read image {image_path}")
                    continue

                image = cv2.resize(image, image_size)
                _, buffer = cv2.imencode('.jpg', image)
                txn.put(f"image-{idx:05d}".encode(), buffer.tobytes())
                label = label_mapping.get(wnid, -1)
                txn.put(f"label-{idx:05d}".encode(), np.array(label, dtype=np.int64).tobytes())
            txn.put(b"__len__", str(len(image_list)).encode())

        env.close()
        print(f"LMDB created: {lmdb_path}")


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def _check_exists(self):
        return self._processed_exists()


import lmdb
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

import lmdb
import cv2
import numpy as np
from PIL import Image
from torchvision.datasets import VisionDataset

import os
import torch
from torch.utils.data import Dataset



def collate_fn_stackoverflow(batch, pad_id=0):
    seqs = [x for (x, _) in batch]

    max_len = max(s.size(0) for s in seqs)
    B = len(seqs)

    labels = torch.full((B, max_len), pad_id, dtype=torch.long)

    for i, seq in enumerate(seqs):
        L = seq.size(0)
        labels[i, :L] = seq

    return labels, torch.tensor(0)

class StackOverflowLM(Dataset):
    """
    CIFAR10 风格的 Dataset：
    - 读取 ./datasets/stackoverflow/stackoverflow_{train,val}.pt
    - 每条样本是一个 token 序列（LongTensor），再加一个 dummy label（目前没用）
    """
    def __init__(self, root, split="train", download=True):
        """
        Args:
            root: 根目录，例如 "./datasets/stackoverflow"
            split: "train" 或 "val"
        """
        assert split in ["train", "val", "test"]
        if split == "test":
            split = "val"          # 兼容写法

        self.root = root
        self.split = split

        if download:
            ensure_stackoverflow_preprocessed(save_dir=root)

        pt_path = os.path.join(root, f"stackoverflow_{split}.pt")
        if not os.path.exists(pt_path):
            raise FileNotFoundError(
                f"{pt_path} not found. Install TensorFlow Federated and rerun, or generate it manually with "
                "bases/vision/build_stackoverflow_simple.py."
            )
        
        obj = torch.load(pt_path)
        self.sequences = obj["x"]
        self.labels = obj["y"]
        self.user_ids  = obj.get("user_ids", None)  # list[int]，如果你已经改了 export_split

        assert len(self.sequences) == len(self.labels), "x / y length mismatch"
        if self.user_ids is not None:
            assert len(self.user_ids) == len(self.sequences), "user_ids / x length mismatch"


    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        """
        返回风格类似 CIFAR10: (data, target)

        data: LongTensor(seq_len)   -> token id 序列
        target: LongTensor(1) 或标量 -> dummy label，目前可以不用
        """
        x = self.sequences[idx]
        y = self.labels[idx]
        return x, y
    
    
class StackOverflowClientDataset(Dataset):
    def __init__(self, root, seq_length, batch_size, download=True):

        # print('222')
        # print(root)
        self.root = root
        self.split = "val"

        if download:
            ensure_stackoverflow_preprocessed(save_dir=root)

        pt_path = os.path.join(root, "stackoverflow_val.pt")
        if not os.path.exists(pt_path):
            raise FileNotFoundError(
                f"{pt_path} not found. Install TensorFlow Federated and rerun, or generate it manually with "
                "bases/vision/build_stackoverflow_simple.py."
            )
        raw = torch.load(pt_path)
        token = torch.cat(raw["x"], dim=0)
        
        self.seq_length = seq_length
        self.token = token
        num_batch = len(token) // (batch_size * seq_length)
        self.token = self.token.narrow(0, 0, num_batch * batch_size * seq_length)
        self.token = self.token.reshape(-1, batch_size, seq_length)

    def __getitem__(self, index):
        return  self.token[index, :, :].reshape(-1, self.seq_length)

    def __len__(self):
        return len(self.token)
    
class TinyImageNetDataset(VisionDataset):
    def __init__(self, lmdb_path, transform=None):
        super().__init__(lmdb_path, transform=transform)
        self.lmdb_path = lmdb_path
        self.transform = transform
        self.env = lmdb.open(
            lmdb_path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=1,
        )
        self.length = 0
        self.labels = []

        with self.env.begin(write=False) as txn:
            meta_len = txn.get(b"__len__")
            if meta_len is not None:
                self.length = int(meta_len.decode())
            else:
                entries = self.env.stat().get("entries", 0)
                if entries == 0 or txn.get(b"image-00000") is None:
                    self.length = 0
                elif txn.get(b"label-00000") is None:
                    self.length = entries
                else:
                    self.length = entries // 2

            has_labels = txn.get(b"label-00000") is not None
            if has_labels:
                for idx in range(self.length):
                    label_buffer = txn.get(f"label-{idx:05d}".encode())
                    if label_buffer is None:
                        self.labels.append(-1)
                    else:
                        self.labels.append(np.frombuffer(label_buffer, dtype=np.int64)[0])
            else:
                self.labels = [-1] * self.length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with self.env.begin(write=False) as txn:
            image_key = f"image-{idx:05d}".encode()
            image_buffer = txn.get(image_key)
            if image_buffer is None:
                raise ValueError(f"Missing image data for key: {image_key}")

            image = cv2.imdecode(np.frombuffer(image_buffer, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Failed to decode image data for key: {image_key}")

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)

            label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

    def __del__(self):
        env = getattr(self, "env", None)
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


class TinyImageNet(VisionDataset):
    def __init__(self, root, data_type, transform=None, target_transform=None, download=False):
        super().__init__(root, transform=transform, target_transform=target_transform)
        assert data_type in ["train", "val"], "data_type must be 'train' or 'val'."
        self.parent_dir = root
        self.root = root
        self.train_folder = os.path.join(self.root, "tiny-imagenet-200", "train")
        self.val_folder = os.path.join(self.root, "tiny-imagenet-200", "val")
        self.lmdb_train_path = os.path.join(self.root, "tiny_imagenet_train.lmdb")
        self.lmdb_val_path = os.path.join(self.root, "tiny_imagenet_val.lmdb")

        if download:
            self.download()

        if not self._check_exists():
            raise FileNotFoundError("Dataset not found. Use download=True to download it.")

        if data_type == "train":
            path_to_data = self.lmdb_train_path
        elif data_type == "val":
            path_to_data = self.lmdb_val_path

        self.dataset = TinyImageNetDataset(path_to_data, transform=transform)
        self.targets = self.dataset.labels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def _check_exists(self):
        return os.path.exists(self.lmdb_train_path) and os.path.exists(self.lmdb_val_path)

    def download(self):
        if self._check_exists():
            print("Data already downloaded.")
            return

        if not os.path.isdir(self.root):
            os.mkdir(self.root)

        os.system(rf"cd {self.root} "
                  r"&& wget -nc http://cs231n.stanford.edu/tiny-imagenet-200.zip "
                  r"&& unzip -o tiny-imagenet-200.zip")

        self.process()

    def process(self):
        print("Processing validation set...")

        # 确保 val_annotations.txt 存在
        annotations_path = os.path.join(self.val_folder, "val_annotations.txt")
        if not os.path.exists(annotations_path):
            raise FileNotFoundError(f"Missing val_annotations.txt at {annotations_path}")

        # 检查是否已完成类别子目录的处理
        categories = [d for d in os.listdir(self.val_folder) if os.path.isdir(os.path.join(self.val_folder, d))]
        if len(categories) > 0 and "images" not in categories:
            print("Validation set already processed into categories. Skipping reorganization.")
        else:
            # 按照 val_annotations.txt 组织图片到类别子目录
            file_to_class = {}
            with open(annotations_path, 'r') as f:
                for line in f.readlines():
                    split_line = line.split('\t')
                    file_name, class_name = split_line[0], split_line[1]
                    file_to_class[file_name] = class_name
                    dir_path = os.path.join(self.val_folder, class_name)
                    if not os.path.exists(dir_path):
                        os.mkdir(dir_path)

            # 移动图片到对应类别文件夹
            image_folder_path = os.path.join(self.val_folder, "images")
            all_imgs = os.listdir(image_folder_path)
            for img_name in all_imgs:
                if img_name in file_to_class:
                    img_class = file_to_class[img_name]
                    move(os.path.join(image_folder_path, img_name), os.path.join(self.val_folder, img_class, img_name))

            # 删除空的 images 文件夹
            os.rmdir(image_folder_path)
            print("Validation set processing complete.")

        # 创建 class_to_idx 映射（基于训练集的类别）
        classes = sorted(os.listdir(self.train_folder))  # 确保类别顺序与训练集一致
        class_to_idx = {class_name: idx for idx, class_name in enumerate(classes)}

        # 转换为 LMDB 格式
        self.convert_to_lmdb(self.train_folder, self.lmdb_train_path, dataset_type="train", class_to_idx=class_to_idx)
        self.convert_to_lmdb(self.val_folder, self.lmdb_val_path, dataset_type="val", class_to_idx=class_to_idx)

    def convert_to_lmdb(self, image_folder, lmdb_path, dataset_type="train", class_to_idx=None):
        print(f"Converting {dataset_type} dataset to LMDB format...")
        image_paths = []
        labels = []

        if dataset_type == "train":
            # 遍历类别子目录
            for class_name in os.listdir(image_folder):
                class_folder = os.path.join(image_folder, class_name)
                if not os.path.isdir(class_folder):
                    continue  # 跳过非目录项

                # 遍历文件夹中的图片
                for file in os.listdir(class_folder):
                    file_path = os.path.join(class_folder, file)
                    if os.path.isfile(file_path):
                        image_paths.append(file_path)
                        if class_to_idx:
                            labels.append(class_to_idx[class_name])
                        else:
                            print(f"Warning: Class name {class_name} not found in class_to_idx mapping.")
                            labels.append(-1)  # 默认标签 -1

        elif dataset_type == "val":
            # 遍历类别子目录
            for class_name in os.listdir(image_folder):
                class_folder = os.path.join(image_folder, class_name)
                if not os.path.isdir(class_folder):
                    continue  # 跳过非目录项

                for file in os.listdir(class_folder):
                    file_path = os.path.join(class_folder, file)
                    if os.path.isfile(file_path):
                        image_paths.append(file_path)
                        if class_to_idx:
                            labels.append(class_to_idx[class_name])
                        else:
                            print(f"Warning: Class name {class_name} not found in class_to_idx mapping.")
                            labels.append(-1)  # 默认标签 -1

        elif dataset_type == "test":
            # 测试集无标签
            for file in os.listdir(image_folder):
                file_path = os.path.join(image_folder, file)
                image_paths.append(file_path)
                labels.append(-1)  # 测试集无标签

        # 创建 LMDB 数据库
        map_size = len(image_paths) * 3 * 64 * 64 * 2  # 预估 LMDB 大小
        env = lmdb.open(lmdb_path, map_size=map_size)

        with env.begin(write=True) as txn:
            for idx, (image_path, label) in enumerate(tqdm(zip(image_paths, labels), total=len(image_paths))):
                image = cv2.imread(image_path)
                if image is None:  # 检查是否成功读取图片
                    print(f"Error: Unable to read image {image_path}")
                    continue  # 跳过错误图片

                image = cv2.resize(image, (64, 64))  # 确保统一大小
                _, buffer = cv2.imencode('.jpg', image)
                txn.put(f"image-{idx:05d}".encode(), buffer.tobytes())
                if label != -1:
                    txn.put(f"label-{idx:05d}".encode(), np.array(label, dtype=np.int64).tobytes())
            txn.put(b"__len__", str(len(image_paths)).encode())

        print(f"{dataset_type.capitalize()} dataset converted to LMDB successfully.")
