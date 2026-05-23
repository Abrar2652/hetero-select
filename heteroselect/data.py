from __future__ import annotations

import os
from typing import List, Sequence, Tuple

import numpy as np
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset


_NORM = {
    "mnist":        ((0.1307,),                 (0.3081,)),
    "cifar10":      ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    "cifar100":     ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    "tinyimagenet": ((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)),
}


class _TinyImageNetVal(Dataset):
    """TinyImageNet val split (flat `images/` dir + `val_annotations.txt`)."""

    def __init__(self, val_root: str, class_to_idx: dict, transform=None) -> None:
        ann_path = os.path.join(val_root, "val_annotations.txt")
        img_dir = os.path.join(val_root, "images")
        if not os.path.isfile(ann_path):
            raise FileNotFoundError(f"Missing {ann_path}")
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []
        with open(ann_path, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                fname, cls = parts[0], parts[1]
                if cls not in class_to_idx:
                    continue
                self.samples.append((os.path.join(img_dir, fname), class_to_idx[cls]))
        self.targets = [t for _, t in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, target = self.samples[idx]
        with open(path, "rb") as f:
            img = Image.open(f).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, target


def _load_tinyimagenet(data_root: str) -> Tuple[Dataset, Dataset]:
    mean, std = _NORM["tinyimagenet"]
    train_tf = transforms.Compose([
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    tin_root = os.path.join(data_root, "tiny-imagenet-200")
    if not os.path.isdir(tin_root):
        raise FileNotFoundError(
            f"TinyImageNet not found at {tin_root!r}. "
            f"Download http://cs231n.stanford.edu/tiny-imagenet-200.zip "
            f"and extract under {data_root}/ so that {tin_root}/train and "
            f"{tin_root}/val exist."
        )
    train_dir = os.path.join(tin_root, "train")
    val_dir = os.path.join(tin_root, "val")
    train = torchvision.datasets.ImageFolder(train_dir, transform=train_tf)
    test = _TinyImageNetVal(val_dir, train.class_to_idx, transform=test_tf)
    return train, test


def load_data(dataset: str, data_root: str = "./data") -> Tuple[Dataset, Dataset]:
    if dataset == "tinyimagenet":
        return _load_tinyimagenet(data_root)
    if dataset not in _NORM:
        raise ValueError(
            f"Unknown dataset {dataset!r}. "
            f"Use 'mnist', 'cifar10', 'cifar100', or 'tinyimagenet'."
        )
    mean, std = _NORM[dataset]
    if dataset == "mnist":
        train_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_tf = train_tf
        train = torchvision.datasets.MNIST(
            data_root, train=True,  download=True, transform=train_tf,
        )
        test = torchvision.datasets.MNIST(
            data_root, train=False, download=True, transform=test_tf,
        )
        return train, test
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    cls = (
        torchvision.datasets.CIFAR10
        if dataset == "cifar10"
        else torchvision.datasets.CIFAR100
    )
    train = cls(data_root, train=True,  download=True, transform=train_tf)
    test  = cls(data_root, train=False, download=True, transform=test_tf)
    return train, test


def _dataset_labels(train_ds: Dataset) -> np.ndarray:
    targets = getattr(train_ds, "targets", None)
    if targets is not None:
        return np.asarray(targets)
    return np.array([train_ds[i][1] for i in range(len(train_ds))])


def partition_data(
    train_ds: Dataset,
    cfg: dict,
    num_clients: int,
    seed: int,
) -> List[List[int]]:
    labels = _dataset_labels(train_ds)
    psi = cfg["psi"]
    rng = np.random.RandomState(seed)
    dataset = cfg["dataset"]

    if dataset in ("cifar10", "mnist"):
        num_classes = int(labels.max()) + 1
        n_total = len(labels)
        n_per = n_total // num_clients
        class_idx = {c: list(np.where(labels == c)[0]) for c in range(num_classes)}
        for c in range(num_classes):
            rng.shuffle(class_idx[c])
        class_ptr = {c: 0 for c in range(num_classes)}

        def draw(cls: int, n: int) -> Sequence[int]:
            pool = class_idx[cls]
            p = class_ptr[cls]
            chunk = pool[p:p + n]
            class_ptr[cls] += n
            return chunk

        clients: List[List[int]] = []
        for _ in range(num_clients):
            dominant = rng.randint(0, num_classes)
            n_dom = int(psi * n_per)
            n_oth = n_per - n_dom
            others = [c for c in range(num_classes) if c != dominant]
            n_each = max(1, n_oth // len(others))
            idx = list(draw(dominant, n_dom))
            for c in others:
                idx += list(draw(c, n_each))
            clients.append(
                idx if idx else rng.choice(n_total, 10, replace=False).tolist()
            )
        return clients

    if dataset in ("cifar100", "tinyimagenet"):
        num_classes = int(labels.max()) + 1
        n_missing = int(psi) if psi >= 1 else int(psi * num_classes)
        n_missing = max(0, min(n_missing, num_classes - 1))
        clients = [[] for _ in range(num_clients)]
        for k in range(num_clients):
            missing = set(rng.choice(num_classes, n_missing, replace=False).tolist())
            for c in range(num_classes):
                if c in missing:
                    continue
                idx = np.where(labels == c)[0]
                if idx.size == 0:
                    continue
                n = max(1, len(idx) // num_clients)
                clients[k].extend(
                    rng.choice(idx, min(n, len(idx)), replace=False).tolist()
                )
        pool = list(range(len(labels)))
        for k in range(num_clients):
            if not clients[k]:
                clients[k] = rng.choice(pool, 10, replace=False).tolist()
        return clients

    raise ValueError(f"Unknown dataset '{dataset}'.")


def make_loaders(
    train_ds: Dataset,
    client_idx: Sequence[Sequence[int]],
    batch_size: int,
) -> List[DataLoader]:
    return [
        DataLoader(
            Subset(train_ds, list(idx)),
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=0,
            pin_memory=True,
        )
        for idx in client_idx
    ]
