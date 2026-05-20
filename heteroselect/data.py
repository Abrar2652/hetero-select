"""Dataset loading and non-IID client partitioning.

The partition rules below match the FedCG protocol exactly:

    * CIFAR-10  : psi-LDA partition with controllable concentration ``psi``
    * CIFAR-100 : skewed-label partition where each client is missing
                  ``psi`` randomly chosen classes
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset


_NORM = {
    "cifar10":  ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    "cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
}


def load_data(dataset: str, data_root: str = "./data") -> Tuple[Dataset, Dataset]:
    """Download (if necessary) and return the ``(train, test)`` datasets."""
    if dataset not in _NORM:
        raise ValueError(f"Unknown dataset '{dataset}'. Use 'cifar10' or 'cifar100'.")
    mean, std = _NORM[dataset]
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


def partition_data(
    train_ds: Dataset,
    cfg: dict,
    num_clients: int,
    seed: int,
) -> List[List[int]]:
    """Split ``train_ds`` into ``num_clients`` non-IID shards.

    For CIFAR-10 each client has a dominant class with mass ``psi`` and the
    remainder filled uniformly from the other classes.  For CIFAR-100 each
    client is missing ``psi`` random classes.
    """
    labels = np.array([train_ds[i][1] for i in range(len(train_ds))])
    psi = cfg["psi"]
    rng = np.random.RandomState(seed)
    dataset = cfg["dataset"]

    if dataset == "cifar10":
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

    if dataset == "cifar100":
        num_classes = int(labels.max()) + 1
        clients = [[] for _ in range(num_clients)]
        for k in range(num_clients):
            missing = set(rng.choice(num_classes, int(psi), replace=False).tolist())
            for c in range(num_classes):
                if c in missing:
                    continue
                idx = np.where(labels == c)[0]
                n = max(1, len(idx) // num_clients)
                clients[k].extend(rng.choice(idx, n, replace=False).tolist())
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
    """Wrap each client's index list in a shuffling DataLoader."""
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
