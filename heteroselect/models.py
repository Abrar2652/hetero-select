from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,   64,  3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64,  192, 3, padding=1),           nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(192, 384, 3, padding=1),           nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, 3, padding=1),           nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),           nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(torch.flatten(self.features(x), 1))


def _cbr(in_c: int, out_c: int, pool: bool = False) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Conv2d(in_c, out_c, 3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class _ResBlock(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.block(x) + x)


class ResNet9(nn.Module):
    def __init__(self, num_classes: int = 100) -> None:
        super().__init__()
        self.prep = _cbr(3, 64)
        self.layer1 = nn.Sequential(_cbr(64, 128, pool=True), _ResBlock(128))
        self.layer2 = _cbr(128, 256, pool=True)
        self.layer3 = nn.Sequential(_cbr(256, 512, pool=True), _ResBlock(512))
        self.head = nn.Sequential(
            nn.AdaptiveMaxPool2d(1), nn.Flatten(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.layer3(self.layer2(self.layer1(self.prep(x)))))


def build_model(dataset: str, device: torch.device) -> nn.Module:
    if dataset == "cifar10":
        model: nn.Module = AlexNet(num_classes=10)
    elif dataset == "cifar100":
        model = ResNet9(num_classes=100)
    else:
        raise ValueError(f"Unknown dataset '{dataset}'. Use 'cifar10' or 'cifar100'.")
    return model.to(device)


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def size_mb(model: nn.Module) -> float:
    return n_params(model) * 4 / 1e6
