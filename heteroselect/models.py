from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LogisticRegression(nn.Module):
    """Flat-image multinomial logistic regression (InfoCom 2023 MNIST baseline)."""

    def __init__(self, in_features: int = 28 * 28, num_classes: int = 10) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.flatten(1))


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


class _BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut: nn.Module
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x), inplace=True)


class ResNet18(nn.Module):
    """ResNet-18 with a 3x3 stride-1 stem (no initial maxpool) for 64x64 input."""

    def __init__(self, num_classes: int = 200) -> None:
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 2, 1)
        self.layer2 = self._make_layer(128, 2, 2)
        self.layer3 = self._make_layer(256, 2, 2)
        self.layer4 = self._make_layer(512, 2, 2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers: list[nn.Module] = []
        for s in strides:
            layers.append(_BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.layer4(self.layer3(self.layer2(self.layer1(out))))
        return self.fc(self.avgpool(out).flatten(1))


def build_model(dataset: str, device: torch.device) -> nn.Module:
    if dataset == "mnist":
        model: nn.Module = LogisticRegression(in_features=28 * 28, num_classes=10)
    elif dataset == "cifar10":
        model = AlexNet(num_classes=10)
    elif dataset == "cifar100":
        model = ResNet9(num_classes=100)
    elif dataset == "tinyimagenet":
        model = ResNet18(num_classes=200)
    else:
        raise ValueError(
            f"Unknown dataset '{dataset}'. "
            f"Use 'mnist', 'cifar10', 'cifar100', or 'tinyimagenet'."
        )
    return model.to(device)


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def size_mb(model: nn.Module) -> float:
    return n_params(model) * 4 / 1e6
