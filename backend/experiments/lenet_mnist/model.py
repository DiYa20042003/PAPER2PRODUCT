"""LeNet-5-style convolutional neural network for MNIST."""

import torch
from torch import nn


MODEL_SPEC = {
    "name": "LeNet-5-style variant",
    "input_shape": [1, 28, 28],
    "layers": [
        "Conv2d(1, 6, kernel_size=5) + ReLU",
        "AvgPool2d(kernel_size=2, stride=2)",
        "Conv2d(6, 16, kernel_size=5) + ReLU",
        "AvgPool2d(kernel_size=2, stride=2)",
        "Flatten",
        "Linear(256, 120) + ReLU",
        "Linear(120, 84) + ReLU",
        "Linear(84, 10)",
    ],
}


class LeNet5(nn.Module):
    """A compact LeNet-5 variant accepting 28x28 grayscale images."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))
