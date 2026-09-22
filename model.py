import torch
from torch import nn


class MNISTMLP(nn.Module):
    """Single-hidden-layer MNIST classifier: 784 -> hidden_size -> 10."""

    def __init__(self, hidden_size: int):
        super().__init__()

        if hidden_size < 1:
            raise ValueError("hidden_size must be at least 1")

        self.hidden_size = hidden_size
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 10),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.network(images)

    def count_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


if __name__ == "__main__":
    model = MNISTMLP(hidden_size=32)
    fake_images = torch.randn(8, 1, 28, 28)
    outputs = model(fake_images)

    print("Hidden size:", model.hidden_size)
    print("Input shape:", fake_images.shape)
    print("Output shape:", outputs.shape)
    print("Parameter count:", model.count_parameters())
