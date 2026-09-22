import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def get_data_loaders(
    batch_size: int = 128,
    split_seed: int = 2026,
    loader_seed: int = 42,
):
    """Download MNIST and return fixed train, validation, and test loaders."""

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    full_train_dataset = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=transform,
    )

    split_generator = torch.Generator().manual_seed(split_seed)

    train_dataset, val_dataset = random_split(
        full_train_dataset,
        lengths=[54_000, 6_000],
        generator=split_generator,
    )

    loader_generator = torch.Generator().manual_seed(loader_seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=loader_generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_data_loaders()
    images, labels = next(iter(train_loader))

    print("Train samples:", len(train_loader.dataset))
    print("Validation samples:", len(val_loader.dataset))
    print("Test samples:", len(test_loader.dataset))
    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)
