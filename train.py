import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from data_utils import get_data_loaders
from model import MNISTMLP


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def select_device(device_name: str) -> torch.device:
    if device_name == "cpu":
        return torch.device("cpu")

    if device_name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
        return torch.device("mps")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def train_one_epoch(model, data_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct_count = 0
    sample_count = 0

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        correct_count += (outputs.argmax(dim=1) == labels).sum().item()
        sample_count += batch_size

    return (
        total_loss / sample_count,
        100.0 * correct_count / sample_count,
    )


@torch.inference_mode()
def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct_count = 0
    sample_count = 0

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        correct_count += (outputs.argmax(dim=1) == labels).sum().item()
        sample_count += batch_size

    return (
        total_loss / sample_count,
        100.0 * correct_count / sample_count,
    )


def run_experiment(
    hidden_size: int,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    patience: int,
    device_name: str,
):
    set_random_seed(seed)
    device = select_device(device_name)

    print(f"Device: {device}")
    print(f"Hidden size: {hidden_size}")
    print(f"Seed: {seed}")

    train_loader, val_loader, test_loader = get_data_loaders(
        batch_size=batch_size,
        split_seed=2026,
        loader_seed=seed,
    )

    model = MNISTMLP(hidden_size=hidden_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    checkpoint_directory = Path("checkpoints")
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_directory / f"mlp_h{hidden_size}_seed{seed}.pt"

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    epochs_completed = 0

    start_time = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epochs_completed = epoch

        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_accuracy = evaluate(
            model, val_loader, criterion, device
        )

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"train loss {train_loss:.4f} | "
            f"train accuracy {train_accuracy:.2f}% | "
            f"val loss {val_loss:.4f} | "
            f"val accuracy {val_accuracy:.2f}%"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state": model.state_dict(),
                    "hidden_size": hidden_size,
                    "seed": seed,
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val_loss,
                    "val_accuracy": val_accuracy,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping after {patience} unimproved epochs")
            break

    training_seconds = time.perf_counter() - start_time

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state"])
    model.to(device)

    test_loss, test_accuracy = evaluate(
        model, test_loader, criterion, device
    )

    parameter_count = model.count_parameters()
    approximate_macs = 794 * hidden_size
    best_val_accuracy = float(checkpoint["val_accuracy"])

    print("\nTraining complete")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation accuracy: {best_val_accuracy:.2f}%")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.2f}%")
    print(f"Parameters: {parameter_count}")
    print(f"Approximate MACs per sample: {approximate_macs}")
    print(f"Training time: {training_seconds:.2f} seconds")
    print(f"Checkpoint: {checkpoint_path}")

    return {
        "hidden_size": hidden_size,
        "seed": seed,
        "best_epoch": best_epoch,
        "epochs_completed": epochs_completed,
        "best_val_loss": best_val_loss,
        "best_val_accuracy": best_val_accuracy,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "parameter_count": parameter_count,
        "macs": approximate_macs,
        "training_seconds": training_seconds,
        "checkpoint_path": str(checkpoint_path),
    }


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train an MNIST MLP with a configurable hidden width"
    )
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "mps"],
        default="auto",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    run_experiment(
        hidden_size=args.hidden_size,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
        device_name=args.device,
    )
