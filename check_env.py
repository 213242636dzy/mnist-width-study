import sys

import torch


print("Python path:", sys.executable)
print("PyTorch version:", torch.__version__)

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Selected device:", device)

x = torch.arange(1, 6, dtype=torch.float32, device=device)
y = x**2

print("Input tensor:", x.cpu())
print("Squared tensor:", y.cpu())
print("Environment check passed")
