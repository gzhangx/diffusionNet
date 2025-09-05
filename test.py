import torch
import deepinv
from torchvision import datasets, transforms
import os

batch_size=64
image_size=32

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print("Using device:", device)

transform = transforms.Compose(
    [
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.0,), (1.0,)),
    ]
)

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root="/mnt/e/work/gpenv/traindata", train=True, download=True, transform=transform),
    batch_size=batch_size,
    shuffle=True,
)


print(f"Number of batches in train_loader: {len(train_loader)}")
# Or, to get the total number of samples:
print(f"Total number of samples: {len(train_loader.dataset)}")
