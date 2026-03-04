"""
Dependencies:
    pip install torch torchvision matplotlib
    
Run:
    python cnn.py
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ModuleNotFoundError:
    HAS_MATPLOTLIB = False
    plt = None


class SimpleCNN(nn.Module):
    """A very small CNN for 28x28 grayscale images (MNIST)."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # (B, 1, 28, 28) -> (B, 16, 28, 28)
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> (B, 16, 14, 14)
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # -> (B, 32, 14, 14)
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> (B, 32, 7, 7)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),  # 10 classes: digits 0-9
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def get_data_loaders(batch_size: int = 64) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(), # Converts PIL images (0-255) to tensors and scales to [0, 1]
            transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean/std, This centers data around 0 with unit variance, helping training converge faster
        ]
    )

    train_dataset = datasets.MNIST(
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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return correct / total


def plot_training_curves(
    train_losses: list[float], test_accuracies: list[float], save_path: Path
) -> None:
    if not HAS_MATPLOTLIB:
        raise RuntimeError("matplotlib is not installed.")

    epochs = list(range(1, len(train_losses) + 1))
    test_accuracies_percent = [acc * 100 for acc in test_accuracies]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(epochs, train_losses, marker="o", color="tab:blue")
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, test_accuracies_percent, marker="o", color="tab:green")
    axes[1].set_title("Test Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def plot_sample_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    save_path: Path,
    num_images: int = 12,
) -> None:
    if not HAS_MATPLOTLIB:
        raise RuntimeError("matplotlib is not installed.")

    model.eval()
    images, labels = next(iter(loader))
    outputs = model(images.to(device))
    predictions = outputs.argmax(dim=1).cpu()

    num_images = min(num_images, images.size(0))
    cols = 4
    rows = (num_images + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes_list = list(axes.flat) if hasattr(axes, "flat") else [axes]

    for i in range(num_images):
        image = images[i].squeeze(0) * 0.3081 + 0.1307
        image = image.clamp(0.0, 1.0)
        true_label = labels[i].item()
        pred_label = predictions[i].item()
        color = "green" if true_label == pred_label else "red"

        axes_list[i].imshow(image.numpy(), cmap="gray")
        axes_list[i].set_title(f"T:{true_label} P:{pred_label}", color=color, fontsize=10)
        axes_list[i].axis("off")

    for i in range(num_images, len(axes_list)):
        axes_list[i].axis("off")

    fig.suptitle("MNIST Sample Predictions", fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
   
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon MPS GPU 🚀")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using NVIDIA CUDA GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU (no GPU available)")
    
    print(f"Device: {device}")

    train_loader, test_loader = get_data_loaders(batch_size=64)
    model = SimpleCNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 2
    train_losses: list[float] = []
    test_accuracies: list[float] = []
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        accuracy = evaluate(model, test_loader, device)
        train_losses.append(train_loss)
        test_accuracies.append(accuracy)
        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Test Accuracy: {accuracy * 100:.2f}%"
        )

    if HAS_MATPLOTLIB:
        output_dir = Path("outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        training_plot_path = output_dir / "training_metrics.png"
        prediction_plot_path = output_dir / "sample_predictions.png"

        plot_training_curves(train_losses, test_accuracies, training_plot_path)
        plot_sample_predictions(model, test_loader, device, prediction_plot_path)

        print(f"Saved training plot to: {training_plot_path}")
        print(f"Saved prediction plot to: {prediction_plot_path}")
    else:
        print("Skipping plots because matplotlib is not installed.")

    print("Training complete.")


if __name__ == "__main__":
    main()
