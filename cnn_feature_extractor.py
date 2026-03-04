"""
Dependencies:
    pip install torch torchvision matplotlib
Run:
    python cnn_feature_extractor.py
Optional:
    python cnn_feature_extractor.py --epochs 2 --num-images 4 --maps-per-layer 4
"""

from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from cnn import SimpleCNN, get_data_loaders

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ModuleNotFoundError:
    HAS_MATPLOTLIB = False
    plt = None


MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Extract and visualize CNN feature maps.")
    parser.add_argument("--epochs", type=int, default=1, help="Quick training epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for quick training.")
    parser.add_argument(
        "--train-samples",
        type=int,
        default=5000,
        help="How many training samples to use for quick demo training.",
    )
    parser.add_argument("--num-images", type=int, default=4, help="How many test images to visualize.")
    parser.add_argument(
        "--maps-per-layer",
        type=int,
        default=4,
        help="How many strongest feature maps to show from each layer.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("outputs/cnn_feature_maps.png"),
        help="Path to save visualization image.",
    )
    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be >= 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    if args.train_samples < 1:
        parser.error("--train-samples must be >= 1")
    if args.num_images < 1:
        parser.error("--num-images must be >= 1")
    if args.maps_per_layer < 1:
        parser.error("--maps-per-layer must be >= 1")

    return args


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon MPS GPU")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using NVIDIA CUDA GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def quick_train(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int,
    max_samples: int,
) -> None:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    total_samples = min(max_samples, len(train_loader.dataset))
    subset_indices = list(range(total_samples))
    subset = Subset(train_loader.dataset, subset_indices)
    subset_loader = DataLoader(
        subset,
        batch_size=train_loader.batch_size,
        shuffle=True,
    )

    print(f"Quick training on {total_samples} samples for {epochs} epoch(s)...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in subset_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        avg_loss = running_loss / total_samples
        print(f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f}")


@torch.no_grad()
def capture_feature_maps(
    model: nn.Module,
    images: torch.Tensor,
    device: torch.device,
    layer_names: list[str],
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    feature_maps: dict[str, torch.Tensor] = {}
    hooks = []
    modules = dict(model.named_modules())

    for layer_name in layer_names:
        if layer_name not in modules:
            raise ValueError(f"Layer '{layer_name}' not found in model.")

        def save_output(_module, _inputs, output, key: str = layer_name) -> None:
            feature_maps[key] = output.detach().cpu()

        hooks.append(modules[layer_name].register_forward_hook(save_output))

    model.eval()
    logits = model(images.to(device))
    predictions = logits.argmax(dim=1).cpu()

    for hook in hooks:
        hook.remove()

    return feature_maps, predictions


def denormalize_mnist(image: torch.Tensor) -> torch.Tensor:
    image = image * MNIST_STD + MNIST_MEAN
    return image.clamp(0.0, 1.0)


def normalize_map(feature_map: torch.Tensor) -> torch.Tensor:
    fm = feature_map - feature_map.min()
    denom = fm.max().item()
    if denom > 0:
        fm = fm / denom
    return fm


def plot_feature_maps(
    images: torch.Tensor,
    labels: torch.Tensor,
    predictions: torch.Tensor,
    feature_maps: dict[str, torch.Tensor],
    layer_names: list[str],
    maps_per_layer: int,
    save_path: Path,
) -> None:
    if not HAS_MATPLOTLIB:
        raise RuntimeError("matplotlib is not installed. Install with: pip install matplotlib")

    layer_titles = {
        "features.0": "Conv1",
        "features.3": "Conv2",
    }

    num_images = images.size(0)
    cols = 1 + len(layer_names) * maps_per_layer
    fig, axes = plt.subplots(num_images, cols, figsize=(cols * 2.2, num_images * 2.2))

    if num_images == 1:
        axes = [axes]

    for image_idx in range(num_images):
        row_axes = axes[image_idx]

        original = denormalize_mnist(images[image_idx].squeeze(0))
        row_axes[0].imshow(original.numpy(), cmap="gray")
        row_axes[0].set_title(
            f"Original\nT:{labels[image_idx].item()} P:{predictions[image_idx].item()}",
            fontsize=9,
        )
        row_axes[0].axis("off")

        col_idx = 1
        for layer_name in layer_names:
            layer_features = feature_maps[layer_name][image_idx]  # (channels, h, w)
            strengths = layer_features.mean(dim=(1, 2))
            top_k = min(maps_per_layer, layer_features.size(0))
            top_channels = torch.topk(strengths, k=top_k).indices.tolist()

            for channel in range(maps_per_layer):
                ax = row_axes[col_idx]
                if channel < len(top_channels):
                    channel_id = top_channels[channel]
                    fmap = normalize_map(layer_features[channel_id])
                    ax.imshow(fmap.numpy(), cmap="viridis")
                    layer_label = layer_titles.get(layer_name, layer_name)
                    ax.set_title(f"{layer_label}\nch {channel_id}", fontsize=8)
                ax.axis("off")
                col_idx += 1

    fig.suptitle("CNN Feature Extraction: Original vs Detected Features", fontsize=12)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = get_device()
    torch.manual_seed(42)

    train_loader, test_loader = get_data_loaders(batch_size=args.batch_size)
    model = SimpleCNN().to(device)

    quick_train(
        model=model,
        train_loader=train_loader,
        device=device,
        epochs=args.epochs,
        max_samples=args.train_samples,
    )

    test_images, test_labels = next(iter(test_loader))
    num_images = min(args.num_images, test_images.size(0))
    images = test_images[:num_images]
    labels = test_labels[:num_images]

    layer_names = ["features.0", "features.3"]  # conv1 and conv2
    feature_maps, predictions = capture_feature_maps(
        model=model,
        images=images,
        device=device,
        layer_names=layer_names,
    )

    plot_feature_maps(
        images=images,
        labels=labels,
        predictions=predictions,
        feature_maps=feature_maps,
        layer_names=layer_names,
        maps_per_layer=args.maps_per_layer,
        save_path=args.output_path,
    )

    print(f"Saved feature map visualization to: {args.output_path}")


if __name__ == "__main__":
    main()
