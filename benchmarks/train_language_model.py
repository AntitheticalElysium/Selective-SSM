import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def create_optimizer(model, lr=6e-4, weight_decay=0.1):
    """Create AdamW optimizer with selective weight decay."""
    # Separate params into groups
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # No weight decay for biases, layer norms, and 1D parameters
        if param.ndim == 1 or "bias" in name or "ln" in name or "norm" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    # AdamW with betas from the paper
    optimizer = torch.optim.AdamW(optimizer_groups, lr=lr, betas=(0.9, 0.95))
    return optimizer


def create_scheduler(optimizer, warmup_steps, total_steps):
    """Create warmup + cosine decay scheduler."""

    def lr_lambda(step):
        if step < warmup_steps:
            # Linear warmup
            return float(step) / max(1, warmup_steps)
        else:
            # Cosine decay
            progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler


def train_step(model, data, target, optimizer, scheduler, criterion, device):
    """Single training step with gradient clipping.

    Args:
        model: The Mamba model
        data: Input sequences (B, seq_len, d_input)
        target: Target labels (B,)
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        criterion: Loss function
        device: Device to run on

    Returns:
        loss: Scalar loss value
        correct: Number of correct predictions
    """
    model.train()

    data = data.to(device)
    target = target.to(device)

    # Forward
    optimizer.zero_grad()
    logits = model(data)

    # Compute loss
    loss = criterion(logits, target)

    # Backward
    loss.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()
    scheduler.step()

    # Calculate accuracy
    pred = logits.argmax(dim=1)
    correct = pred.eq(target).sum().item()

    return loss.item(), correct


class MambaClassifier(nn.Module):
    """Wrapper to adapt Mamba for classification tasks."""

    def __init__(self, d_model=256, n_layers=4, num_classes=10):
        super().__init__()
        # Input projection for pixel values
        self.input_proj = nn.Linear(1, d_model)

        # Use MambaLayer from modules
        from src.selective_ssm.modules import MambaLayer

        self.layers = nn.ModuleList(
            [
                MambaLayer(d_model, d_state=16, d_conv=4, expand=2)
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

        # Classification head
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: (B, seq_len, 1) for MNIST pixels
        x = self.input_proj(x)  # (B, seq_len, d_model)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        # Global average pooling
        x = x.mean(dim=1)  # (B, d_model)
        x = self.classifier(x)  # (B, num_classes)
        return x


def main():
    """Train Mamba on Sequential MNIST classification."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparameters
    batch_size = 8  # Small batch for GPU memory
    epochs = 1  # Just 1 epoch for testing
    lr = 1e-3
    d_model = 64  # Small model
    n_layers = 2  # Fewer layers

    # Load Sequential MNIST dataset
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    print(
        f"Dataset loaded: {len(train_dataset)} train, {len(test_dataset)} test samples"
    )

    # Model
    model = MambaClassifier(d_model=d_model, n_layers=n_layers, num_classes=10)
    model = model.to(device)

    # Count params
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}\n")

    optimizer = create_optimizer(model, lr=lr, weight_decay=0.1)
    criterion = nn.CrossEntropyLoss()

    # Scheduler
    warmup_steps = 500
    total_steps = len(train_loader) * epochs
    scheduler = create_scheduler(
        optimizer, warmup_steps=warmup_steps, total_steps=total_steps
    )

    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            # Flatten images to sequences: (B, 1, 28, 28) -> (B, 784, 1)
            data = data.view(data.size(0), -1, 1)

            loss, batch_correct = train_step(
                model, data, target, optimizer, scheduler, criterion, device
            )

            train_loss += loss
            correct += batch_correct
            total += target.size(0)

            if (batch_idx + 1) % 100 == 0:
                accuracy = 100.0 * correct / total
                avg_loss = train_loss / (batch_idx + 1)
                lr_current = scheduler.get_last_lr()[0]
                print(
                    f"Epoch {epoch + 1}/{epochs} | "
                    f"Batch {batch_idx + 1}/{len(train_loader)} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"Acc: {accuracy:.2f}% | "
                    f"LR: {lr_current:.6f}"
                )

        # Evaluation
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in test_loader:
                data = data.view(data.size(0), -1, 1).to(device)
                target = target.to(device)

                output = model(data)
                test_loss += criterion(output, target).item()

                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)

        test_loss /= len(test_loader)
        test_accuracy = 100.0 * correct / total

        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch + 1}/{epochs} Summary:")
        print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_accuracy:.2f}%")
        print(f"{'=' * 60}\n")

    return model


if __name__ == "__main__":
    main()
