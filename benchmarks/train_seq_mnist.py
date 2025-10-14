import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.selective_ssm import S4


class SequentialMNIST(nn.Module):
    """S4 model for Sequential MNIST classification."""

    def __init__(self, d_model, d_state, n_layers, dropout=0.1):
        super().__init__()
        # Project input pixels to d_model
        self.encoder = nn.Linear(1, d_model)
        # Stack of S4 layers
        self.layers = nn.ModuleList(
            [S4Layer(d_model, d_state, dropout) for _ in range(n_layers)]
        )
        # Classification head
        self.norm = nn.LayerNorm(d_model)
        self.decoder = nn.Linear(d_model, 10)

    def forward(self, x):
        # x: (B, 784, 1) - flattened MNIST images
        x = self.encoder(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        # Pool over sequence length => take mean
        x = x.mean(dim=1)
        x = self.decoder(x)
        return x


class S4Layer(nn.Module):
    """Single S4 layer with residual connection and normalization."""

    def __init__(self, d_model, d_state, dropout=0.1):
        super().__init__()
        self.s4 = S4(d_model, d_state)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # S4 with residual connection
        z = self.s4(x)
        z = self.dropout(z)
        x = x + z
        x = self.norm(x)
        return x


def train_mnist(d_model, d_state, n_layers, batch_size, epochs, lr):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print(f"Model: d_model={d_model}, d_state={d_state}, n_layers={n_layers}")

    # Load MNIST dataset
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            # MNIST mean and std
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    model = SequentialMNIST(d_model, d_state, n_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            # Flatten images to sequences: (B, 1, 28, 28) -> (B, 784, 1)
            data = data.view(data.size(0), -1, 1).to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

            if (batch_idx + 1) % 100 == 0:
                accuracy = 100.0 * correct / total
                avg_loss = train_loss / (batch_idx + 1)
                print(
                    f"Epoch {epoch + 1}/{epochs} | "
                    f"Batch {batch_idx + 1}/{len(train_loader)} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"Acc: {accuracy:.2f}%"
                )

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
    train_mnist(256, 128, 4, 64, 5, 0.001)
