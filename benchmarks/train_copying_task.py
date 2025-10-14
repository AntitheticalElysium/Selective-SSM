import torch
import torch.nn as nn
import torch.optim as optim

from src.selective_ssm import S4


class CopyingModel(nn.Module):
    """Simple model with S4 layer for copying task."""

    def __init__(self, vocab_size, d_model, d_state):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.s4 = S4(d_model, d_state)
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)  # (B, L) -> (B, L, D)
        x = self.s4(x)  # (B, L, D) -> (B, L, D)
        x = self.output(x)  # (B, L, D) -> (B, L, vocab_size)
        return x


def generate_copying_data(batch_size, seq_len, delay, vocab_size):
    """
    Generate copying task data.

    Args:
        batch_size: Number of sequences
        seq_len: Length of sequence to copy
        delay: Number of blank tokens between input and output
        vocab_size: Size of vocabulary (excluding special tokens)

    Returns:
        inputs, targets: Tensors of shape (batch_size, total_len)
    """
    # Token IDs: 0 = blank, 1 = delimiter, 2+ = data tokens
    BLANK = 0
    DELIM = 1
    # +1 for delimiter
    total_len = seq_len + delay + seq_len + 1

    inputs = torch.zeros(batch_size, total_len, dtype=torch.long)
    targets = torch.full((batch_size, total_len), BLANK, dtype=torch.long)

    for i in range(batch_size):
        # Generate random sequence (tokens 2 to vocab_size+1)
        sequence = torch.randint(2, vocab_size + 2, (seq_len,))

        # Input: [sequence] [delimiter] [blanks]
        inputs[i, :seq_len] = sequence
        inputs[i, seq_len] = DELIM
        inputs[i, seq_len + 1 :] = BLANK

        # Target: [blanks] [sequence to copy]
        targets[i, : seq_len + delay + 1] = BLANK
        targets[i, seq_len + delay + 1 :] = sequence

    return inputs, targets


def train_copying_task(
    seq_len=10,
    delay=5,
    vocab_size=8,
    d_model=64,
    d_state=64,
    batch_size=32,
    num_batches=5000,
    lr=0.001,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # +2 for blank and delim
    model = CopyingModel(vocab_size + 2, d_model, d_state).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Ignore blank tokens
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Sequence length: {seq_len}, Delay: {delay}, Vocab size: {vocab_size}")
    print(f"Total sequence length: {seq_len + delay + seq_len + 1}\n")

    model.train()
    for batch_idx in range(num_batches):
        inputs, targets = generate_copying_data(batch_size, seq_len, delay, vocab_size)
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        # Reshape for loss
        loss = criterion(outputs.view(-1, vocab_size + 2), targets.view(-1))
        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % 50 == 0:
            # Calculate accuracy on relevant positions (where target != 0)
            preds = outputs.argmax(dim=-1)
            mask = targets != 0
            correct = (preds[mask] == targets[mask]).float().mean()

            print(
                f"Batch {batch_idx + 1}/{num_batches} | "
                f"Loss: {loss.item():.4f} | "
                f"Accuracy: {correct.item() * 100:.2f}%"
            )

    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)

    model.eval()
    with torch.no_grad():
        inputs, targets = generate_copying_data(1, seq_len, delay, vocab_size)
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        preds = outputs.argmax(dim=-1)

        print("\nExample:")
        print(f"Input:  {inputs[0].cpu().tolist()}")
        print(f"Target: {targets[0].cpu().tolist()}")
        print(f"Pred:   {preds[0].cpu().tolist()}")

        mask = targets != 0
        correct = (preds[mask] == targets[mask]).float().mean()
        print(f"\nFinal Accuracy: {correct.item() * 100:.2f}%")


if __name__ == "__main__":
    train_copying_task()
