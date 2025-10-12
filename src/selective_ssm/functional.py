import torch
import torch.fft as fft


def discretize_zoh(delta, A, B):
    """
    Discretize continuous-time SSM parameters (A, B) using the Zero-Order Hold method.

    Args:
        delta (torch.Tensor): The discretization timestep, shape (D,).
        A (torch.Tensor): The state matrix, shape (D, N).
        B (torch.Tensor): The input matrix, shape (D, N).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the discretized
        state matrix A_bar (D, N) and input matrix B_bar (D, N).
    """
    # Unsqueeze to broadcast across N
    delta = delta.unsqueeze(-1)
    delta_A = A * delta

    A_bar = torch.exp(delta_A)
    B_bar = (A_bar - 1) / A * B
    return A_bar, B_bar


def ssm_convolutional(u, A_bar, B_bar, C):
    """
    Compute the SSM output in parallel using FFT-based convolution.

    Args:
        u (torch.Tensor): The input sequence, shape (B, L, D).
        A_bar (torch.Tensor): The discretized state matrix, shape (D, N).
        B_bar (torch.Tensor): The discretized input matrix, shape (D, N).
        C (torch.Tensor): The output matrix, shape (D, N).

    Returns:
        torch.Tensor: The output sequence y, shape (B, L, D).
    """
    _, L, _ = u.shape
    # Compute the SSM convolution kernel K_bar
    # A_powers stores A_bar^0, A_bar^1, ..., A_bar^(L-1)
    A_powers = torch.zeros(L, *A_bar.shape, device=u.device, dtype=A_bar.dtype)
    A_powers[0] = torch.ones_like(A_bar)
    for k in range(1, L):
        A_powers[k] = A_powers[k - 1] * A_bar

    # Sum over state dim N: (1, D, N) * (L, D, N) * (1, D, N) => (L, D)
    K_bar = (C.unsqueeze(0) * A_powers * B_bar.unsqueeze(0)).sum(dim=-1)
    K_bar = K_bar.T

    # Power of 2 is more efficient for FFT but 2 * L  is more simple for now
    fft_len = 2 * L
    # Apply FFT to the kernel and the input (rfft for real vals)
    K_fft = fft.rfft(K_bar, n=fft_len, dim=-1)

    # L needs to be last => (B, D, L)
    u_transposed = u.transpose(1, 2)
    u_fft = fft.rfft(u_transposed, n=fft_len, dim=-1)

    # K_fft needs a batch dim to multiply with u_fft => (B, D, L_fft)
    y_fft = u_fft * K_fft.unsqueeze(0)
    # Inverse step
    y_transposed = fft.irfft(y_fft, n=fft_len, dim=-1)

    # Truncate to L
    y = y_transposed[..., :L]
    # Back to (B, L, D)
    y = y.transpose(1, 2)

    if y.is_complex():
        y = y.real
    return y


def ssm_recurrent(u, A_bar, B_bar, C):
    """
    Compute the SSM output sequentially using a recurrent loop.

    Args:
        u (torch.Tensor): The input sequence, shape (B, L, D).
        A_bar (torch.Tensor): The discretized state matrix, shape (D, N).
        B_bar (torch.Tensor): The discretized input matrix, shape (D, N).
        C (torch.Tensor): The output matrix, shape (D, N).

    Returns:
        torch.Tensor: The output sequence y, shape (B, L, D).
    """
    B, L, D = u.shape
    N = A_bar.shape[-1]
    # Hidden state with batch, channel and state dimensions
    h = torch.zeros(
        B,
        D,
        N,
        device=u.device,
        dtype=torch.cfloat if A_bar.is_complex() else torch.float,
    )

    outputs = []
    for t in range(L):
        # Input at current timestep
        x_t = u[:, t, :]
        # Update hidden state using recurrence relation
        h = A_bar * h + B_bar * x_t.unsqueeze(-1)
        # Current output for step
        y_t = (C * h).sum(dim=-1)

        # If params are complex, we'll need the real part of sum
        if y_t.is_complex():
            y_t = y_t.real
        outputs.append(y_t)

    # Stack along L to create (B, L, D) from (B, D)
    y = torch.stack(outputs, dim=1)
    return y
