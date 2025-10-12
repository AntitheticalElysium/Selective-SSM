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

    # L needs to be last => (B, D, L)
    u_transposed = u.transpose(1, 2)

    # If A_bar was complex, K_bar will also be complex
    if K_bar.dtype.is_complex:
        K_fft = fft.fft(K_bar, n=fft_len, dim=-1)
        # Convert real to complex for proper convolution
        u_complex = u_transposed.to(torch.cfloat)
        u_fft_full = fft.fft(u_complex, n=fft_len, dim=-1)
        # Convolution in freq domain: multiply FFTs element-wise
        y_fft = u_fft_full * K_fft.unsqueeze(0)
        # Inverse FFT to get convolution result
        y_transposed = fft.ifft(y_fft, n=fft_len, dim=-1)
        # Take real part since output should be real (need to check this?)
        if not u.dtype.is_complex:
            y_transposed = y_transposed.real
    else:
        # Use rfft for the real-valued kernel
        K_fft = fft.rfft(K_bar, n=fft_len, dim=-1)
        u_fft = fft.rfft(u_transposed, n=fft_len, dim=-1)

        y_fft = u_fft * K_fft.unsqueeze(0)

        y_transposed = fft.irfft(y_fft, n=fft_len, dim=-1)

    # Truncate to L
    y = y_transposed[..., :L]
    # Back to (B, L, D)
    y = y.transpose(1, 2)
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
