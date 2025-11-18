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
    B_bar = (A_bar - 1) / delta_A * delta * B
    return A_bar, B_bar


def discretize_zoh_selective(delta, A, B):
    """
    Discretize continuous-time SSM parameters for selective (time-varying) SSMs.

    Args:
        delta (torch.Tensor): The discretization timestep, shape (B, D, 1).
        A (torch.Tensor): The state matrix, shape (D, N).
        B (torch.Tensor): The input matrix, shape (B, D, N).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - A_bar: shape (B, D, N)
            - B_bar: shape (B, D, N)
    """
    # A needs batch dimension for broadcasting
    delta_A = delta * A.unsqueeze(0)
    A_bar = torch.exp(delta_A)
    B_bar = (A_bar - 1) / delta_A * delta * B

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
    A_powers = []
    A_powers.append(torch.ones_like(A_bar))
    for k in range(1, L):
        A_powers.append(A_powers[k - 1] * A_bar)
    A_powers = torch.stack(A_powers, dim=0)

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
        # Update hidden state using recurrence relation (no in-place for autograd)
        h_new = A_bar * h + B_bar * x_t.unsqueeze(-1)
        h = h_new
        # Current output for step
        y_t = (C * h).sum(dim=-1)
        # If params are complex, we'll need the real part of sum
        if y_t.is_complex():
            y_t = y_t.real
        outputs.append(y_t)

    # Stack along L to create (B, L, D) from (B, D)
    y = torch.stack(outputs, dim=1)
    return y


def ssm_binary_operator(elem_1, elem_2):
    """
    Binary associative operator for SSM parallel scan.

    Args:
        elem_1: Tuple of (A_1, b_1) where:
            A_1: torch.Tensor of shape (B, D, N) - Left transition matrix
            b_1: torch.Tensor of shape (B, D, N) - Left accumulated input
        elem_2: Tuple of (A_2, b_2) where:
            A_2: torch.Tensor of shape (B, D, N) - Right transition matrix
            b_2: torch.Tensor of shape (B, D, N) - Right accumulated input

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Combined (A_combined, b_combined)
            A_combined: (B, D, N) - Product A_2 * A_1
            b_combined: (B, D, N) - A_2 * b_1 + b_2

    """
    A_1, b_1 = elem_1
    A_2, b_2 = elem_2

    A_combined = A_2 * A_1
    b_combined = A_2 * b_1 + b_2

    return A_combined, b_combined


def naive_sequential_scan(A_bar, B_bar_x, initial_state=None):
    """
    Naive sequential scan for SSM - uses binary operator in a simple loop.
    Reference implementation for testing correctness- not optimized yet...

    Args:
        A_bar: torch.Tensor of shape (B, L, D, N)
            Discretized state transition matrices for all timesteps
        B_bar_x: torch.Tensor of shape (B, L, D, N)
            Precomputed B_bar * x for all timesteps (input effect)
        initial_state: torch.Tensor of shape (B, D, N), optional
            Initial hidden state. If None, uses zeros.

    Returns:
        torch.Tensor of shape (B, L, D, N)
            Hidden states h_t for all timesteps t=1..L
    """
    B, L, D, N = A_bar.shape
    if initial_state is None:
        initial_state = torch.zeros(B, D, N, device=A_bar.device, dtype=A_bar.dtype)

    # Start with identity element
    A_accumulated = torch.ones(B, D, N, device=A_bar.device, dtype=A_bar.dtype)
    h_accumulated = initial_state.clone()

    results = []

    for t in range(L):
        # Get transition for this timestep
        A_t = A_bar[:, t, :, :]  # (B, D, N)
        b_t = B_bar_x[:, t, :, :]  # same

        A_accumulated, h_accumulated = ssm_binary_operator(
            # Previous accumulation (apply first)
            (A_accumulated, h_accumulated),
            # Current timestep (apply second)
            (A_t, b_t),
        )

        # The second component is our hidden state h_t
        results.append(h_accumulated.clone())

    h_all = torch.stack(results, dim=1)
    return h_all


def blelloch_parallel_scan(A_bar, B_bar_x, initial_state=None):
    """
    Work-efficient parallel scan for SSM using a modified Blelloch's algorithm.

    Args:
        A_bar: torch.Tensor of shape (B, L, D, N)
            Discretized state transition matrices for all timesteps
        B_bar_x: torch.Tensor of shape (B, L, D, N)
            Precomputed B_bar * x for all timesteps (input effect)
        initial_state: torch.Tensor of shape (B, D, N), optional
            Initial hidden state. If None, uses zeros.

    Returns:
        torch.Tensor of shape (B, L, D, N)
            Hidden states h_t for all timesteps t=1..L
    """
    B, L, D, N = A_bar.shape
    device = A_bar.device
    dtype = A_bar.dtype

    if initial_state is None:
        initial_state = torch.zeros(B, D, N, device=device, dtype=dtype)

    # Pad to next power of 2
    L_original = L
    L_pow2 = 1
    while L_pow2 < L:
        L_pow2 *= 2

    if L_pow2 > L_original:
        # Pad with identity elements
        pad_width = L_pow2 - L_original
        A_identity = torch.ones(B, pad_width, D, N, device=device, dtype=dtype)
        b_identity = torch.zeros(B, pad_width, D, N, device=device, dtype=dtype)

        A_work = torch.cat([A_bar, A_identity], dim=1)
        b_work = torch.cat([B_bar_x, b_identity], dim=1)
    else:
        A_work = A_bar.clone()
        b_work = B_bar_x.clone()

    # Iterative doubling for inclusive scan
    offset = 1
    while offset < L_pow2:
        # Create indices for elements that will be updated
        indices = torch.arange(offset, L_pow2, dtype=torch.long, device=device)
        prev_indices = indices - offset

        # Get elements to combine
        A_prev = A_work[:, prev_indices]
        b_prev = b_work[:, prev_indices]
        A_curr = A_work[:, indices].clone()
        b_curr = b_work[:, indices].clone()

        # Apply binary operator
        A_combined, b_combined = ssm_binary_operator((A_prev, b_prev), (A_curr, b_curr))

        # Update current elements
        A_work[:, indices] = A_combined
        b_work[:, indices] = b_combined

        offset *= 2

    # Remove padding
    if L_pow2 > L_original:
        A_work = A_work[:, :L_original]
        b_work = b_work[:, :L_original]

    # Apply initial state:
    h_all = A_work * initial_state.unsqueeze(1) + b_work
    return h_all
