import torch
import torch.nn as nn

from . import functional as F


class S4(nn.Module):
    """
    The non-selective S4 layer, baseline SSM architecture (Linear Time Invariant).
    """

    def __init__(self, d_model, d_state, **kwargs):
        """
        Args:
            d_model (int): The dimension of the input and output (D).
            d_state (int): The dimension of the hidden state (N).
        """
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # Trainable params, A is initialized as complex
        # Also has real part with range (0.5 to -1.0) for stability
        # Maybe implem HiPPO matrices later?
        A_real = -0.5 - 0.5 * torch.rand(d_model, d_state)
        A_imag = torch.randn(d_model, d_state)
        self.A = nn.Parameter(torch.complex(A_real, A_imag))

        self.B = nn.Parameter(torch.randn(d_model, d_state))
        self.C = nn.Parameter(torch.randn(d_model, d_state))
        # Delta controls discretization, larger = more signal propagation
        self.log_delta = nn.Parameter(torch.log(0.1 * torch.ones(d_model)))

    def forward(self, u):
        """
        Args:
            u (torch.Tensor): The input sequence, shape (B, L, D).

        Returns:
            torch.Tensor: The output sequence y, shape (B, L, D).
        """
        # log_delta to delta with softplus for positivity
        delta = torch.nn.functional.softplus(self.log_delta)
        A_bar, B_bar = F.discretize_zoh(delta, self.A, self.B)

        if self.training:
            # Use more efficient convolutional mode
            y = F.ssm_convolutional(u, A_bar, B_bar, self.C)
        else:
            # Autoregressive inference
            y = F.ssm_recurrent(u, A_bar, B_bar, self.C)
        return y


class SelectiveSSM(nn.Module):
    """
    Selective State Space Model (with delta, B, and C input-dependent).
    """

    def __init__(self, d_model, d_state, **kwargs):
        """
        Args:
            d_model (int): The dimension of the input and output (D).
            d_state (int): The dimension of the hidden state (N).
        """
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # Static param A: real, init with neg values for stability
        # (D, N) => one independent SSM per channel
        self.A = nn.Parameter(-0.5 - 0.5 * torch.rand(d_model, d_state))
        # Projections to make delta, B, C input-dependent
        self.delta_proj = nn.Linear(d_model, d_model, bias=False)
        self.delta_bias = nn.Parameter(torch.zeros(d_model))
        # B and C project from input dimension D to (D * N)
        # => (B, L, D, N) in the forward pass
        self.B_proj = nn.Linear(d_model, d_model * d_state, bias=False)
        self.C_proj = nn.Linear(d_model, d_model * d_state, bias=False)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): The input sequence, shape (B, L, D).

        Returns:
            torch.Tensor: The output sequence y, shape (B, L, D).
        """
        B, L, D = x.shape
        N = self.d_state

        # Compute input-dependent parameters for the entire sequence
        # delta is (B, L, D) => timestep size for each token and channel
        delta = torch.nn.functional.softplus(self.delta_proj(x) + self.delta_bias)

        # Input matrix for each token
        B_selective = self.B_proj(x).view(B, L, D, N)
        # Output matrix for each token
        C_selective = self.C_proj(x).view(B, L, D, N)
        # Initialize hidden state
        h = torch.zeros(B, D, N, device=x.device, dtype=x.dtype)

        outputs = []

        for t in range(L):
            x_t = x[:, t, :]
            delta_t = delta[:, t, :]
            B_t = B_selective[:, t, :, :]
            C_t = C_selective[:, t, :, :]
            # Reshape for broadcast => (B, D, 1)
            delta_t = delta_t.unsqueeze(-1)
            x_t = x_t.unsqueeze(-1)
            # Use ZOH selective
            A_bar, B_bar = F.discretize_zoh_selective(delta_t, self.A, B_t)
            # SSM recurrence relation
            h = A_bar * h + B_bar * x_t
            # Compute output => (B, D)
            y_t = (h * C_t).sum(dim=-1)
            outputs.append(y_t)

        # Stack outputs along the sequence dim => (B, L, D)
        y = torch.stack(outputs, dim=1)
        return y
