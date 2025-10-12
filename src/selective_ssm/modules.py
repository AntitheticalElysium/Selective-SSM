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
        self.A = nn.Parameter(torch.randn(d_model, d_state, dtype=torch.cfloat))
        self.B = nn.Parameter(torch.randn(d_model, d_state))
        self.C = nn.Parameter(torch.randn(d_model, d_state))
        # Delta is used for discretization (ZOH)
        self.delta = nn.Parameter(torch.randn(d_model))

    def forward(self, u):
        """
        Args:
            u (torch.Tensor): The input sequence, shape (B, L, D).

        Returns:
            torch.Tensor: The output sequence y, shape (B, L, D).
        """
        A_bar, B_bar = F.discretize_zoh(self.delta, self.A, self.B)

        if self.training:
            # Use more efficient convolutional mode
            y = F.ssm_convolutional(u, A_bar, B_bar, self.C)
        else:
            # Autoregressive inference
            y = F.ssm_recurrent(u, A_bar, B_bar, self.C)
        return y
