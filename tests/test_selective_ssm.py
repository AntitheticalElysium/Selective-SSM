import unittest

import torch

from src.selective_ssm.modules import SelectiveSSM


class TestSelectiveSSM(unittest.TestCase):
    def setUp(self):
        self.d_model = 4
        self.d_state = 8
        self.batch_size = 2
        self.seq_len = 10
        torch.manual_seed(42)

    def test_basic_forward_pass(self):
        model = SelectiveSSM(d_model=self.d_model, d_state=self.d_state)
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)

        y = model(x)

        self.assertEqual(y.shape, (self.batch_size, self.seq_len, self.d_model))
        self.assertFalse(torch.isnan(y).any(), "Output contains NaN values")
        self.assertFalse(torch.isinf(y).any(), "Output contains Inf values")
        print("Basic forward pass test passed")

    def test_output_dtype_consistency(self):
        model = SelectiveSSM(d_model=self.d_model, d_state=self.d_state)
        x = torch.randn(
            self.batch_size, self.seq_len, self.d_model, dtype=torch.float32
        )

        y = model(x)

        self.assertEqual(y.dtype, x.dtype)
        print("Output dtype consistency test passed")

    def test_gradient_flow(self):
        model = SelectiveSSM(d_model=self.d_model, d_state=self.d_state)
        x = torch.randn(self.batch_size, self.seq_len, self.d_model, requires_grad=True)

        y = model(x)
        loss = y.sum()
        loss.backward()

        self.assertIsNotNone(x.grad, "Gradient not computed for input")
        self.assertFalse(torch.isnan(x.grad).any(), "Gradient contains NaN")

        # Check that model parameters have gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"No gradient for {name}")

        print("Gradient flow test passed")

    def test_batch_independence(self):
        model = SelectiveSSM(d_model=self.d_model, d_state=self.d_state)

        # together
        x_batch = torch.randn(2, self.seq_len, self.d_model)
        y_batch = model(x_batch)

        # separately
        with torch.no_grad():
            y_0 = model(x_batch[0:1])
            y_1 = model(x_batch[1:2])

        self.assertTrue(torch.allclose(y_batch[0], y_0[0], atol=1e-6))
        self.assertTrue(torch.allclose(y_batch[1], y_1[0], atol=1e-6))
        print("Batch independence test passed")


class TestTheorem1(unittest.TestCase):
    """Test Theorem 1: Selective SSM reduces to gated RNN when N=1, A=-1, B=1, C=1."""

    def test_gating_equivalence(self):
        # Setup: N=1 for single state dimension
        d_model = 2
        d_state = 1
        model = SelectiveSSM(d_model=d_model, d_state=d_state)

        # Force A = -1 for all channels
        with torch.no_grad():
            model.A.data = -torch.ones(d_model, d_state)
        model.A.requires_grad = False

        B, L = 2, 5
        x = torch.randn(B, L, d_model)

        y = model(x)

        # Compute using gated RNN
        delta = torch.nn.functional.softplus(model.delta_proj(x) + model.delta_bias)
        h_manual = torch.zeros(B, d_model, d_state)
        outputs_manual = []

        for t in range(L):
            x_t = x[:, t, :]
            delta_t = delta[:, t, :, None]
            B_t = model.B_proj(x[:, t, :]).view(B, d_model, d_state)
            C_t = model.C_proj(x[:, t, :]).view(B, d_model, d_state)

            # Discretize with A=-1
            A_bar = torch.exp(-delta_t)
            # Derived from the ZOH formula
            B_bar = (1 - A_bar) * B_t
            # Recurrence
            h_manual = A_bar * h_manual + B_bar * x_t.unsqueeze(-1)
            y_t_manual = (h_manual * C_t).sum(dim=-1)
            outputs_manual.append(y_t_manual)

        y_manual = torch.stack(outputs_manual, dim=1)

        # Verify equivalence between model output and manual computation
        self.assertTrue(
            torch.allclose(y, y_manual, atol=1e-5),
            "Model output doesn't match manual gated RNN computation",
        )

        # Verify gating behavior
        gate = 1 - torch.exp(-delta)

        self.assertTrue(
            (gate >= 0).all() and (gate <= 1).all(),
            f"Gate values outside [0,1]: min={gate.min()}, max={gate.max()}",
        )

        print("Theorem 1 verification passed!")
        print(f"  Delta range: [{delta.min().item():.4f}, {delta.max().item():.4f}]")
        print(f"  Gate range: [{gate.min().item():.4f}, {gate.max().item():.4f}]")
        print("  Interpretation: gate=1-exp(-Δ) acts as forget gate")

    def test_extreme_gating_behavior(self):
        d_model = 1
        d_state = 1
        model = SelectiveSSM(d_model=d_model, d_state=d_state)

        # Force A = -1
        with torch.no_grad():
            model.A.data = -torch.ones(d_model, d_state)
        model.A.requires_grad = False
        # Large delta should cause strong update => forget old state
        with torch.no_grad():
            model.delta_bias.data = torch.tensor([5.0])

        x_large_delta = torch.ones(1, 3, 1)
        x_large_delta[0, 0, 0] = 1.0
        x_large_delta[0, 1, 0] = 2.0
        x_large_delta[0, 2, 0] = 3.0
        y_large = model(x_large_delta)
        # With large delta, output should be more influenced by current input
        # Small delta should cause weak update => keep old state
        with torch.no_grad():
            model.delta_bias.data = torch.tensor([-5.0])

        y_small = model(x_large_delta)

        self.assertFalse(
            torch.allclose(y_large, y_small, atol=0.1),
            "Large and small delta should produce different behaviors",
        )

        print("Extreme gating behavior test passed")


if __name__ == "__main__":
    unittest.main()
