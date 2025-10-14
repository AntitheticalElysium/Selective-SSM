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


if __name__ == "__main__":
    unittest.main()
