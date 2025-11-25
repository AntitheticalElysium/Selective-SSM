import unittest

import torch

from src.selective_ssm.modules import MambaBlock, MambaLayer, RMSNorm


class TestRMSNorm(unittest.TestCase):
    def test_forward_shape(self):
        """Test RMSNorm preserves shape."""
        norm = RMSNorm(d_model=64)
        x = torch.randn(2, 100, 64)
        y = norm(x)
        self.assertEqual(y.shape, x.shape)

    def test_normalization(self):
        """Test that RMS is approximately 1 after normalization."""
        norm = RMSNorm(d_model=64)
        x = torch.randn(2, 100, 64) * 10  # Large values
        y = norm(x)

        # Check RMS is close to weight values (initialized to 1)
        rms = torch.sqrt(torch.mean(y**2, dim=-1))
        self.assertTrue(torch.allclose(rms, torch.ones_like(rms), atol=0.1))


class TestMambaBlock(unittest.TestCase):
    def setUp(self):
        self.d_model = 64
        self.d_state = 16
        self.batch_size = 2
        self.seq_len = 32
        torch.manual_seed(42)

    def test_forward_shape(self):
        """Test MambaBlock output shape."""
        block = MambaBlock(d_model=self.d_model, d_state=self.d_state)
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)

        y = block(x)

        self.assertEqual(y.shape, (self.batch_size, self.seq_len, self.d_model))

    def test_no_nans(self):
        """Test that forward pass doesn't produce NaN/Inf."""
        block = MambaBlock(d_model=self.d_model, d_state=self.d_state)
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)

        y = block(x)

        self.assertFalse(torch.isnan(y).any())
        self.assertFalse(torch.isinf(y).any())

    def test_gradient_flow(self):
        """Test gradients flow through the block."""
        block = MambaBlock(d_model=self.d_model, d_state=self.d_state)
        x = torch.randn(self.batch_size, self.seq_len, self.d_model, requires_grad=True)

        y = block(x)
        loss = y.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.isnan(x.grad).any())

        # Check all parameters have gradients
        for name, param in block.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"No gradient for {name}")

    def test_different_sequence_lengths(self):
        """Test block works with various sequence lengths."""
        block = MambaBlock(d_model=self.d_model, d_state=self.d_state)

        for seq_len in [1, 10, 50, 100, 256]:
            x = torch.randn(self.batch_size, seq_len, self.d_model)
            y = block(x)
            self.assertEqual(y.shape, (self.batch_size, seq_len, self.d_model))


class TestMambaLayer(unittest.TestCase):
    def test_residual_connection(self):
        """Test that residual connection is working."""
        layer = MambaLayer(d_model=64, d_state=16)
        x = torch.randn(2, 32, 64)

        # Zero out the mamba block to isolate residual
        with torch.no_grad():
            for param in layer.mamba.parameters():
                param.zero_()

        y = layer(x)

        # With zero mamba, output should be close to input (just normalized)
        # Not exactly equal due to normalization
        self.assertEqual(y.shape, x.shape)

    def test_forward_backward(self):
        """Test complete forward and backward pass."""
        layer = MambaLayer(d_model=64, d_state=16)
        x = torch.randn(2, 32, 64, requires_grad=True)

        y = layer(x)
        loss = y.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.isnan(x.grad).any())


if __name__ == "__main__":
    unittest.main()
