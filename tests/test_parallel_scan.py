import unittest

import torch

from src.selective_ssm.functional import ssm_binary_operator


class TestBinaryOperator(unittest.TestCase):
    """Test the associative binary operator for SSM scan."""

    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        self.B = 2  # Batch size
        self.D = 4  # Model dimension
        self.N = 8  # State dimension

    def test_operator_output_shape(self):
        """Test that operator preserves tensor shapes."""
        A1 = torch.randn(self.B, self.D, self.N)
        b1 = torch.randn(self.B, self.D, self.N)
        A2 = torch.randn(self.B, self.D, self.N)
        b2 = torch.randn(self.B, self.D, self.N)

        A_combined, b_combined = ssm_binary_operator((A1, b1), (A2, b2))

        self.assertEqual(A_combined.shape, (self.B, self.D, self.N))
        self.assertEqual(b_combined.shape, (self.B, self.D, self.N))
        print("Operator output shape test passed")

    def test_associativity(self):
        """
        Test that the operator is associative.

        Verifies: ((e1 ⊕ e2) ⊕ e3) == (e1 ⊕ (e2 ⊕ e3))

        This is the fundamental property required for parallel scan.
        """
        # Create three random elements
        A1 = torch.randn(self.B, self.D, self.N)
        b1 = torch.randn(self.B, self.D, self.N)
        A2 = torch.randn(self.B, self.D, self.N)
        b2 = torch.randn(self.B, self.D, self.N)
        A3 = torch.randn(self.B, self.D, self.N)
        b3 = torch.randn(self.B, self.D, self.N)

        # Left-associative: ((e1 ⊕ e2) ⊕ e3)
        temp1 = ssm_binary_operator((A1, b1), (A2, b2))
        left_A, left_b = ssm_binary_operator(temp1, (A3, b3))

        # Right-associative: (e1 ⊕ (e2 ⊕ e3))
        temp2 = ssm_binary_operator((A2, b2), (A3, b3))
        right_A, right_b = ssm_binary_operator((A1, b1), temp2)

        # Verify they produce the same result
        self.assertTrue(
            torch.allclose(left_A, right_A, atol=1e-5),
            f"A matrices don't match: max diff = {(left_A - right_A).abs().max()}",
        )
        self.assertTrue(
            torch.allclose(left_b, right_b, atol=1e-5),
            f"b vectors don't match: max diff = {(left_b - right_b).abs().max()}",
        )
        print("✓ Associativity test passed")

    def test_identity_element(self):
        """
        Test that (ones, zeros) acts as an identity element.

        For any element (A, b):
            (A, b) ⊕ (I, 0) = (A, b)
            (I, 0) ⊕ (A, b) = (A, b)

        where I is the identity (all ones for element-wise multiplication).
        """
        A = torch.randn(self.B, self.D, self.N)
        b = torch.randn(self.B, self.D, self.N)

        # Identity element
        Identity = torch.ones(self.B, self.D, self.N)
        zero = torch.zeros(self.B, self.D, self.N)

        # Right identity: (A, b) ⊕ (I, 0) = (A, b)
        A_right, b_right = ssm_binary_operator((A, b), (Identity, zero))
        self.assertTrue(torch.allclose(A_right, A, atol=1e-6))
        self.assertTrue(torch.allclose(b_right, b, atol=1e-6))

        # Left identity: (I, 0) ⊕ (A, b) = (A, b)
        A_left, b_left = ssm_binary_operator((Identity, zero), (A, b))
        self.assertTrue(torch.allclose(A_left, A, atol=1e-6))
        self.assertTrue(torch.allclose(b_left, b, atol=1e-6))

        print("Identity element test passed")

    def test_manual_computation(self):
        """
        Test against manual computation for a simple case.

        Verifies the mathematical correctness by computing the result
        manually and comparing.
        """
        # Simple 2x2 case for easy verification
        A1 = torch.tensor([[[2.0, 3.0]]])  # (1, 1, 2)
        b1 = torch.tensor([[[1.0, 2.0]]])  # (1, 1, 2)
        A2 = torch.tensor([[[4.0, 5.0]]])  # (1, 1, 2)
        b2 = torch.tensor([[[3.0, 4.0]]])  # (1, 1, 2)

        A_combined, b_combined = ssm_binary_operator((A1, b1), (A2, b2))

        # Manual computation:
        # A_combined = A2 * A1 = [4*2, 5*3] = [8, 15]
        # b_combined = A2 * b1 + b2 = [4*1, 5*2] + [3, 4] = [4, 10] + [3, 4] = [7, 14]
        expected_A = torch.tensor([[[8.0, 15.0]]])
        expected_b = torch.tensor([[[7.0, 14.0]]])

        self.assertTrue(torch.allclose(A_combined, expected_A, atol=1e-6))
        self.assertTrue(torch.allclose(b_combined, expected_b, atol=1e-6))

        print("Manual computation test passed")

    def test_sequential_composition(self):
        """
        Test that the operator correctly represents sequential SSM steps.

        Verifies that applying two transitions via the operator gives
        the same result as applying them sequentially.
        """
        # Initial hidden state
        h0 = torch.randn(self.B, self.D, self.N)

        # Two transitions
        A1 = torch.randn(self.B, self.D, self.N)
        b1 = torch.randn(self.B, self.D, self.N)
        A2 = torch.randn(self.B, self.D, self.N)
        b2 = torch.randn(self.B, self.D, self.N)

        # Apply sequentially
        h1 = A1 * h0 + b1
        h2_sequential = A2 * h1 + b2

        # Apply via operator
        A_combined, b_combined = ssm_binary_operator((A1, b1), (A2, b2))
        h2_operator = A_combined * h0 + b_combined

        # Should give same result
        self.assertTrue(
            torch.allclose(h2_sequential, h2_operator, atol=1e-5),
            f"Sequential vs operator mismatch: max diff = "
            f"{(h2_sequential - h2_operator).abs().max()}",
        )
        print("Sequential composition test passed")

    def test_different_batch_sizes(self):
        """Test operator works with different batch sizes."""
        for batch_size in [1, 2, 8, 16]:
            A1 = torch.randn(batch_size, self.D, self.N)
            b1 = torch.randn(batch_size, self.D, self.N)
            A2 = torch.randn(batch_size, self.D, self.N)
            b2 = torch.randn(batch_size, self.D, self.N)

            A_combined, b_combined = ssm_binary_operator((A1, b1), (A2, b2))

            self.assertEqual(A_combined.shape, (batch_size, self.D, self.N))
            self.assertEqual(b_combined.shape, (batch_size, self.D, self.N))

        print("Different batch sizes test passed")

    def test_dtype_preservation(self):
        """Test that operator preserves data types."""
        for dtype in [torch.float32, torch.float64]:
            A1 = torch.randn(self.B, self.D, self.N, dtype=dtype)
            b1 = torch.randn(self.B, self.D, self.N, dtype=dtype)
            A2 = torch.randn(self.B, self.D, self.N, dtype=dtype)
            b2 = torch.randn(self.B, self.D, self.N, dtype=dtype)

            A_combined, b_combined = ssm_binary_operator((A1, b1), (A2, b2))

            self.assertEqual(A_combined.dtype, dtype)
            self.assertEqual(b_combined.dtype, dtype)

        print("Data type preservation test passed")

    def test_no_nans_or_infs(self):
        """Test that operator doesn't produce NaN or Inf values."""
        A1 = torch.randn(self.B, self.D, self.N)
        b1 = torch.randn(self.B, self.D, self.N)
        A2 = torch.randn(self.B, self.D, self.N)
        b2 = torch.randn(self.B, self.D, self.N)

        A_combined, b_combined = ssm_binary_operator((A1, b1), (A2, b2))

        self.assertFalse(torch.isnan(A_combined).any(), "A_combined contains NaN")
        self.assertFalse(torch.isinf(A_combined).any(), "A_combined contains Inf")
        self.assertFalse(torch.isnan(b_combined).any(), "b_combined contains NaN")
        self.assertFalse(torch.isinf(b_combined).any(), "b_combined contains Inf")

        print("No NaN/Inf test passed")


if __name__ == "__main__":
    unittest.main()
