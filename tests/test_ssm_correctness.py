import unittest

import torch

from src.selective_ssm.functional import (
    discretize_zoh,
    ssm_convolutional,
    ssm_recurrent,
)


class TestSSMCorrectness(unittest.TestCase):
    def setUp(self):
        # Batch size, seq len, model dim & state dim
        self.B = 2
        self.L = 128
        self.D = 4
        self.N = 16

        torch.manual_seed(42)

        # Dummy sequence
        self.u = torch.randn(self.B, self.L, self.D)
        # Real-valued SSM
        self.A_real = -torch.rand(self.D, self.N)  # A is usually negative
        self.B_real = torch.randn(self.D, self.N)
        self.C_real = torch.randn(self.D, self.N)
        self.delta_real = torch.rand(self.D)
        self.A_bar_real, self.B_bar_real = discretize_zoh(
            self.delta_real, self.A_real, self.B_real
        )
        # Complex-valued SSM (like original S4)
        self.A_complex = self.A_real.to(torch.cfloat)
        self.B_complex = self.B_real.to(torch.cfloat)
        self.C_complex = self.C_real.to(torch.cfloat)
        self.delta_complex = self.delta_real.clone()  # Delta is always real though
        self.A_bar_complex, self.B_bar_complex = discretize_zoh(
            self.delta_complex, self.A_complex, self.B_complex
        )

    def test_recurrent_output_shape(self):
        y = ssm_recurrent(self.u, self.A_bar_real, self.B_bar_real, self.C_real)
        self.assertEqual(y.shape, (self.B, self.L, self.D))

    def test_convolutional_output_shape(self):
        y = ssm_convolutional(self.u, self.A_bar_real, self.B_bar_real, self.C_real)
        self.assertEqual(y.shape, (self.B, self.L, self.D))

    def test_equivalence_real(self):
        y_rec = ssm_recurrent(self.u, self.A_bar_real, self.B_bar_real, self.C_real)
        y_conv = ssm_convolutional(
            self.u, self.A_bar_real, self.B_bar_real, self.C_real
        )

        self.assertTrue(torch.allclose(y_rec, y_conv, atol=1e-6))
        print("Real parameter test passed!")

    def test_equivalence_complex(self):
        y_rec = ssm_recurrent(
            self.u, self.A_bar_complex, self.B_bar_complex, self.C_complex
        )
        y_conv = ssm_convolutional(
            self.u, self.A_bar_complex, self.B_bar_complex, self.C_complex
        )

        self.assertTrue(torch.allclose(y_rec, y_conv, atol=1e-6))
        print("Complex parameter test passed!")


if __name__ == "__main__":
    unittest.main()
