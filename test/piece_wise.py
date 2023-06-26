import unittest
import psutil

import torch
import torch_activation
import torch.nn.functional as F


class TestDELU(unittest.TestCase):
    def test_forward(self):
        n = 2.0
        x = torch.tensor([-1.0, 0.5, 2.0])

        m = torch_activation.DELU(n=n)
        output = m(x)

        expected_output = torch.where(
            x <= 0, F.silu(x), (n + 0.5) * x + torch.abs(torch.exp(-x) - 1)
        )
        self.assertTrue(torch.allclose(output, expected_output))

    def test_forward_inplace(self):
        n = 2.0
        x = torch.tensor([-1.0, 0.5, 2.0])
        x_ = x.detach().clone()

        m = torch_activation.DELU(n=n, inplace=True)
        m(x_)

        expected_output = torch.where(
            x <= 0, F.silu(x), (n + 0.5) * x + torch.abs(torch.exp(-x) - 1)
        )
        self.assertTrue(torch.allclose(x_, expected_output))


if __name__ == "__main__":
    unittest.main()
