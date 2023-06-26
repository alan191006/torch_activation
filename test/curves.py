import unittest
import psutil

import torch
import torch_activation
import torch.nn.functional as F


class TestCoLU(unittest.TestCase):
    def test_forward(self):
        x = torch.tensor([-1.0, 0.5, 2.0])

        m = torch_activation.CoLU()
        output = m(x)

        expected_output = x / (1 - x * torch.exp(-1 * (x + torch.exp(x))))
        self.assertTrue(torch.allclose(output, expected_output))

    def test_forward_inplace(self):
        x = torch.tensor([-1.0, 0.5, 2.0])
        x_ = x.detach().clone()

        m = torch_activation.CoLU(inplace=True)
        m(x_)

        expected_output = x.div_(1 - x * torch.exp(-1 * (x + torch.exp(x))))
        self.assertTrue(torch.allclose(x_, expected_output))


class TestScaledSoftSign(unittest.TestCase):
    def test_forward(self):
        alpha = 0.5
        beta = 1.0
        x = torch.tensor([-1.0, 0.5, 2.0])

        m = torch_activation.ScaledSoftSign(alpha=alpha, beta=beta)
        output = m(x)

        abs_x = x.abs()
        expected_output = (alpha * x) / (beta + abs_x)
        self.assertTrue(torch.allclose(output, expected_output))


class TestPhish(unittest.TestCase):
    def test_forward(self):
        x = torch.tensor([-1.0, 0.5, 2.0])

        m = torch_activation.Phish()
        output = m(x)

        expected_output = x * F.tanh(F.gelu(x))
        self.assertTrue(torch.allclose(output, expected_output))


if __name__ == "__main__":
    unittest.main()
