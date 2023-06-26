import unittest
import psutil

import torch
import torch_activation
import torch.nn.functional as F


class TestShiLU(unittest.TestCase):
    def test_forward(self):
        alpha = 2.0
        beta = 1.0
        x = torch.tensor([-1.0, 0.5, 2.0])

        expected_output = alpha * torch.relu(x) + beta
        m = torch_activation.ShiLU(alpha=alpha, beta=beta)

        output = m(x)

        self.assertTrue(torch.allclose(output, expected_output))

    def test_forward_inplace(self):
        alpha = 2.0
        beta = 1.0
        x = torch.tensor([-1.0, 0.5, 2.0])
        x_ = x.detach().clone()

        expected_output = alpha * torch.relu(x) + beta
        m = torch_activation.ShiLU(alpha=alpha, beta=beta, inplace=True)

        m(x_)

        self.assertTrue(torch.allclose(x_, expected_output))


class TestCReLU(unittest.TestCase):
    def test_forward(self):
        dim = 1
        x = torch.tensor([[-1.0, 0.5, 2.0], [-2.0, -3.0, 4.0]])

        crelu = torch_activation.CReLU(dim=dim)
        output = crelu(x)

        expected_output = F.relu(torch.cat((x, -x), dim=dim))
        self.assertTrue(torch.allclose(output, expected_output))


class TestReLUN(unittest.TestCase):
    def test_forward(self):
        n = 6
        x = torch.tensor([-1.0, 5.0, 7.0])

        expected_output = torch.clamp(x, 0, n)
        m = torch_activation.ReLUN(n=n)

        output = m(x)

        self.assertTrue(torch.allclose(output, expected_output))

    def test_forward_inplace(self):
        n = 6
        x = torch.tensor([-1.0, 5.0, 7.0])
        x_ = x.detach().clone()

        expected_output = torch.clamp(x, 0, n)
        m = torch_activation.ReLUN(n=n, inplace=True)

        m(x_)

        self.assertTrue(torch.allclose(x_, expected_output))


class TestStarReLU(unittest.TestCase):
    def test_forward(self):
        s = 0.8944
        b = -0.4472
        x = torch.tensor([-1.0, 0.5, 2.0])

        expected_output = s * torch.relu(x).pow(2) + b
        m = torch_activation.StarReLU(s=s, b=b)

        output = m(x)

        self.assertTrue(torch.allclose(output, expected_output))

    def test_forward_inplace(self):
        s = 0.8944
        b = -0.4472
        x = torch.tensor([-1.0, 0.5, 2.0])
        x_ = x.detach().clone()

        expected_output = s * torch.relu(x).pow(2) + b
        m = torch_activation.StarReLU(s=s, b=b, inplace=True)

        m(x_)

        self.assertTrue(torch.allclose(x_, expected_output))


if __name__ == "__main__":
    unittest.main()
