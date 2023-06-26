import unittest
import psutil

import torch
import torch_activation
import torch.nn.functional as F


class TestGCU(unittest.TestCase):
    def test_forward(self):
        x = torch.tensor([-1.0, 0.5, 2.0])

        m = torch_activation.GCU()
        output = m(x)

        expected_output = x * torch.cos(x)
        self.assertTrue(torch.allclose(output, expected_output))

    def test_forward_inplace(self):
        x = torch.tensor([-1.0, 0.5, 2.0])

        m = torch_activation.GCU(inplace=True)
        output = m(x)

        expected_output = x.mul_(torch.cos(x))
        self.assertTrue(torch.allclose(output, expected_output))


class TestCosLU(unittest.TestCase):
    def test_forward(self):
        a = 2.0
        b = 1.0
        x = torch.tensor([-1.0, 0.5, 2.0])
        x_ = x.detach().clone()

        m = torch_activation.CosLU(a=a, b=b)
        output = m(x)

        expected_output = (x + 5.0 * torch.cos(6.0 * x)) * torch.sigmoid(x)
        self.assertTrue(torch.allclose(output, expected_output))

    def test_forward_inplace(self):
        a = 2.0
        b = 1.0
        x = torch.tensor([-1.0, 0.5, 2.0])
        x_ = x.detach().clone()

        m = torch_activation.CosLU(a=a, b=b, inplace=True)
        m(x_)

        expected_output = (x + 5.0 * torch.cos(6.0 * x)) * torch.sigmoid(x)
        self.assertTrue(torch.allclose(x_, expected_output))


class TestSinLU(unittest.TestCase):
    def test_forward(self):
        a = 2.0
        b = 1.0
        x = torch.tensor([-1.0, 0.5, 2.0])
        x_ = x.detach().clone()

        m = torch_activation.SinLU(a=a, b=b)
        output = m(x)

        expected_output = (x + 5.0 * torch.sin(6.0 * x)) * torch.sigmoid(x)
        self.assertTrue(torch.allclose(output, expected_output))

    def test_forward_inplace(self):
        a = 2.0
        b = 1.0
        x = torch.tensor([-1.0, 0.5, 2.0])
        x_ = x.detach().clone()

        m = torch_activation.SinLU(a=a, b=b, inplace=True)
        m(x_)

        expected_output = (x + 5.0 * torch.sin(6.0 * x)) * torch.sigmoid(x)
        self.assertTrue(torch.allclose(x_, expected_output))


if __name__ == "__main__":
    unittest.main()
