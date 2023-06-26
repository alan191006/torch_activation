import unittest
import psutil

import torch
import torch_activation
import torch.nn.functional as F


class TestReGLU(unittest.TestCase):
    def test_forward(self):
        x = torch.randn(3, 20, 20)

        m = torch_activation.ReGLU()
        output = m(x)

        a, b = x.chunk(2, dim=-1)
        expected_output = a * F.relu(b)
        self.assertTrue(torch.allclose(output, expected_output))


class TestGeGLU(unittest.TestCase):
    def test_forward(self):
        x = torch.randn(3, 20, 20)

        m = torch_activation.GeGLU()
        output = m(x)

        a, b = x.chunk(2, dim=-1)
        expected_output = a * F.gelu(b)
        self.assertTrue(torch.allclose(output, expected_output))


class TestSwiGLU(unittest.TestCase):
    def test_forward(self):
        x = torch.randn(3, 20, 20)

        m = torch_activation.SwiGLU()
        output = m(x)

        a, b = x.chunk(2, dim=-1)
        expected_output = a * F.silu(b)
        self.assertTrue(torch.allclose(output, expected_output))


class TestSeGLU(unittest.TestCase):
    def test_forward(self):
        x = torch.randn(3, 20, 20)

        m = torch_activation.SeGLU()
        output = m(x)

        a, b = x.chunk(2, dim=-1)
        expected_output = a * F.selu(b)
        self.assertTrue(torch.allclose(output, expected_output))


if __name__ == "__main__":
    unittest.main()
