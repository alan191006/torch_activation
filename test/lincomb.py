import unittest
import psutil

import torch
import torch_activation
import torch.nn as nn
import torch.nn.functional as F

class TestLinComb(unittest.TestCase):
    def test_forward(self):
        activations = [nn.ReLU(), nn.Sigmoid()]
        input = torch.randn(10)

        m = torch_activation.LinComb(activations)
        output = m(input)

        expected_output = torch.sum(torch.stack([
            m.weights[i] * activations[i](input)
            for i in range(len(activations))
        ]), dim=0)
        self.assertTrue(torch.allclose(output, expected_output))
        
class TestNormLinComb(unittest.TestCase):
    def test_forward(self):
        activations = [nn.ReLU(), nn.Sigmoid()]
        m = torch_activation.NormLinComb(activations)
        input = torch.randn(10)
        output = m(input)

        # Compute the expected output
        activation_outputs = [activations[i](input) for i in range(len(activations))]
        expected_output = torch.sum(torch.stack(activation_outputs), dim=0) / torch.norm(m.weights)

        # Check if the output matches the expected output
        self.assertTrue(torch.allclose(output, expected_output))

if __name__ == '__main__':
    unittest.main()