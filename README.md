# PyTorch Activations

PyTorch Activations is a collection of activation functions for the PyTorch library. This project aims to provide an easy-to-use solution for experimenting with different activation functions or simply adding variety to your models.


## Installation

You can install PyTorch Activations using pip:

```bash
$ pip install torch-activation
```

## Usage

To use the activation functions, import them from torch_activation. Here's an example:

```python
import torch_activation as tac

m = tac.ShiLU(inplace=True)
x = torch.rand(16, 3, 384, 384)
m(x)
```

Activation functions can be imported directly from the package, such as `torch_activation.CoLU`, or from submodules, such as `torch_activation.non_linear.CoLU`.

For a comprehensive list of available functions, please refer to the [LIST_OF_FUNCTION](LIST_OF_FUNCTION.md) file.

To learn more about usage, please refer to [Documentation]()

We hope you find PyTorch Activations useful for your experimentation and model development. Enjoy exploring different activation functions!

## Contact

Alan Huynh - [LinkedIn](https://www.linkedin.com/in/alan-huynh-64b357194/) - [hdmquan@outlook.com](mailto:hdmquan@outlook.com)

Project Link: [https://github.com/alan191006/torch_activation](https://github.com/alan191006/torch_activation)

Documentation Link: [Documentation]()

PyPI Link: [https://pypi.org/project/torch-activation/](https://pypi.org/project/torch-activation/)

