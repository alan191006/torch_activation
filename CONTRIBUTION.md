# Contributing to torch_activation

Thank you for your interest in contributing to torch_activation. Your contribution to this project is appreciated.

---

## How to contribute

To contribute, please follow these steps:

1. Fork the repository.

2. Install the required dependencies by running `pip install -r requirements.txt`.

3. Create you module in one of the existing files or create a new one in `torch_activation` folder.

4. Add your module to `__init__.py` file.

```python
from .your_module import your_activation

__all__ = ["ScaledSoftSign", "your_activation"]
```

5. If you want to plot your activation, run the `plot_activations()` function in `utils.py`. If the activation you implement is not element-wise or one-to-one, please add to the `exception` list.

6. If you want to test your inplace implementation of your activation, please run `test_inplace()` function in `utils.py`.

7. Create a Pull Request to the repository.

