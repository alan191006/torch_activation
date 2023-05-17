import os
import torch
import psutil
import plotly.io as pio
import plotly.graph_objects as go

import torch_activation
import torch_activation.__init__ as act_init

exception = ["CReLU", "GeGLU", "ReGLU", "SwiGLU"]


def plot_activation_(activation, save_dir, x_range=(-5, 5), y_range=None, preview=False):
    if type(activation).__name__ in exception:
        print(f"{type(activation).__name__} is not element-wise and/or one-to-one")
        return

    x = torch.linspace(x_range[0], x_range[1], 1000)
    y = activation(x)

    # Determine the y-axis range
    if y_range is None:
        y_min = torch.min(y).item()
        y_max = torch.max(y).item()
        y_padding = (y_max - y_min) * 0.1  # Add a 10% padding
        y_range = (y_min - y_padding, y_max + y_padding)

    fig = go.Figure(go.Scatter(x=x.detach().numpy(), y=y.detach().numpy()))
    fig.update_layout(xaxis=dict(range=x_range), yaxis=dict(range=y_range))

    if preview:
        fig.show()
    file_name = os.path.join(save_dir, type(activation).__name__ + '.png')
    fig.write_image(file_name)
    print(f"Image saved as {file_name}")


def plot_activations():
    r"""
    Plot all functions from __init__.py (except in exception list).
    """
    lst = list(map(act_init.__dict__.get, act_init.__all__))
    for activation in lst:
        activation = getattr(torch_activation, activation.__name__)()
        plot_activation_(activation, r"images/activation_images/")


def test_inplace_(activation):
    r"""
    Check and output activation function's output memory usage with inplace=False and inplace=True (if implemented)
    """

    if activation.__name__ in exception:
        return
    print('_' * 20)
    print()
    print(f"Testing {activation.__name__}...\n")

    x = torch.rand(5)

    m = activation()
    y = m(x)

    print(f"Output:          {y.tolist()}")

    if not hasattr(activation(), 'inplace'):
        print(f"{activation.__name__} does not have in-place option")
        return

    m_ = activation(inplace=True)
    m_(x)

    print(f"In-place output: {x.tolist()}\n")

    if torch.allclose(x, y):
        print("\033[92mOutput test: passed\033[0m")
    else:
        print("\033[91m Output test: failed\033[0m")

    print()

    process = psutil.Process()

    x = torch.randn(10000, 10000)

    mem_before = process.memory_info().rss
    _ = m(x)
    mem_after = process.memory_info().rss

    print("No in-place")
    print(f"Memory before: {mem_before}, memory after: {mem_after}")
    print(f"\033[1mMemory diffrence: {mem_after-mem_before}\033[0m")
    print()

    x = torch.randn(10000, 10000)

    mem_before = process.memory_info().rss
    m_(x)
    mem_after = process.memory_info().rss

    print("In-place")
    print(f"Memory before: {mem_before}, memory after: {mem_after}")
    print(f"\033[1mMemory diffrence: {mem_after-mem_before}\033[0m")

    print()


def test_inplace():
    r"""
    Test inplace implementation of all functions from __init__.py (if exists).
    """

    lst = list(map(act_init.__dict__.get, act_init.__all__))
    for activation in lst:
        activation = getattr(torch_activation, activation.__name__)
        test_inplace_(activation)


if __name__ == "__main__":
    # You don't need to import anything if
    # your activation is written in __all__ in __init__.py

    plot_activations()
    test_inplace()
