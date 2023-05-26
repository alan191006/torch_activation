import os
import torch
import psutil
import plotly
import plotly.graph_objects as go


def plot_activation(activation, params, param_names, save_dir="./images/activation_images",
                                x_range=(-5, 5), y_range=None, preview=False):
    x = torch.linspace(x_range[0], x_range[1], 1000)

    fig = go.Figure()

    # Color for each param
    colors = plotly.colors.qualitative.D3[:len(params)]

    for input_values, color in zip(params, colors):
        m = activation(*input_values)
        y = m(x)

        label = "Params:"
        for name, value in zip(param_names, input_values):
            label += f" {name} {value},"

        label = label.rstrip(",")

        # Determine the y-axis range
        if y_range is None:
            y_min = torch.min(y).item()
            y_max = torch.max(y).item()
            y_padding = (y_max - y_min) * 0.1  # Add a 10% padding
            y_range = (y_min - y_padding, y_max + y_padding)

        fig.add_trace(go.Scatter(x=x.detach().numpy(),
                      y=y.detach().numpy(), name=label, line=dict(color=color)))

    fig.update_layout(xaxis=dict(range=x_range), yaxis=dict(
        range=y_range), legend=dict(title="Params"))

    if preview:
        fig.show()
    file_name = os.path.join(save_dir, type(activation).__name__ + '.png')
    fig.write_image(file_name)
    print(f"Image saved as {file_name}")


def test_inplace(activation):
    r"""
    Check and output activation function's output memory usage with inplace=False and inplace=True (if implemented)
    """

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
