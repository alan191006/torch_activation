import os
import torch
import psutil
import plotly
import plotly.graph_objects as go


def plot_activation(activation, params, param_names, save_dir="./images/activation_images",
                    x_range=(-5, 5), y_range=None, preview=False, plot_derivative=True):
    r"""
    Plot the activation function and optionally its derivative.

    Parameters:
        activation (callable): The activation function to plot.
        params (list): A list of parameter tuples for the activation function.
        param_names (list): A list of parameter names corresponding to each tuple in `params`.
        save_dir (str, optional): The directory to save the generated image. Defaults to "./images/activation_images".
        x_range (tuple, optional): The x-axis range for the plot. Defaults to (-5, 5).
        y_range (tuple, optional): The y-axis range for the plot. Defaults to None (auto-scale).
        preview (bool, optional): Whether to display the plot interactively. Defaults to False.
        plot_derivative (bool, optional): Whether to plot the derivative of the activation function. Defaults to True.

    Returns:
        None

    The function plots the activation function and, optionally, its derivative for the given parameters.
    The resulting plot is saved as an image in the specified `save_dir` directory.

    If `preview` is set to True, the plot will also be displayed interactively.

    Example:
        # Plotting the sigmoid activation function and its derivative
        params = [(0.5,), (1.0,), (2.0,)]
        param_names = ['alpha']
        plot_activation(torch.sigmoid, params, param_names, save_dir="./images/activation_images",
                        x_range=(-10, 10), y_range=(-0.5, 1.5), preview=True, plot_derivative=True)
    """
    
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

        if plot_derivative:
            d = torch.autograd.grad(y, x, create_graph=True)[0]
            fig.add_trace(go.Scatter(x=x.detach().numpy(),
                                     y=d.detach().numpy(), name=f"Derivative {label}", line=dict(color=color, dash='dash')))

    fig.update_layout(xaxis=dict(range=x_range), yaxis=dict(range=y_range), legend=dict(title="Params"))

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
        print("\033[91mOutput test: failed\033[0m")

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
