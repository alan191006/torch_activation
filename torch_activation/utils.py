import os
import torch
import psutil
import plotly
import plotly.graph_objects as go


def plot_activation(activation: torch.nn.Module, params: dict, save_dir="./images/activation_images",
                    x_range=(-5, 5), y_range=None, preview=False, plot_derivative=True):
    """
    Plot the activation function and optionally its derivative.

    Parameters:
        activation (torch.nn.Module): The activation function to plot.
        params (dict): A dictionary of parameter names and values for the activation function.
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

    Example::

        >>> # Plotting the sigmoid activation function and its derivative
        >>> params = {'n': 1.0}
        >>> plot_activation(torch.nn.Sigmoid(), params)

    """
    x = torch.linspace(x_range[0], x_range[1], 1000)

    fig = go.Figure()

    # Color for each param
    colors = plotly.colors.qualitative.D3[:len(params)]

    for param_name, param_value, color in zip(params.keys(), params.values(), colors):
        m = activation(**{param_name: param_value})
        y = m(x)

        label = f"{param_name}: {param_value}"

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


if __name__ == "__main__":
    pass