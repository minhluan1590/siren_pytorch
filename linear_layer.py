import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import laplace, sobel
from torch.utils.data import Dataset


def paper_init_(weight, is_first=False, omega=1):
    """Initialize the weight matrix of the linear layer using the paper's method.

    Parameters:
    -----------
    :param weight: the weight matrix of the linear layer.
    :param is_first: whether the layer is the first layer.
    :param omega: the hyper parameter of the paper.
    :return: the initialized weight matrix.
    """

    in_features = weight.shape[1]

    with torch.no_grad():
        if is_first:
            bound = 1 / in_features
        else:
            # The author of the paper spends a lot of time on this part, figuring out how to initialize the weight matrix.
            # This helps the model to converge, as we are using a special Sine function as the activation function.
            bound = np.sqrt(6 / in_features) / omega

        weight.uniform_(-bound, bound)


class SineLayer(nn.Module):
    """Linear layer with sine activation function.

    Parameters:
    -----------
    :param in_features: the number of input features.
    :param out_features: the number of output features.
    :param bias: whether to use bias.
    :param is_first: whether the layer is the first layer.
    :param omega: the hyper parameter of the paper.
    :param custom_init_function: None or callable function.
        If None, then we are using the paper_init_ function above.

    Attributes:
    -----------
    linear: Linear layer.
    omega: the hyper parameter of the paper.
    """

    def __init__(
            self,
            in_features,
            out_features,
            bias=True,
            is_first=False,
            omega=1,
            custom_init_function=None,
    ):
        super().__init__()
        self.omega = omega
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        if custom_init_function is None:
            paper_init_(self.linear.weight, is_first, omega)
        else:
            custom_init_function(self.linear.weight)

    def forward(self, x):
        """Run forward pass.

        Parameters:
        -----------
        :param x: input tensor of shape (n_samples, in_features).

        Returns:
        --------
        :return: output tensor of shape (n_samples, in_features).
        """

        return torch.sin(self.omega * self.linear(x))
