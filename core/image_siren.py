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


class ImageSiren(nn.Module):
    """ Network composed of SineLayers.

    Parameters:
    -----------
    :param hidden_features: int
        Number of hidden features (each hidden layer would have the same).
    :param hidden_layers: int
        Number of hidden layers.
    :param first_omega: float
        Hyperparameter.
    :param hidden_omega: float
        Hyperparameter.
    :param custom_init_function: None or callable.
        If None, then use the paper_init_ function.

    Attributes:
    -----------
    net: nn.Sequential
        Sequential including "SineLayer" and "nn.Linear" at the end.
    """

    def __init__(
            self,
            hidden_features,
            hidden_layers=1,
            first_omega=30,
            hidden_omega=30,
            custom_init_function=None,
    ):
        super().__init__()

        # We will have 2 input features, representing the coordinates of the pixel.
        in_features = 2

        # We will have a single output feature, representing the pixel's value.
        out_features = 1

        net = nn.Sequential()
        net.add_module('SineLayer',
                       SineLayer(in_features, hidden_features, is_first=True, omega=first_omega,
                                 custom_init_function=custom_init_function))

        for i in range(hidden_layers):
            net.add_module('SineLayer_{}'.format(i),
                           SineLayer(hidden_features, hidden_features, omega=hidden_omega,
                                     custom_init_function=custom_init_function))

        net.add_module('Linear', nn.Linear(hidden_features, out_features))

        # SineLayer already initialized, so we only need to initialize the Linear layer.
        if custom_init_function is None:
            paper_init_(net[-1].weight, is_first=False, omega=hidden_omega)
        else:
            custom_init_function(net[-1].weight)

        self.net = net

    def forward(self, x):
        """Run forward pass.

        Parameters:
        -----------
        :param x: input tensor of shape (n_samples, 2).

        Returns:
        --------
        :return: output tensor of shape (n_samples, 1).
        """
        return self.net(x)


def generate_coordinates(n):
    """Generate regular grid of 2D coordinates on [0, n] x [0, n].

    Parameters:
    -----------
    :param n: int
        Number of pixels in each dimension.

    Returns:
    --------
    :return: np.array of shape (n^2, 2).
        An array of 2D coordinates, in which each row is a 2D coordinate.
    """
    rows, cols = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
    coords_abs = np.stack([rows.ravel(), cols.ravel()], axis=1)

    return coords_abs


class PixelDataset(Dataset):
    """Dataset yielding coordinates, intensities and (higher) derivatives.

    Parameters:
    -----------
    :param img: np.array
        2D image representing a grayscale image.

    Attributes:
    -----------
    size: int
        Height and width of the square image.

    coords_abs: np.ndarray
        Array of shape (size ^ 2, 2) containing the absolute coordinates of each pixel.

    grad: np.array
        Array of shape (size, size, 2) representing the gradient of the image.

    grad_norm: np.array
        Array of shape (size, size) representing the approximate gradient norm of the image.

    laplace: np.array
        Array of shape (size, size) representing the Laplacian of the image.
    """

    def __init__(self, img):
        if not (img.ndim == 2 and img.shape[0] == img.shape[1]):
            raise ValueError('Image must be a square matrix.')

        self.img = img
        self.size = img.shape[0]
        self.coords_abs = generate_coordinates(self.size)
        self.grad = np.stack([
            sobel(img, axis=0),
            sobel(img, axis=1)
        ], axis=-1)
        self.grad_norm = np.linalg.norm(self.grad, axis=-1)
        self.laplace = laplace(img)

    def __len__(self):
        """Determine the size of the dataset."""
        return self.size ** 2

    def __getitem__(self, idx):
        """Get all relevant data for a single coordinate."""
        coords_abs = self.coords_abs[idx]
        r, c = coords_abs
        coords = 2 * (coords_abs / self.size) - 1

        return {
            'coords': coords,
            'coords_abs': coords_abs,
            'intensity': self.img[r, c],
            'grad': self.grad[r, c],
            'grad_norm': self.grad_norm[r, c],
            'laplace': self.laplace[r, c]
        }
