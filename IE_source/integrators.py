import logging
import warnings
from typing import Callable, Optional, Union
import time
import numpy as np
import torch

from scipy import integrate

logger = logging.getLogger("idesolver")
logger.setLevel(logging.WARNING)#(logging.DEBUG)

import matplotlib.pyplot as plt

# !pip install git+https://github.com/rtqichen/torchdiffeq
# pip install gputil
# !pip install git+https://github.com/patrick-kidger/torchcubicspline.git
# !pip install tqdm
# !pip install seaborn

import torchcubicspline

from torchcubicspline import(natural_cubic_spline_coeffs, 
                             NaturalCubicSpline)
                             
from torchdiffeq import odeint

import math
import numpy as np
from IPython.display import clear_output
# from tqdm import tqdm_notebook as tqdm
# Please use `tqdm.notebook.tqdm` instead of `tqdm.tqdm_notebook`
# from tqdm import notebook.tqdm as tqdm
from tqdm.notebook import tqdm



import matplotlib as mpl
import matplotlib.pyplot as plt

# import seaborn as sns
# sns.color_palette("bright")
import matplotlib as mpl
import matplotlib.cm as cm

import torch
from torch import Tensor
from torch import nn
from torch.nn  import functional as F
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()

from functools import reduce

from torch import Tensor 
# import abc

# from torch import nn

# from torch.autograd.functional import vjp
# # !pip install torch torchvision

# # Adapted codes from M. Surtsukov's blog:
# # https://msurtsukov.github.io/Neural-ODE/


# torchquad implemented below without infos appearing
#!conda install torchquad -c conda-forge -c pytorch

#from torchquad import MonteCarlo, enable_cuda

#mc = MonteCarlo()



class BaseIntegrator:
    """The (abstract) integrator that all other integrators inherit from. Provides no explicit definitions for methods."""

    # Function to evaluate
    _fn = None

    # Dimensionality of function to evaluate
    _dim = None

    # Integration domain
    _integration_domain = None

    # Number of function evaluations
    _nr_of_fevals = None

    def __init__(self):
        self._nr_of_fevals = 0

    def integrate(self):
        raise (
            NotImplementedError("This is an abstract base class. Should not be called.")
        )

    def _eval(self, points):
        """Evaluates the function at the passed points and updates nr_of_evals
        Args:
            points (torch.tensor): Integration points
        """
        self._nr_of_fevals += len(points)
        result = self._fn(points)
        if type(result) != torch.Tensor:
            warnings.warn(
                "The passed function did not return a torch.tensor. Will try to convert. Note that this may be slow as it results in memory transfers between CPU and GPU, if torchquad uses the GPU."
            )
            result = torch.tensor(result)

        if len(result) != len(points):
            raise ValueError(
                f"The passed function was given {len(points)} points but only returned {len(result)} value(s)."
                f"Please ensure that your function is vectorized, i.e. can be called with multiple evaluation points at once. It should return a tensor "
                f"where first dimension matches length of passed elements. "
            )

        return result

    def _check_inputs(self, dim=None, N=None, integration_domain=None):
        """Used to check input validity
        Args:
            dim (int, optional): Dimensionality of function to integrate. Defaults to None.
            N (int, optional): Total number of integration points. Defaults to None.
            integration_domain (list, optional): Integration domain, e.g. [[0,1],[1,2]]. Defaults to None.
        Raises:
            ValueError: if inputs are not compatible with each other.
        """
        logger.debug("Checking inputs to Integrator.")
        if dim is not None:
            if dim < 1:
                raise ValueError("Dimension needs to be 1 or larger.")

            if integration_domain is not None:
                if dim != len(integration_domain):
                    raise ValueError(
                        "Dimension of integration_domain needs to match the passed function dimensionality dim."
                    )

        if N is not None:
            if N < 1 or type(N) is not int:
                raise ValueError("N has to be a positive integer.")

        if integration_domain is not None:
            for bounds in integration_domain:
                if len(bounds) != 2:
                    raise ValueError(
                        bounds,
                        " in ",
                        integration_domain,
                        " does not specify a valid integration bound.",
                    )
                if bounds[0] > bounds[1]:
                    raise ValueError(
                        bounds,
                        " in ",
                        integration_domain,
                        " does not specify a valid integration bound.",
                    )
                    

class IntegrationGrid:
    """This class is used to store the integration grid for methods like Trapezoid or Simpsons, which require a grid."""

    points = None  # integration points
    h = None  # mesh width
    _N = None  # number of mesh points
    _dim = None  # dimensionality of the grid
    _runtime = None  # runtime for the creation of the integration grid

    def __init__(self, N, integration_domain):
        """Creates an integration grid of N points in the passed domain. Dimension will be len(integration_domain)
        Args:
            N (int): Total desired number of points in the grid (will take next lower root depending on dim)
            integration_domain (list): Domain to choose points in, e.g. [[-1,1],[0,1]].
        """
        start = perf_counter()
        self._check_inputs(N, integration_domain)
        self._dim = len(integration_domain)

        # TODO Add that N can be different for each dimension
        # A rounding error occurs for certain numbers with certain powers,
        # e.g. (4**3)**(1/3) = 3.99999... Because int() floors the number,
        # i.e. int(3.99999...) -> 3, a little error term is useful
        self._N = int(N ** (1.0 / self._dim) + 1e-8)  # convert to points per dim

        self.h = torch.zeros([self._dim])

        logger.debug(
            "Creating "
            + str(self._dim)
            + "-dimensional integration grid with "
            + str(N)
            + " points over"
            + str(integration_domain),
        )

        # Check if domain requires gradient
        if hasattr(integration_domain, "requires_grad"):
            requires_grad = integration_domain.requires_grad
        else:
            requires_grad = False

        grid_1d = []
        # Determine for each dimension grid points and mesh width
        for dim in range(self._dim):
            grid_1d.append(
                _linspace_with_grads(
                    integration_domain[dim][0],
                    integration_domain[dim][1],
                    self._N,
                    requires_grad=requires_grad,
                )
            )
            self.h[dim] = grid_1d[dim][1] - grid_1d[dim][0]

        logger.debug("Grid mesh width is " + str(self.h))

        # Get grid points
        points = torch.meshgrid(*grid_1d)

        # Flatten to 1D
        points = [p.flatten() for p in points]

        self.points = torch.stack((tuple(points))).transpose(0, 1)

        logger.info("Integration grid created.")

        self._runtime = perf_counter() - start

    def _check_inputs(self, N, integration_domain):
        """Used to check input validity"""

        logger.debug("Checking inputs to IntegrationGrid.")
        dim = len(integration_domain)

        if dim < 1:
            raise ValueError("len(integration_domain) needs to be 1 or larger.")

        if N < 2:
            raise ValueError("N has to be > 1.")

        if N ** (1.0 / dim) < 2:
            raise ValueError(
                "Cannot create a ",
                dim,
                "-dimensional grid with ",
                N,
                " points. Too few points per dimension.",
            )

        for bounds in integration_domain:
            if len(bounds) != 2:
                raise ValueError(
                    bounds,
                    " in ",
                    integration_domain,
                    " does not specify a valid integration bound.",
                )
            if bounds[0] > bounds[1]:
                raise ValueError(
                    bounds,
                    " in ",
                    integration_domain,
                    " does not specify a valid integration bound.",
                )

class Boole(BaseIntegrator):

    """Boole's rule in torch. See https://en.wikipedia.org/wiki/Newton%E2%80%93Cotes_formulas#Closed_Newton%E2%80%93Cotes_formulas ."""

    def __init__(self):
        super().__init__()

    def integrate(self, fn, dim, N=None, integration_domain=None):
        """Integrates the passed function on the passed domain using Boole's rule.
        Args:
            fn (func): The function to integrate over.
            dim (int): Dimensionality of the integration domain.
            N (int, optional): Total number of sample points to use for the integration. N has to be such that N^(1/dim) - 1 % 4 == 0. Defaults to 5 points per dimension if None is given.
            integration_domain (list, optional): Integration domain, e.g. [[-1,1],[0,1]]. Defaults to [-1,1]^dim.
        Returns:
            float: integral value
        """

        # If N is unspecified, set N to 5 points per dimension
        if N is None:
            N = 5 ** dim

        self._integration_domain = _setup_integration_domain(dim, integration_domain)
        self._check_inputs(dim=dim, N=N, integration_domain=self._integration_domain)
        N = self._adjust_N(dim=dim, N=N)

        self._dim = dim
        self._fn = fn

        logger.debug(
            "Using Boole for integrating a fn with a total of "
            + str(N)
            + " points over "
            + str(self._integration_domain)
            + "."
        )

        # Create grid and assemble evaluation points
        self._grid = IntegrationGrid(N, self._integration_domain)

        logger.debug("Evaluating integrand on the grid.")
        function_values = self._eval(self._grid.points)

        # Reshape the output to be [N,N,...] points instead of [dim*N] points
        function_values = function_values.reshape([self._grid._N] * dim)

        logger.debug("Computing areas.")

        # This will contain the Simpson's areas per dimension
        cur_dim_areas = function_values

        # We collapse dimension by dimension
        for cur_dim in range(dim):
            cur_dim_areas = (
                self._grid.h[cur_dim]
                / 22.5
                * (
                    7 * cur_dim_areas[..., 0:-4][..., ::4]
                    + 32 * cur_dim_areas[..., 1:-3][..., ::4]
                    + 12 * cur_dim_areas[..., 2:-2][..., ::4]
                    + 32 * cur_dim_areas[..., 3:-1][..., ::4]
                    + 7 * cur_dim_areas[..., 4:][..., ::4]
                )
            )
            cur_dim_areas = torch.sum(cur_dim_areas, dim=dim - cur_dim - 1)
        logger.info("Computed integral was " + str(cur_dim_areas) + ".")

        return cur_dim_areas

    def _adjust_N(self, dim, N):
        """Adjusts the total number of points to a valid number, i.e. N satisfies N^(1/dim) - 1 % 4 == 0.
        Args:
            dim (int): Dimensionality of the integration domain.
            N (int): Total number of sample points to use for the integration.
        Returns:
            int: An N satisfying N^(1/dim) - 1 % 4 == 0.
        """
        n_per_dim = int(N ** (1.0 / dim) + 1e-8)
        logger.debug(
            "Checking if N per dim is >=5 and N = 1 + 4n, where n is a positive integer."
        )

        # Boole's rule requires N per dim >=5 and N = 1 + 4n,
        # where n is a positive integer, for correctness.
        if n_per_dim < 5:
            warnings.warn(
                "N per dimension cannot be lower than 5. "
                "N per dim will now be changed to 5."
            )
            N = 5 ** dim
        elif (n_per_dim - 1) % 4 != 0:
            new_n_per_dim = n_per_dim - ((n_per_dim - 1) % 4)
            warnings.warn(
                "N per dimension must be N = 1 + 4n with n a positive integer due to necessary subdivisions. "
                "N per dim will now be changed to the next lower N satisfying this, i.e. "
                f"{n_per_dim} -> {new_n_per_dim}."
            )
            N = (new_n_per_dim) ** (dim)
        return N
    
    
class MonteCarlo(BaseIntegrator):
    """Monte Carlo integration in torch."""

    def __init__(self):
        super().__init__()

    def integrate(self, fn, dim, N=1000, integration_domain=None, seed=None, out_dim = -2):
        """Integrates the passed function on the passed domain using vanilla Monte Carlo Integration.
        Args:
            fn (func): The function to integrate over.
            dim (int): Dimensionality of the function to integrate.
            N (int, optional): Number of sample points to use for the integration. Defaults to 1000.
            integration_domain (list, optional): Integration domain, e.g. [[-1,1],[0,1]]. Defaults to [-1,1]^dim.
            seed (int, optional): Random number generation seed to the sampling point creation, only set if provided. Defaults to None.
        Raises:
            ValueError: If len(integration_domain) != dim
        Returns:
            float: integral value
        """
        self._check_inputs(dim=dim, N=N, integration_domain=integration_domain)
        logger.debug(
            "Monte Carlo integrating a "
            + str(dim)
            + "-dimensional fn with "
            + str(N)
            + " points over "
            + str(integration_domain),
        )

        self._dim = dim
        self.out_dim = out_dim
        self._nr_of_fevals = 0
        self.fn = fn
        self._integration_domain = _setup_integration_domain(dim, integration_domain)
        if seed is not None:
            torch.random.manual_seed(seed)

        logger.debug("Picking random sampling points")
        sample_points = torch.zeros([N, dim])
        for d in range(dim):
            scale = self._integration_domain[d, 1] - self._integration_domain[d, 0]
            offset = self._integration_domain[d, 0]
            sample_points[:, d] = torch.rand(N) * scale + offset

        logger.debug("Evaluating integrand")
        function_values = fn(sample_points)

        logger.debug("Computing integration domain volume")
        scales = self._integration_domain[:, 1] - self._integration_domain[:, 0]
        volume = torch.prod(scales)

        # Integral = V / N * sum(func values)
        #integral = volume * torch.sum(function_values) / N
        integral = volume * torch.sum(function_values,self.out_dim) / N
        #logger.info("Computed integral was " + str(integral))
        return integral
    
def _linspace_with_grads(start, stop, N, requires_grad):
    """Creates an equally spaced 1D grid while keeping gradients
    in regard to inputs
    Args:
        start (torch.tensor): Start point (inclusive)
        stop (torch.tensor): End point (inclusive)
        N (torch.tensor): Number of points
        requires_grad (bool): Indicates if output should be
            recorded for backpropagation
    Returns:
        torch.tensor: Equally spaced 1D grid
    """
    if requires_grad:
        # Create 0 to 1 spaced grid
        grid = torch.linspace(0, 1, N)

        # Scale to desired range , thus keeping gradients
        grid *= stop - start
        grid += start

        return grid
    else:
        return torch.linspace(start, stop, N)


def _setup_integration_domain(dim, integration_domain):
    """Sets up the integration domain if unspecified by the user.
    Args:
        dim (int): Dimensionality of the integration domain.
        integration_domain (list, optional): Integration domain, e.g. [[-1,1],[0,1]]. Defaults to [-1,1]^dim.
    Returns:
        torch.tensor: Integration domain.
    """

    # Store integration_domain
    # If not specified, create [-1,1]^d bounds
    logger.debug("Setting up integration domain.")
    if integration_domain is not None:
        if len(integration_domain) != dim:
            raise ValueError(
                "Dimension and length of integration domain don't match. Should be e.g. dim=1 dom=[[-1,1]]."
            )
        if type(integration_domain) == torch.Tensor:
            return integration_domain
        else:
            return torch.tensor(integration_domain)
    else:
        return torch.tensor([[-1, 1]] * dim)
