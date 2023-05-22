import logging
import warnings
from typing import Callable, Optional, Union

import numpy as np
import torch

from scipy import integrate

logger = logging.getLogger("idesolver")
logger.setLevel(logging.WARNING)#(logging.DEBUG)

import matplotlib.pyplot as plt


import torchcubicspline

from torchcubicspline import(natural_cubic_spline_coeffs, 
                             NaturalCubicSpline)
                             
from torchdiffeq import odeint

    
import math
import numpy as np
from IPython.display import clear_output
from tqdm import tqdm_notebook as tqdm

import matplotlib as mpl
import matplotlib.pyplot as plt

import seaborn as sns
sns.color_palette("bright")
import matplotlib as mpl
import matplotlib.cm as cm

from functools import reduce
from IE_source import kernels, integrators 
from utils import to_np
import random

import torch
from torch import Tensor
from torch import nn
from torch.nn  import functional as F
from torch.autograd import Variable
from joblib import Parallel, delayed

use_cuda = torch.cuda.is_available()


if torch.cuda.is_available():  
    device = "cuda:0" 
else:  
    device = "cpu"
    

def global_error(y1: torch.Tensor, y2: torch.Tensor) -> float:
    """
    The default global error function.

    The estimate is the square root of the sum of squared differences between `y1` and `y2`.

    Parameters
    ----------
    y1 : :class:`numpy.ndarray`
        A guess of the solution.
    y2 : :class:`numpy.ndarray`
        Another guess of the solution.

    Returns
    -------
    error : :class:`float`
        The global error estimate between `y1` and `y2`.
    """
    diff = y1 - y2
    return torch.sqrt(torch.dot(diff.flatten(), diff.flatten()))






class IESolver_monoidal:
    
    def __init__(
        self,
        x: torch.Tensor,
        dim: int = 2,
        c: Optional[Callable] = None,
        d: Optional[Callable] = None,
        k: Optional[Callable] = None,
        f: Optional[Callable] = None,
        G: Optional[Callable] = None,
        lower_bound: Optional[Callable] = None,
        upper_bound: Optional[Callable] = None,
        global_error_tolerance: float = 1e-6,
        max_iterations: Optional[int] = None,
        integration_dim: int = 0,
        mc_samplings = 1000,
        num_internal_points=100,
        kernel_split: bool = True,
        kernel_nn: bool = False,
        G_nn: bool = False,
        Last_grad_only: bool = True,
        int_atol: float = 1e-5,
        int_rtol: float = 1e-5,
        #interpolation_kind: str = "cubic",
        smoothing_factor: float = 0.5,
        store_intermediate_y: bool = False,
        global_error_function: Callable = global_error,
    ):
        
        self.mc = integrators.MonteCarlo()
        
        self.dim = dim
        
        self.x = x
        
        self.samp = num_internal_points
        
        if num_internal_points is not None:
            x_d = torch.sort(torch.rand(self.samp).to(device),0)[0]*(self.x[-1]-self.x[0])

            self.x_d = torch.cat([torch.Tensor([self.x[0]]).to(device),x_d,torch.Tensor([self.x[-1]]).to(device)])
        else:
            self.x_d = self.x
        
        self.integration_dim = integration_dim
        
        self.mc_samplings = mc_samplings
        
        self.Last_grad_only = Last_grad_only

        if c is None:
            c = lambda x: self._zeros()
        if d is None:
            d = lambda x, y: torch.Tensor([1])
        if k is None:
            k = lambda x, s: torch.Tensor([1])
        if f is None:
            f = lambda y: self._zeros()
            
            
        self.c = lambda x: c(x)
        self.d = lambda x, y: d(x,y)
        if kernel_nn is True:
            self.k = k
        else:
            self.k = lambda x, s: k(x, s)
        self.f = lambda y: f(y)
        
        if G_nn is True:
            self.G = G
        elif G_nn is False and G is not None:
            self.G = lambda y, x, s: G(y,x,s)
        
        self.kernel_split = kernel_split
        self.kernel_nn = kernel_nn
        self.G_nn = G_nn
                         
        self.Last_iter = False
                
        if lower_bound is None:
            lower_bound = lambda x: self.x[0]
        if upper_bound is None:
            upper_bound = lambda x: self.x[-1]
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        
        
        
        if global_error_tolerance == 0 and max_iterations is None:
            raise exceptions.InvalidParameter("global_error_tolerance cannot be 0 if max_iterations is None")
        if global_error_tolerance < 0:
            raise exceptions.InvalidParameter("global_error_tolerance cannot be negative")
        self.global_error_tolerance = global_error_tolerance
        self.global_error_function = global_error_function
        
        
        #self.interpolation_kind = interpolation_kind

        if not 0 < smoothing_factor < 1:
            raise exceptions.InvalidParameter("Smoothing factor must be between 0 and 1")
        self.smoothing_factor = smoothing_factor

        
        
        if max_iterations is not None and max_iterations <= 0:
            raise exceptions.InvalidParameter("If given, max iterations must be greater than 0")
        
        
        self.max_iterations = max_iterations

        self.int_atol = int_atol
        self.int_rtol = int_rtol

        self.store_intermediate = store_intermediate_y
        if self.store_intermediate:
            self.y_intermediate = []

        self.iteration = None
        self.y = None
        self.global_error = None
        
    def _zeros(self) -> torch.Tensor:
        return torch.zeros(self.dim).to(device)   
    
    
    
    def solve(self, callback: Optional[Callable] = None) -> torch.Tensor:
            
        with warnings.catch_warnings():
                    warnings.filterwarnings(
                    action="error",
                    message="Casting complex values",
                    category=np.ComplexWarning,
                    )
            
            
            #try:
                    y_current = self._initial_y()
                    y_guess = self._solve_rhs_with_known_y(y_current) 
                    error_current = self._global_error(y_current, y_guess)
                    
                    #if self.store_intermediate:
                    #    self.y_intermediate.append(y_current)
                    

                    self.iteration = 0

                    logger.debug(
                        f"Advanced to iteration {self.iteration}. Current error: {error_current}."
                    )
                    
                    if callback is not None:
                        logger.debug(f"Calling {callback} after iteration {self.iteration}")
                        callback(self, y_guess, error_current)
                    
                    #if error_current <= self.global_error_tolerance:
                    #    warnings.warn(
                    #            f"Error less than tolerance at iteration {self.iteration}",
                    #            #exceptions.IDEConvergenceWarning,
                    #        )
                    #    
                    #    self.Last_iter = True
                    #        
                    #    new_current = self._next_y(y_current, y_guess)
                    #    new_guess = self._solve_rhs_with_known_y(new_current)
                    # 
                    #    y_current, y_guess = (
                    #        new_current,
                    #        new_guess
                    #        )
                        
                        
                    #else:
                    while True:

                        ### To check if IDE solver uses GPU
                        #GPUtil.showUtilization()
                        ###

                        new_current = self._next_y(y_current, y_guess)
                        new_guess = self._solve_rhs_with_known_y(new_current)
                        new_error = self._global_error(new_current, new_guess)
                        if new_error > error_current:
                            warnings.warn(
                                f"Error increased on iteration {self.iteration}",
                                #exceptions.IDEConvergenceWarning,
                            )

                        y_current, y_guess, error_current = (
                            new_current,
                            new_guess,
                            new_error,
                            )


                        if self.store_intermediate:
                            self.y_intermediate.append(y_current)

                        self.iteration += 1


                        logger.debug(
                        f"Advanced to iteration {self.iteration}. Current error: {error_current}."
                        )

                        if callback is not None:
                            logger.debug(f"Calling {callback} after iteration {self.iteration}")
                            callback(self, y_guess, error_current)

                        if (self.max_iterations is not None and self.iteration > self.max_iterations) or error_current < self.global_error_tolerance:

                            self.Last_iter = True

                            new_current = self._next_y(y_current, y_guess)
                            new_guess = self._solve_rhs_with_known_y(new_current)

                            y_current, y_guess = (
                                new_current,
                                new_guess
                                )

                            if self.max_iterations is not None and self.iteration < self.max_iterations:
                                warnings.warn(
                                f"Needed less iterations ({self.iteration}) than {self.max_iterations}",
                                #exceptions.IDEConvergenceWarning,
                                )

                            break
            
            
            #except (np.ComplexWarning, TypeError) as e:
             #     print("There has been an error")

        
        
        self.y = y_guess
        self.global_error = error_current

        return self.y
    
    
    
    
    def _initial_y(self) -> torch.Tensor:
        """Calculate the initial guess for `y`, by considering only `c` on the right-hand side of the IDE."""
        return self._solve_ie(self.c)

    
    
    
    def _next_y(self, curr: torch.Tensor, guess: torch.Tensor) -> torch.Tensor:
        """Calculate the next guess at the solution by merging two guesses."""
        return (self.smoothing_factor * curr) + ((1 - self.smoothing_factor) * guess)

    
    
    
    def _global_error(self, y1: torch.Tensor, y2: torch.Tensor) -> float:
        """
        Return the global error estimate between `y1` and `y2`.

        Parameters
        ----------
        y1
            A guess of the solution.
        y2
            Another guess of the solution.

        Returns
        -------
        error : :class:`float`
            The global error estimate between `y1` and `y2`.
        """
        return self.global_error_function(y1, y2)

    
    
    
    def _solve_rhs_with_known_y(self, y: torch.Tensor) -> torch.Tensor:
        """Solves the right-hand-side of the IDE as if :math:`y` was `y`."""
        interpolated_y = self._interpolate_y(y)

        #mc = MonteCarlo()

        def integral(x):
            number_MC_samplings = self.mc_samplings
            x = x.to(device)
            
            def integrand(s):
                
                if self.samp is not None and self.Last_grad_only is True:
                    s = s.to(device).detach().requires_grad_(True)
                else:
                    s = s.to(device).requires_grad_(True)
                if self.kernel_split is True:
                    if self.kernel_nn is False:
                        out = torch.bmm(self.k(x, s), self.f(interpolated_y(s[:])).reshape(number_MC_samplings,self.dim,1))
                    else:
                        y_in = self.f(interpolated_y(s[:])).reshape(number_MC_samplings,self.dim)
                        out = self.k.forward(y_in,x.repeat(number_MC_samplings).reshape(number_MC_samplings,1),s)
                        out = out.unsqueeze(2)
                else:
                    if self.G_nn is False:
                        out = self.G(interpolated_y(s[:]),x,s)
                    else:
                        out = self.G.forward(interpolated_y(s[:]),x.repeat(number_MC_samplings).reshape(number_MC_samplings,1),s)
                        out = out.unsqueeze(2)
                return out
                
            ####
            if self.lower_bound(x) < self.upper_bound(x):
                interval = [[self.lower_bound(x),self.upper_bound(x)]]
            else: 
                interval = [[self.upper_bound(x),self.lower_bound(x)]]
            ####
            

            return self.mc.integrate(
                           fn= lambda s: torch.sign(self.upper_bound(x)-self.lower_bound(x))*integrand(s)[:,:self.dim,0],
                           dim= 1,
                           N=number_MC_samplings,
                           integration_domain = interval, #[[self.lower_bound(x),self.upper_bound(x)]]
                           out_dim = self.integration_dim,
                           )
        
        
        

        def rhs(x):
            if self.samp is not None and self.Last_grad_only is True:
                x = x.to(device).detach().requires_grad_(True)
            else:
                x = x.to(device).requires_grad_(True)
            #y = y.to(device).detach().requires_grad_(True)
            
            out = self.c(x).to(device) + (self.d(x,interpolated_y(x).to(device)).to(device)*integral(x).to(device)).to(device)
            return out
            
        return self._solve_ie(rhs)

    
    
    def _interpolate_y(self, y: torch.Tensor):# -> torch.Tensor: #inter.interp1d:
        """
        Interpolate `y` along `x`, using `interpolation_kind`.

        Parameters
        ----------
        y : :class:`numpy.ndarray`
            The y values to interpolate (probably a guess at the solution).

        Returns
        -------
        interpolator : :class:`scipy.interpolate.interp1d`
            The interpolator function.
        """

        x=self.x_d
        
        y = y
        coeffs = natural_cubic_spline_coeffs(x, y)
        interpolation = NaturalCubicSpline(coeffs)
        
        def output(point:torch.Tensor):
            return interpolation.evaluate(point.to(device))
        
        return output

    
    
    def _solve_ie(self, rhs: Callable) -> torch.Tensor:
        
        
        fun = rhs
        
        if self.Last_iter is False:
            ts = self.x_d
        else:
            ts = self.x
            
        def process(i):
            return fun(torch.Tensor([ts[i]])).unsqueeze(0)
        all_evals = Parallel(n_jobs=1)(delayed(process)(i) for i in range(ts.size(0)))
        
        sol = torch.cat(all_evals,0)

        return sol
