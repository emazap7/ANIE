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

# import seaborn as sns
# sns.color_palette("bright")
import matplotlib as mpl
import matplotlib.cm as cm

from functools import reduce
from IE_source import kernels, integrators 
from IE_source.utils import to_np
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


class masking_function():
    def __init__(self,lower_bound,upper_bound,n_batch=1):
        self.lower_bound=lower_bound
        self.upper_bound=upper_bound
        self.n_batch=n_batch
        
    def create_mask(self,x):
        bounds = interval_function(self.lower_bound, self.upper_bound, x)
        
        masking_matrix = torch.cat([bounds.create_mask_interval(x[i]).unsqueeze(0) for i in range(x.size(0))])
        
        masking_matrix=masking_matrix.unsqueeze(0).repeat(self.n_batch,1,1)
        return masking_matrix


class interval_function():
    def __init__(self,lower_bound,upper_bound,time_int):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.time_int = time_int

    def create_interval(self,x):
            values = torch.Tensor([])
            for i in range(0,self.time_int.size(0)):
                if self.time_int[i]>= self.lower_bound(x) and self.time_int[i]<= self.upper_bound(x):
                    values = torch.cat([values,torch.Tensor([self.time_int[i]])])
                else:
                    #pass
                    values = torch.cat([values,torch.Tensor([0.])])
            return values
        
    def create_mask_interval(self,x):
            values = torch.Tensor([])
            for i in range(0,self.time_int.size(0)):
                if self.time_int[i]>= self.lower_bound(x) and self.time_int[i]<= self.upper_bound(x):
                    values = torch.cat([values,torch.Tensor([1.])])
                else:
                    #pass
                    values = torch.cat([values,torch.Tensor([0.])])
            return values 
          
          
    
class Integral_attention_solver_multbatch:
    
    def __init__(
        self,
        x: torch.Tensor,
        y_0: Union[float, np.float64, complex, np.complex128, np.ndarray, list,torch.Tensor,torch.tensor],
        y_init: Optional[Callable] = None,
        c: Optional[Callable] = None,
        d: Optional[Callable] = None,
        Encoder:Optional[Callable] = None,
        lower_bound: Optional[Callable] = None,
        upper_bound: Optional[Callable] = None,
        mask: Optional[Callable] = None,
        sampling_points: int = 100,
        use_support: bool=False,
        n_support_points = 100,
        support_tensors: Optional[Callable] = None,
        global_error_tolerance: float = 1e-6,
        max_iterations: Optional[int] = None,
        int_atol: float = 1e-5,
        int_rtol: float = 1e-5,
        smoothing_factor: float = 0.5,
        store_intermediate_y: bool = False,
        global_error_function: Callable = global_error,
        output_support_tensors=False,
        return_function=False,
        accumulate_grads=False,
        initialization=False
    ):
        
        
        self.y_0 = y_0.to(device)
        
        self.x = x.to(device)
        
        self.y_init = y_init
        
        self.samp = sampling_points
        
        self.use_support=use_support
        
        self.output_support_tensors=output_support_tensors
        
        self.accumulate_grads=accumulate_grads
        
        self.initialization=initialization
        
        self.dim = self.y_0.shape[-1]
        
        self.n_batch = self.y_0.shape[0]
        
        
        
        if use_support is False:
            self.support_tensors=x.to(device)
        
        elif support_tensors is None:
            self.support_tensors=torch.sort(torch.rand(sampling_points))[0]*(x[-1]-x[0])+x[0]
        else:
            self.support_tensors=support_tensors.to(device)
            
        self.return_function=return_function

        if c is None:
            c = lambda x: self.y_0.repeat(1,self.support_tensors.shape[0],1).to(device)
        if d is None:
            d = lambda x: torch.Tensor([1]).to(device)
            
            
        self.c = lambda x: c(x)
        self.d = lambda x: d(x)
 
        if self.initialization is True and y_init is None:
            raise exceptions.InvalidParameter(
                "y_init cannot be None if initialization is True"
            ) 
              
        self.Encoder = Encoder.to(device)
        
        
        if lower_bound is not None and upper_bound is not None:
            self.masking_map =  masking_function(self.lower_bound,self.upper_bound,n_batch=self.y_0.size(0))
            mask_times = torch.sort(torch.rand(self.samp))[0].to(device)*(self.x[-1]-self.x[0]) + self.x[0]
            self.mask = self.masking_map.create_mask(mask_times)
        else:
            self.mask=None #This is a Fredholm IE
            
        if mask is not None:
            self.mask = mask
        
        #self.I_func = interval_function(self.lower_bound,self.upper_bound,self.x)
                
        
        if global_error_tolerance == 0 and max_iterations is None:
            raise exceptions.InvalidParameter(
                "global_error_tolerance cannot be 0 if max_iterations is None"
            )
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
        return torch.zeros_like(self.y_0)    
    
    
    
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

                    while error_current > self.global_error_tolerance:
                        
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

                        if self.max_iterations is not None and self.iteration >= self.max_iterations:
                        #    warnings.warn(
                        #        exceptions.IDEConvergenceWarning(
                        #            f"Used maximum number of iterations ({self.max_iterations}), but only got to global error {error_current} (target {self.global_error_tolerance})"
                        #        )
                        #    )
                        
                        
                            break
            
            
            #except (np.ComplexWarning, TypeError) as e:
             #     print("There has been an error")

        
        self.y = y_guess
        self.global_error = error_current
        
        if self.output_support_tensors is False or self.return_function is True:
            interpolated_y = self._interpolate_y(self.y)
            self.y = interpolated_y(self.x)
            del interpolated_y
            
        if self.return_function is True:

            return interpolated_y
        
        else:
            
            return self.y
    
    
    def _initial_y(self) -> torch.Tensor:
        """Calculate the initial guess for `y`, by considering only `c` on the right-hand side of the IDE.""" 
        if self.initialization is False:
            y_initial = self.c(self.support_tensors) 
        else: 
            y_initial = self.y_init
            
        return y_initial

    
    
    
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
        
        def integral(x):
            
            if self.use_support is True:
                time= torch.sort(torch.rand(self.samp-2,1),0)[0].to(device)*(self.x[-1]-self.x[0]) + self.x[0]
                time= torch.cat([self.x[:1].unsqueeze(-1),time,self.x[-1:].unsqueeze(-1)],0).unsqueeze(0)
                interpolated_y = self._interpolate_y(y)
                y_in = interpolated_y(time.view(self.samp))
            else:
                time= self.support_tensors.view(1,self.x.shape[0],1)
                y_in = y
            
            
            time = time.repeat(self.n_batch,1,1)
            
            y_in = torch.cat([y_in,time],-1)
            
            model = self.Encoder
            
            if self.accumulate_grads is False:
                y_in = y_in.detach().requires_grad_(True)
            else:
                y_in = y_in.requires_grad_(True)
                
            T = model.forward(y_in,dynamical_mask=self.mask)
            ##z = T[T.size(0)-1,T.size(1)-1,:self.y_0.size(1)]
            #z = T[T.size(0)-1,0,:self.y_0.size(1)]
            
            z = T[:,:,:]
            
            
            z = z + y_in[:,:,:]
            
            z = z[:,:,:self.dim]

            del y_in
            del time
            if self.use_support is True: 
                del interpolated_y
            
            return z
        

        def rhs(x):
            
            return self.c(self.support_tensors) + (self.d(x)*integral(x))
            

        return self.solve_IE(rhs)

    
    
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
        x=self.support_tensors
        y = y
        coeffs = natural_cubic_spline_coeffs(x, y)
        interpolation = NaturalCubicSpline(coeffs)
        
        def output(point:torch.Tensor):
            return interpolation.evaluate(point)
        
        return output
    
    def solve_IE(self, rhs: Callable) -> torch.Tensor:
        
        func = rhs 
        
        #sol = torch.cat([func(torch.Tensor([self.x[i]])).unsqueeze(0) for i in range(1,self.x.size(0))],-2).squeeze(0)
        #sol = torch.cat([self.y_0,sol],-2)
        
        times = self.support_tensors
        
        sol = rhs(times)
        
        return sol

class Integral_attention_solver:
    
    def __init__(
        self,
        x: torch.Tensor,
        y_0: Union[float, np.float64, complex, np.complex128, np.ndarray, list,torch.Tensor,torch.tensor],
        c: Optional[Callable] = None,
        d: Optional[Callable] = None,
        Encoder:Optional[Callable] = None,
        lower_bound: Optional[Callable] = None,
        upper_bound: Optional[Callable] = None,
        mask: Optional[Callable] = None,
        sampling_points: int = 100,
        use_support: bool=True,
        n_support_points = 100,
        support_tensors: Optional[Callable] = None,
        global_error_tolerance: float = 1e-6,
        max_iterations: Optional[int] = None,
        int_atol: float = 1e-5,
        int_rtol: float = 1e-5,
        smoothing_factor: float = 0.5,
        store_intermediate_y: bool = False,
        global_error_function: Callable = global_error,
        output_support_tensors=False,
        return_function=False
    ):
        
        
        self.y_0 = y_0.to(device)
        
        self.x = x.to(device)
        
        self.samp = sampling_points
        
        self.use_support=use_support
        
        self.output_support_tensors=output_support_tensors
        
                
        self.dim = self.y_0.size(-1)
        
        
        if use_support is False:
            self.support_tensors=x.to(device)
        
        elif support_tensors is None:
            self.support_tensors=torch.sort(torch.rand(sampling_points))[0]*(x[-1]-x[0])+x[0]
                
        else:
            self.support_tensors=support_tensors.to(device)
            
        self.return_function=return_function

        if c is None:
            c = lambda x: self.y_0.repeat(self.support_tensors.size(0),1).to(device)
        if d is None:
            d = lambda x: torch.Tensor([1]).to(device)
            
            
        self.c = lambda x: c(x)
        self.d = lambda x: d(x)
        
        
        if lower_bound is not None:
            self.lower_bound = lower_bound
        if upper_bound is not None:
            self.upper_bound = upper_bound
        
        
        #self.I_func = interval_function(self.lower_bound,self.upper_bound,self.x)
        
        if lower_bound is not None and upper_bound is not None:
            self.masking_map =  masking_function(self.lower_bound,self.upper_bound,n_batch=1)
            mask_times = torch.sort(torch.rand(self.samp))[0].to(device)*(self.x[-1]-self.x[0]) + self.x[0]
            self.mask = self.masking_map.create_mask(mask_times)
        else:
            self.mask=None #This is a Fredholm IE
            
        if mask is not None:
            self.mask = mask
        
           
        self.Encoder = Encoder.to(device)
        
        
        if global_error_tolerance == 0 and max_iterations is None:
            raise exceptions.InvalidParameter(
                "global_error_tolerance cannot be 0 if max_iterations is None"
            )
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
        return torch.zeros_like(self.y_0)    
    
    
    
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

                    while error_current > self.global_error_tolerance:
                        
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

                        if self.max_iterations is not None and self.iteration >= self.max_iterations:
                        #    warnings.warn(
                        #        exceptions.IDEConvergenceWarning(
                        #            f"Used maximum number of iterations ({self.max_iterations}), but only got to global error {error_current} (target {self.global_error_tolerance})"
                        #        )
                        #    )
                        
                        
                            break
            
            
            #except (np.ComplexWarning, TypeError) as e:
             #     print("There has been an error")

        
        self.y = y_guess
        self.global_error = error_current
        
        if self.output_support_tensors is False or self.return_function is True:
            interpolated_y = self._interpolate_y(self.y)
            self.y = interpolated_y(self.x)
                  
        if self.return_function is True:

            return interpolated_y
        
        else:
            
            del interpolated_y
            
            return self.y
    
    
    def _initial_y(self) -> torch.Tensor:
        """Calculate the initial guess for `y`, by considering only `c` on the right-hand side of the IDE."""
        y_initial = self.c(self.support_tensors.unsqueeze(1)) 
        
        return y_initial

    
    
    
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
        
        def integral(x):
            
            
            #glob_time = torch.cat([self.x[i].repeat(1,self.samp,1) for i in range(self.x.size(0))]).to(device)
            
            ##loc_time = torch.cat([torch.sort(torch.rand(1,self.samp,1),1)[0].to(device)*(self.upper_bound(self.x[i])-self.lower_bound(self.x[i])) + self.lower_bound(self.x[i]) for i in range(self.x.size(0))]).to(device)
            loc_time= torch.sort(torch.rand(1,self.samp,1),1)[0].to(device)*(self.x[-1]-self.x[0]) + self.x[0]
            y_in = interpolated_y(loc_time).squeeze(2)
            
            y_in = torch.cat([y_in,loc_time],-1)
            
            model = self.Encoder
            
            y_in = y_in.detach().requires_grad_(True)
            
            T = model.forward(y_in,dynamical_mask=self.mask)
            ##z = T[T.size(0)-1,T.size(1)-1,:self.y_0.size(1)]
            #z = T[T.size(0)-1,0,:self.y_0.size(1)]
            
            z = T[:,:,:]
            
            
            z = z + y_in[:,:,:]

            
            z = z[0,:,:self.dim]

            del y_in
            del loc_time
            #del interpolated_y
               
            return z
            

        def rhs(x):
            
            return self.c(x.squeeze(0)) + (self.d(x)*integral(x))

        return self.solve_IE(rhs)

    
    
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
        x=self.support_tensors
        y = y
        coeffs = natural_cubic_spline_coeffs(x, y)
        interpolation = NaturalCubicSpline(coeffs)
        
        def output(point:torch.Tensor):
            return interpolation.evaluate(point)
        
        return output
    
    def solve_IE(self, rhs: Callable) -> torch.Tensor:
        
        func = rhs 
        
        #sol = torch.cat([func(torch.Tensor([self.x[i]])).unsqueeze(0) for i in range(1,self.x.size(0))],-2).squeeze(0)
        #sol = torch.cat([self.y_0,sol],-2)
        
        times = self.support_tensors
        
        sol = rhs(times)
        
        return sol


class Integral_spatial_attention_solver_multbatch:
    
    def __init__(
        self,
        x: torch.Tensor,
        y_0: Union[float, np.float64, complex, np.complex128, np.ndarray, list,torch.Tensor,torch.tensor],
        y_init: Optional[Callable] = None,
        c: Optional[Callable] = None,
        d: Optional[Callable] = None,
        Encoder:Optional[Callable] = None,
        lower_bound: Optional[Callable] = None,
        upper_bound: Optional[Callable] = None,
        mask: Optional[Callable] = None,
        sampling_points: int = 100,
        use_support: bool=False,
        n_support_points = 100,
        support_tensors: Optional[Callable] = None,
        spatial_integration: bool=False,
        spatial_domain: Optional[Callable] = None,
        spatial_domain_dim: int = 2,
        spatial_dimensions = None,
        global_error_tolerance: float = 1e-6,
        max_iterations: Optional[int] = None,
        int_atol: float = 1e-5,
        int_rtol: float = 1e-5,
        smoothing_factor: float = 0.5,
        store_intermediate_y: bool = False,
        global_error_function: Callable = global_error,
        output_support_tensors=False,
        return_function=False,
        accumulate_grads=True,
        initialization=False 
    ):
        
        
        self.y_0 = y_0.to(device)
        
        self.x = x.to(device)
        
        self.y_init = y_init
        
        self.samp = sampling_points
        
        self.use_support=use_support
        
        self.output_support_tensors=output_support_tensors
        
        self.spatial_integration = spatial_integration
        
        self.spatial_domain_dim = spatial_domain_dim
        
        self.spatial_dimensions = spatial_dimensions
        
        self.accumulate_grads = accumulate_grads
        
        self.initialization = initialization
            
                
        self.dim = self.y_0.shape[-1]
        self.n_batch= self.y_0.shape[0]
        
        
        if spatial_integration is False: 
            if use_support is False:
                self.support_tensors=x.to(device)

            elif support_tensors is None:
                self.support_tensors=torch.sort(torch.rand(sampling_points))[0]*(x[-1]-x[0])+x[0]

            else:
                self.support_tensors=support_tensors.to(device)
         
        
        if spatial_integration is True: 
            if spatial_domain is None:
                spatial_domain_i = \
                torch.meshgrid([torch.linspace(0,1,sampling_points) for i in range(self.spatial_domain_dim)])[0]
                self.spatial_domain = torch.cat([spatial_domain_i[i].flatten().unsqueeze(-1)\
                                            for i in range(len(spatial_domain_i))],-1)
            else:
                self.spatial_domain = spatial_domain
        
            D_shape = [1,1] + [sampling_points]
            
            self.support_tensors = self.spatial_domain.view\
                          ([self.spatial_domain.shape[i] for i in range(2)]+[1])\
                           .repeat(D_shape)
            
            self.mesh_side = int(torch.Tensor([self.spatial_domain.size(0)])\
                          .pow(1/self.spatial_domain_dim).item())
        
            self.volume = self.spatial_domain.shape[0]
            
        
        self.return_function=return_function
        

        if c is None:
            if spatial_integration is False:
                c = lambda x: self.y_0.repeat(self.support_tensors.size(0),1).to(device)
            else:
                if self.spatial_dimensions is None:
                    c = lambda x: self.y_0.view([self.n_batch]\
                                            +[self.mesh_side for i in range(len(D_shape)-1)]\
                                            +[1]+[self.dim])\
                        .repeat([1]\
                        +[1 for i in range(len(D_shape)-1)]\
                        +[sampling_points]+[1]).to(device)
                else:
                    c = lambda x: self.y_0.view([self.n_batch]\
                                            +self.spatial_dimensions\
                                            +[1]+[self.dim])\
                        .repeat([1]\
                        +[1 for i in range(len(D_shape)-1)]\
                        +[sampling_points]+[1]).to(device)

        if d is None:
            d = lambda x: torch.Tensor([1]).to(device)
            
            
        self.c = lambda x: c(x)
        self.d = lambda x: d(x)
        
        if self.initialization is True and y_init is None:
            raise exceptions.InvalidParameter(
                "y_init cannot be None if initialization is True"
            ) 
        
        
        if lower_bound is not None:
            self.lower_bound = lower_bound
        if upper_bound is not None:
            self.upper_bound = upper_bound
        
        
        #self.I_func = interval_function(self.lower_bound,self.upper_bound,self.x)
        
        if lower_bound is not None and upper_bound is not None:
            self.masking_map =  masking_function(self.lower_bound,self.upper_bound,n_batch=1)
            mask_times = torch.sort(torch.rand(self.samp))[0].to(device)*(self.x[-1]-self.x[0]) + self.x[0]
            self.mask = self.masking_map.create_mask(mask_times)
        else:
            self.mask=None #This is a Fredholm IE
            
        if mask is not None:
            self.mask = mask
        
           
        self.Encoder = Encoder.to(device)
        
        
        if global_error_tolerance == 0 and max_iterations is None:
            raise exceptions.InvalidParameter(
                "global_error_tolerance cannot be 0 if max_iterations is None"
            )
        if global_error_tolerance < 0:
            raise exceptions.InvalidParameter("global_error_tolerance cannot be negative")
        self.global_error_tolerance = global_error_tolerance
        self.global_error_function = global_error_function
        
        
        #self.interpolation_kind = interpolation_kind

        if not 0 <= smoothing_factor < 1:
            raise exceptions.InvalidParameter("Smoothing factor must be in [0,1)")
        self.smoothing_factor = smoothing_factor

        
        
        if max_iterations is not None and max_iterations <= 0:
            raise exceptions.InvalidParameter("If given, max iterations must be > 0")
        
        
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
        return torch.zeros_like(self.y_0)    
    
    
    
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

                    while error_current > self.global_error_tolerance:
                        
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

                        if self.max_iterations is not None and self.iteration >= self.max_iterations:
                        #    warnings.warn(
                        #        exceptions.IDEConvergenceWarning(
                        #            f"Used maximum number of iterations ({self.max_iterations}), but only got to global error {error_current} (target {self.global_error_tolerance})"
                        #        )
                        #    )
                        
                        
                            break
            
            
            #except (np.ComplexWarning, TypeError) as e:
             #     print("There has been an error")

        
        self.y = y_guess
        self.global_error = error_current
        
        if self.use_support is True:
            if self.output_support_tensors is False or self.return_function is True:
                interpolated_y = self._interpolate_y(self.y)
                self.y = interpolated_y(self.x)
                del interpolated_y
                  
        if self.return_function is True:

            return interpolated_y
        
        else:
            
            return self.y
    
    
    def _initial_y(self) -> torch.Tensor:
        """Calculate the initial guess for `y`, by considering only `c` on the right-hand side of the IDE."""
        if self.initialization is False:
            if self.spatial_integration is False:
                y_initial = self.c(self.support_tensors.unsqueeze(1)) 
            else:
                y_initial = self.c(self.support_tensors)
        else:
            y_initial = self.y_init
        
        return y_initial

    
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
        
        
        def integral(x):
            
            if self.spatial_integration is False:
                time= torch.sort(torch.rand(1,self.samp,1),1)[0].to(device)*(self.x[-1]-self.x[0]) + self.x[0]
                y_in = interpolated_y(time).squeeze(2)

                y_in = torch.cat([y_in,time],-1)

                model = self.Encoder

                y_in = y_in.detach().requires_grad_(True)
                
                T = model.forward(y_in,dynamical_mask=self.mask)
                ##z = T[T.size(0)-1,T.size(1)-1,:self.y_0.size(1)]
                #z = T[T.size(0)-1,0,:self.y_0.size(1)]

                z = T[:,:,:]


                z = z + y_in[:,:,:]

                z = z[0,:,:self.dim]
            
            else:
                
                if self.use_support is True:
                    time = torch.sort(torch.rand(self.samp-2,1),0)[0].to(device)*(self.x[-1]-self.x[0]) + self.x[0]
                    time= torch.cat([self.x[:1].unsqueeze(-1),time,self.x[-1:].unsqueeze(-1)],0).unsqueeze(0)
                    interpolated_y = self._interpolate_y(y)
                    y_ = interpolated_y(time.view(self.samp)) 
                else:
                    time = self.x.view(1,self.samp,1)
                    y_ = y
                
                #space= torch.linspace(0,1,self.volume).unsqueeze(-1).to(device).repeat(1,self.spatial_domain_dim)
                space = self.spatial_domain
                
                space= space.view([1]+[self.spatial_domain.shape[i]\
                                       for i in range(len(self.spatial_domain.shape))])\
                                     .repeat([self.n_batch]+[self.samp]+[1 for i in range(len(self.spatial_domain.shape)-1)])
                                  
                
                y_in = y_.view([self.n_batch,self.samp*self.volume,self.dim])
                
                y_in = torch.cat([y_in,space,time.repeat(self.n_batch,self.volume,1).sort(dim=-2)[0]],-1)
                
                model = self.Encoder
                
                if self.accumulate_grads is False:
                    y_in = y_in.detach().requires_grad_(True)
                else:
                    y_in = y_in.requires_grad_(True)
                
                T = model.forward(y_in,dynamical_mask=self.mask)
                
                
                z = T[:,:,:] + y_in[:,:,:]
                
                z = z[:,:,:self.dim]
                
                if self.spatial_dimensions is not None:
                    z = z[:,:,:].view([self.n_batch]\
                                      +self.spatial_dimensions\
                                      +[self.samp,self.dim])
                else:
                    z = z[:,:,:].view([self.n_batch]\
                                      +[self.mesh_side for i in range(self.spatial_domain_dim)]\
                                      +[self.samp,self.dim])
                
                
            del y_in
            del time
            if self.use_support is True:
                del interpolated_y
               
            return z
            

        def rhs(x):
            
            return self.c(self.support_tensors) + (self.d(x)*integral(x))

        return self.solve_IE(rhs)

    
    
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
        if self.spatial_integration is False:
            x=self.support_tensors
        else:
            x = torch.linspace(self.x[0],self.x[-1],self.samp)
        y = y
        coeffs = natural_cubic_spline_coeffs(x, y)
        interpolation = NaturalCubicSpline(coeffs)
        
        def output(point:torch.Tensor):
            return interpolation.evaluate(point)
        
        return output
    
    def solve_IE(self, rhs: Callable) -> torch.Tensor:
        
        func = rhs 
        
        #sol = torch.cat([func(torch.Tensor([self.x[i]])).unsqueeze(0) for i in range(1,self.x.size(0))],-2).squeeze(0)
        #sol = torch.cat([self.y_0,sol],-2)
        
        times = self.support_tensors
        
        sol = rhs(times)
        
        return sol
    
    
