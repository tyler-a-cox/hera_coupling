
import tqdm

import jax
import optax
from jax import numpy as jnp
from typing import Tuple, Union, List
from .modeling import deconvolve_visibilties

@jax.jit
def _loss_function_(parameters: dict, data: jnp.ndarray, data_fft: jnp.ndarray, idx: jnp.ndarray, mask: jnp.ndarray, window: jnp.ndarray, min_val: float, lamb: float):
    """
    Loss function for FFT version of mutual coupling solver
    
    Parameters:
    ----------
        parameters : dict
            Dictionary of parameters containing keys "coupling"
        data : jnp.ndarray
            Data in grid
        min_val : float
            Minimum value in 
        lamb : float
            Weighting of relative terms in the loss function
        
    Return:
    ------
        Value of loss function
    
    """
    # Deconvolve coupling data using FFT
    data_deconv = deconvolve_visibilties(
        parameters=parameters, data_fft=data_fft, mask=mask
    )
    diff = (data_deconv[:, :, idx[:, 0], idx[:, 1]] - data[:, :, idx[:, 0], idx[:, 1]]) * mask[:, :, idx[:, 0], idx[:, 1]] # Consider diff-ing the squares of the deconvolved data and data
    #gain = jnp.abs(parameters['coupling'][:, :, 0, 0] - 1.0).sum()

    diff_fft = jnp.fft.fft(data_deconv[:, :, idx[:, 0], idx[:, 1]] * window, axis=0) # just minimize deconvolved data
    windowing_term = jnp.mean(jnp.log10(jnp.abs(diff_fft) + min_val) * mask[:, :, idx[:, 0], idx[:, 1]])
    #return lamb * gain + (1 - lamb) * (windowing_term - jnp.log10(min_val))
    return lamb * jnp.sqrt(jnp.mean(jnp.square(jnp.abs(diff)))) + (1 - lamb) * (windowing_term - jnp.log10(min_val))
        
def optimize(
    parameters: dict, 
    grid_data: jnp.ndarray, 
    mask: jnp.ndarray, 
    window: jnp.ndarray, 
    min_val: float = 30.0, 
    lamb: float = 1e-3, 
    maxiter: int = 100, 
    use_LBFGS: bool = True, 
    optimizer: optax.GradientTransformation = None,
    tol: float = 1e-6,
    verbose: bool = False
) -> Tuple[dict, Union[dict, List[float]]]:
    """
    Optimize parameters using either L-BFGS or a custom optimizer.

    This function supports two optimization strategies:
    1. L-BFGS (recommended for most cases)
    2. Custom optimizer with manual gradient descent

    Parameters:
    -----------
        parameters : dict
            Initial parameters to be optimized
        grid_data : jnp.ndarray
            Input grid data for optimization
        mask : jnp.ndarray
            Mask applied during optimization
        window : jnp.ndarray
            Window function applied to the data
        min_val : float, optional
            Minimum value constraint (default: 30.0)
        lamb : float, optional
            Regularization parameter (default: 1e-3)
        maxiter : int, optional
            Maximum number of iterations (default: 100)
        use_LBFGS : bool, optional
            Whether to use L-BFGS optimizer (default: True)
        optimizer : optax.GradientTransformation, optional
            Custom optimizer if not using L-BFGS
        tol : float, optional
            Tolerance for optimization convergence (default: 1e-6)

    Returns:
    --------
    Tuple[dict, Union[dict, List[float]]]
        Optimized parameters and metadata/loss history
    """
    # Compute FFT of input data
    data_fft = jnp.fft.fft2(grid_data)
    
    # Validate inputs
    if use_LBFGS:        
        # Use L-BFGS optimizer
        solver = jaxopt.LBFGS(
            fun=loss_function, 
            tol=tol, 
            maxiter=maxiter,
            verbose=verbose
        )

        solved_params, meta = solver.run(
            parameters, 
            data=grid_data, 
            data_fft=data_fft, 
            mask=mask, 
            window=window, 
            min_val=min_val, 
            lamb=lamb
        )

        return solved_params, meta

    else:
        # Custom optimizer gradient descent
        if optimizer is None:
            raise ValueError("Must provide an optimizer when use_LBFGS is False")
        
        opt_state = optimizer.init(parameters)
        loss_history = []
        
        for nit in tqdm.tqdm(range(maxiter), desc="Optimization Progress"):
            # Compute loss and gradients
            loss_value, grads = jax.value_and_grad(loss_function)(
                parameters, 
                grid_data, 
                data_fft, 
                mask, 
                window, 
                min_val, 
                lamb
            )
            
            # Update parameters
            updates, opt_state = optimizer.update(grads, opt_state)
            parameters = optax.apply_updates(parameters, updates)
            
            # Track loss history
            loss_history.append(loss_value)
            
            # Optional early stopping
            if len(loss_history) > 1 and abs(loss_history[-1] - loss_history[-2]) < tol:
                if verbose:
                    print(f"Converged after {nit+1} iterations")
                break
        
        return parameters, loss_history
            

def _optimize_(
    parameters: dict, 
    grid_data: jnp.ndarray, 
    idx: jnp.ndarray,
    mask: jnp.ndarray, 
    window: jnp.ndarray, 
    min_val: float = 30.0, 
    lamb: float = 1e-3, 
    maxiter: int = 100, 
    use_LBFGS: bool = True, 
    optimizer: optax.GradientTransformation = None,
    tol: float = 1e-6,
    verbose: bool = False
) -> Tuple[dict, Union[dict, List[float]]]:
    """
    Optimize parameters using either L-BFGS or a custom optimizer.

    This function supports two optimization strategies:
    1. L-BFGS (recommended for most cases)
    2. Custom optimizer with manual gradient descent

    Parameters:
    -----------
        parameters : dict
            Initial parameters to be optimized
        grid_data : jnp.ndarray
            Input grid data for optimization
        mask : jnp.ndarray
            Mask applied during optimization
        window : jnp.ndarray
            Window function applied to the data
        min_val : float, optional
            Minimum value constraint (default: 30.0)
        lamb : float, optional
            Regularization parameter (default: 1e-3)
        maxiter : int, optional
            Maximum number of iterations (default: 100)
        use_LBFGS : bool, optional
            Whether to use L-BFGS optimizer (default: True)
        optimizer : optax.GradientTransformation, optional
            Custom optimizer if not using L-BFGS
        tol : float, optional
            Tolerance for optimization convergence (default: 1e-6)

    Returns:
    --------
    Tuple[dict, Union[dict, List[float]]]
        Optimized parameters and metadata/loss history
    """
    # Compute FFT of input data
    data_fft = jnp.fft.fft2(grid_data)
    
    # Validate inputs
    if use_LBFGS:        
        # Use L-BFGS optimizer
        solver = jaxopt.LBFGS(
            fun=_loss_function_, 
            tol=tol, 
            maxiter=maxiter,
            verbose=verbose
        )

        solved_params, meta = solver.run(
            parameters, 
            data=grid_data, 
            data_fft=data_fft, 
            idx=idx,
            mask=mask, 
            window=window, 
            min_val=min_val, 
            lamb=lamb
        )

        return solved_params, meta

    else:
        # Custom optimizer gradient descent
        if optimizer is None:
            raise ValueError("Must provide an optimizer when use_LBFGS is False")
        
        opt_state = optimizer.init(parameters)
        loss_history = []
        
        for nit in tqdm.tqdm(range(maxiter), desc="Optimization Progress"):
            # Compute loss and gradients
            loss_value, grads = jax.value_and_grad(_loss_function_)(
                parameters, 
                grid_data, 
                data_fft, 
                idx,
                mask, 
                window, 
                min_val, 
                lamb
            )
            
            # Update parameters
            updates, opt_state = optimizer.update(grads, opt_state)
            parameters = optax.apply_updates(parameters, updates)
            
            # Track loss history
            loss_history.append(loss_value)
            
            # Optional early stopping
            if len(loss_history) > 1 and abs(loss_history[-1] - loss_history[-2]) < tol:
                if verbose:
                    print(f"Converged after {nit+1} iterations")
                break
        
        return parameters, loss_history