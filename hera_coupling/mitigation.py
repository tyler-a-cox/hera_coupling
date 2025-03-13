
import tqdm

import jax
jax.config.update("jax_enable_x64", True)
import optax
import jaxopt
from jax import numpy as jnp
from typing import Tuple, Union, List
from .modeling import deconvolve_visibilties, select_batch, couple_visibilities
from .loss_functions import log_loss, mean_squared_error

@jax.jit
def deconv_loss_function(
    parameters: dict, 
    data_fft: jnp.ndarray, 
    idx: jnp.ndarray, 
    mask: jnp.ndarray, 
    window: jnp.ndarray, 
    min_val: float, 
    lamb: float
) -> float:
    """
    Loss function for FFT version of mutual coupling solver
    
    Parameters:
    ----------
        parameters : dict
            Dictionary of parameters containing keys "coupling"
        data_fft : jnp.ndarray
            FFT of data
        idx : jnp.ndarray
            Index of data to consider
        mask : jnp.ndarray
            Mask applied to data
        window : jnp.ndarray
            Window function applied to data
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

    # Consider diff-ing the squares of the deconvolved data and data
    diff_fft = jnp.fft.fft(data_deconv[:, :, idx[:, 0], idx[:, 1]] * window, axis=0) # just minimize deconvolved data
    return lamb * jnp.mean(jnp.square(jnp.abs(diff - 1))) + (1 - lamb) * log_loss(diff_fft, alpha, min_val)

@jax.jit
def first_order_loss_function(parameters, visibility_diff, v0, nsamples, key):
    """
    Stochastic loss function for the mutual coupling solver.
    
    Parameters:
    parameters (dict): Parameters
    amat (array-like): Amplitude matrix
    v0 (array-like): Visibilities
    key (jax.random.PRNGKey): Random key

    Returns:
    float: Loss value
    """
    # Get the number of times
    ntimes = v0.shape[0]

    # Randomly select a subset of the data
    batch_indices = jax.random.choice(key=key, a=ntimes, shape=(nsamples,), replace=False)
    
    # Select the subset of data from the batch
    v0subset = select_batch(v0, batch_indices)
    amat_subset = select_batch(visibility_diff, batch_indices)

    # Couple the visibilities
    a_est = couple_visibilities(parameters['coupling'], v0subset)

    return mean_squared_error(amat_subset, a_est)

@jax.jit
def second_order_loss_function(parameters, v1, v0, nsamples, key):
    """
    Stochastic loss function for the mutual coupling solver.
    
    Parameters:
    parameters (dict): Parameters
    amat (array-like): Amplitude matrix
    v0 (array-like): Visibilities
    key (jax.random.PRNGKey): Random key

    Returns:
    float: Loss value
    """
    # Get the number of times
    ntimes = v0.shape[0]

    # Randomly select a subset of the data
    batch_indices = jax.random.choice(key=key, a=ntimes, shape=(nsamples,), replace=False)
    
    # Select the subset of data from the batch
    v0subset = select_batch(v0, batch_indices)
    v1subset = select_batch(v1, batch_indices)

    # Couple the visibilities
    a_est = couple_visibilities(parameters['coupling'], v0subset)

    return mean_squared_error(v1subset, a_est)
        
def deconvolve_redundantly_averaged(
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
            fun=deconv_loss_function, 
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
            loss_value, grads = jax.value_and_grad(deconv_loss_function)(
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
    

def decouple_non_redundantly_averaged(
    parameters: dict, 
    v0: jnp.ndarray,
    v1: jnp.ndarray,
    maxiter: int = 100, 
    use_LBFGS: bool = True, 
    optimizer: optax.GradientTransformation = None,
    tol: float = 1e-6,
    verbose: bool = False,
    nsamples: int = 20,
    key: jax.random.PRNGKey = jax.random.PRNGKey(42)
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
        v0 : jnp.ndarray
            Estimate of the decoupled visibilities. Shape is (ntimes, nants, nants)
        v1 : jnp.ndarray
            Measured visibilities. Shape is (ntimes, nants, nants)
        maxiter : int, optional
            Maximum number of iterations (default: 100)
        use_LBFGS : bool, optional
            Whether to use L-BFGS optimizer (default: True)
        optimizer : optax.GradientTransformation, optional
            Custom optimizer if not using L-BFGS
        tol : float, optional
            Tolerance for optimization convergence (default: 1e-6)
        verbose : bool, optional
            Whether to print verbose output (default: False)
        nsamples : int, optional
            Number of samples to use for stochastic optimization (default: 20)
        key : jax.random.PRNGKey, optional
            Random key for stochastic optimization (default: jax.random.PRNGKey(42))

    Returns:
    --------
    Tuple[dict, Union[dict, List[float]]]
        Optimized parameters and metadata/loss history
    """
    # Compute FFT of input data
    visibility_diff = v1 - v0
    
    # Validate inputs
    if use_LBFGS:        
        # Use L-BFGS optimizer
        solver = jaxopt.LBFGS(
            fun=stochastic_loss_function, 
            tol=tol, 
            maxiter=maxiter,
            verbose=verbose
        )

        solved_params, meta = solver.run(
            parameters, 
            amat=visibility_diff,
            v0=v0,
            key=key,
            nsamples=nsamples
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
            loss_value, grads = jax.value_and_grad(stochastic_loss_function)(
                parameters, 
                amat=visibility_diff, 
                v0=v0,   
                key=key, 
                nsamples=nsamples,
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