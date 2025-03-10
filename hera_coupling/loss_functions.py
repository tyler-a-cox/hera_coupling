import jax
from jax import numpy as jnp


@jax.jit
def scaled_log1p(x, alpha=1.0, floor=0.0):
    """
    Apply the scaled log1p transformation: f(x) = (1/alpha)*log(1 + alpha*x).
    
    Parameters:
    x (float): Input value
    alpha (float): Scaling factor
    floor (float): Floor value

    Returns:
    float: Transformed value
    """
    return jnp.maximum(jnp.log1p(alpha * x) / jnp.log(alpha), floor)

@jax.jit
def mean_squared_error(y_true, y_pred):
    """
    Calculate the Mean Squared Error (MSE) between true and predicted values.

    Parameters:
    y_true (array-like): True values
    y_pred (array-like): Predicted values

    Returns:
    float: Mean Squared Error
    """
    mse = jnp.mean(jnp.abs(y_true - y_pred) ** 2)
    return mse

@jax.jit
def log_loss(difference, alpha=1.0, min_val=0.0):
    """
    Calculate the Logarithmic Loss (Log Loss) between true and predicted values.

    Parameters:
    y_true (array-like): True values (binary: 0 or 1)
    y_pred (array-like): Predicted probabilities
    eps (float): Small value to avoid division by zero or log(0)

    Returns:
    float: Logarithmic Loss
    """
    #y_pred = np.clip(y_pred, eps, 1 - eps)
    #logloss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    logloss = jnp.mean(
        scaled_log1p(
            jnp.abs(difference), 
            alpha=alpha, 
            min_val=min_val
        )
    )
    return logloss