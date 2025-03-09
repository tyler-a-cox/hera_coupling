import jax
from jax import numpy as np


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
    mse = np.mean((y_true - y_pred) ** 2)
    return mse

@jax.jit
def log_loss(y_true, y_pred, noise):
    """
    Calculate the Logarithmic Loss (Log Loss) between true and predicted values.

    Parameters:
    y_true (array-like): True values (binary: 0 or 1)
    y_pred (array-like): Predicted probabilities
    eps (float): Small value to avoid division by zero or log(0)

    Returns:
    float: Logarithmic Loss
    """
    y_pred = np.clip(y_pred, eps, 1 - eps)
    logloss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return logloss