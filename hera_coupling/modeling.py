import jax
jax.config.update("jax_enable_x64", True)
from jax import numpy as jnp
from jax import random
key = random.key(42)

@jax.jit
def deconvolve_visibilties(parameters, data_fft):
    """
    Deconvolve visibilities using the provided parameters and FFT of the data.

    Parameters:
    parameters (dict): Parameters
    data_fft (array-like): FFT of the data

    Returns:
    array-like: Deconvolved visibilities
    """
    model = jnp.fft.ifft2(data_fft / jnp.fft.fft2(parameters["coupling"]))
    return model

@jax.jit
def couple_visibilities(visibilities, coupling_matrix):
    """
    Couple visibilities using the provided coupling matrix.

    Parameters:
    visibilities (array-like): Visibilities
    coupling_matrix (array-like): Coupling matrix

    Returns:
    array-like: Coupled visibilities
    """
    coupled_visibilities = (
        jnp.einsum(
            "...mn,in->...mi", visibilities, coupling_matrix.conj()
            ) + 
        jnp.einsum(
            "mn,...in->...mi", coupling_matrix, visibilities.conj()
            )
    )
    return coupled_visibilities

@jax.jit
def select_batch(v0, batch_indices):
    return v0[batch_indices]