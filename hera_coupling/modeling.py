import jax
from jax import numpy as jnp

@jax.jit
def frequency_model(basis, coefficients):
    """
    Mapping from basis and coefficients to frequency model.
    """
    raise NotImplementedError(
        "Model for frequency dependence on the coupling not implemented yet."
    )

@jax.jit
def azimuthal_model(basis, coefficients):
    """
    Mapping from basis and coefficients to azimuthal model.
    """
    raise NotImplementedError(
        "Model for azimuth dependence on the coupling not implemented yet."
    )

@jax.jit
def joint_azimuthal_frequency_model(azimuth_basis, frequency_basis, coefficients):
    """
    Joint model for azimuthal and frequency dependence on the coupling.
    """
    return jnp.einsum(
        "ij,jk,lk->il",
        azimuth_basis,
        coefficients,
        frequency_basis,
    )

@jax.jit
def deconvolve_visibilties(parameters, data_fft):
    """
    Deconvolve visibilities using the provided parameters and FFT of the data.
    """
    model = jnp.fft.ifft2(data_fft / jnp.fft.fft2(parameters["coupling"]))
    return model

@jax.jit
def weiner_deconvolution(parameters, data_fft, noise_level):
    """
    Perform Weiner deconvolution on the data.
    """
    coupling_fft = jnp.fft.fft2(parameters["coupling"])
    model = jnp.fft.ifft2(
        (jnp.conj(coupling_fft) / (jnp.abs(coupling_fft) ** 2 + noise_level)) * data_fft
    )
    return model