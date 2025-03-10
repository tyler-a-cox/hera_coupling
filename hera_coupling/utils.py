import jax
from jax import numpy as jnp
jax.config.update("jax_enable_x64", True)

def project_coordinates(antpair, antpos, ratio: int=3):
    """
    Projects the coordinates of an antenna pair onto a coordinate system defined by the antenna positions.
    Specific to the HERA array.
    Parameters:
    ----------
        antpair (tuple): 
            A tuple containing the indices of the antenna pair.
        antpos (dictionary): 
            An array of antenna positions.
        ratio (int, optional): 
            A scaling factor for the unit vectors. Default is 3.
    Returns:
    -------
        jnp.array: A 2-element array containing the projected east and north coordinates.

    """
    unit_ew = (antpos[1] - antpos[0]) / ratio
    unit_ns = (antpos[11] - antpos[0]) / ratio
    unit_vec_ns = unit_ns - jnp.dot(unit_ns, unit_ew) / jnp.linalg.norm(unit_ew) ** 2 * unit_ew
    
    apidx1, apidx2 = antpair
    vec = antpos[apidx2] - antpos[apidx1]
    north = jnp.dot(vec, unit_vec_ns) / jnp.linalg.norm(unit_vec_ns) ** 2
    east = jnp.dot(vec - north * unit_ns, unit_ew) / jnp.linalg.norm(unit_ew) ** 2
    return jnp.array([east, north])