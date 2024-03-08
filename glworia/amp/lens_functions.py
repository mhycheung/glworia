import jax
from jax import config
config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import grad, hessian, jit
from functools import partial
from .utils import *

from typing import List, Tuple, Union, Optional, Dict, Any, Callable

# @jit
# def Psi_plummer(x, lens_params):
#     kappa = lens_params[0]
#     return kappa / 2 * jnp.log(1 + jnp.linalg.norm(x)**2)

# @jit
# def Psi_NFW(x, lens_params):
#     kappa = lens_params[0]
#     x_norm = jnp.linalg.norm(x)
#     dim_1 = jnp.ones(x.shape)
#     x_safe_low = jnp.where(x_norm<1, x, 0.5*dim_1)
#     x_safe_hi = jnp.where(x_norm<1, 2*dim_1, x)
#     x_safe_low_norm = jnp.linalg.norm(x_safe_low)
#     x_safe_hi_norm = jnp.linalg.norm(x_safe_hi)
#     Psi = jnp.where(x_norm<1,
#         kappa / 2 * (jnp.log(x_safe_low_norm/2)**2 - jnp.arctanh(jnp.sqrt(1-x_safe_low_norm**2))**2),
#         kappa / 2 * (jnp.log(x_safe_hi_norm/2)**2 + jnp.arctan(jnp.sqrt(x_safe_hi_norm**2 - 1))**2))
#     return Psi

# @jit
# def Psi_gSIS(x, lens_params):
#     k = lens_params[0]
#     return jnp.linalg.norm(x)**(2 - k)/(2 - k)

# @jit
# def Psi_CIS(x, lens_params):
#     x_c = jnp.abs(lens_params[0])
#     x_t = jnp.sqrt(x_c**2 + jnp.linalg.norm(x)**2)
#     x_c_safe = jnp.where(x_c > 1e-15, x_c, 1e-15)
#     Psi = jnp.where(x_c > 1e-15, 
#             x_t + x_c_safe * jnp.log(2 * x_c_safe / (x_t + x_c_safe)), 
#             x_t
#                     )
#     return Psi

# def Psi_PM(x, lens_params):
#     return jnp.log(jnp.linalg.norm(x))

def make_T_funcs(Psi: Callable) -> Dict[str, Callable]:
    """
    Construct the relevant functions derived from the Fermat Potential.

    Parameters:
        Psi: the Fermat Potential function.

    Returns:
        T_funcs: a dictionary of the relevant functions.
    """

    rot_90 = jnp.array([[0, -1], [1, 0]])   

    Psi_2D = jnp.vectorize(Psi, signature = '(2)->()', excluded=(1,))
    Psi_1D = jnp.vectorize(Psi, signature = '()->()', excluded=(1,))

    @jit
    @partial(jnp.vectorize, signature='(2),(2)->()', excluded = (2,))
    def T(x, y, lens_params):
        return jnp.linalg.norm(x - y)**2/2 - Psi_2D(x, lens_params)
    
    @jit
    @partial(jnp.vectorize, signature='(),()->()', excluded = (2,))
    def T_1D(x, y, lens_params):
        return jnp.linalg.norm(x - y)**2/2 - Psi_1D(x, lens_params)
    
    @partial(jnp.vectorize, excluded = {3}, signature = '(),(),()->()')
    @partial(jit, static_argnums = (3))
    def T_1D_vec(x, y, lens_params, Psi_1D):
        lens_params = jnp.atleast_1d(lens_params)
        return jnp.linalg.norm(x - y)**2/2 - Psi_1D(x, lens_params)
    
    dT = jit(jnp.vectorize(grad(T), signature='(2)->(2)', excluded = (1,2)))
    dT_norm = jit(jnp.vectorize(lambda x, y, lens_params: jnp.linalg.norm(dT(x, y, lens_params)), signature = '(2)->()', excluded = (1,2)))
    f = jit(jnp.vectorize(lambda x, y, lens_params: rot_90@dT(x, y, lens_params)/dT_norm(x, y, lens_params), signature = '(2)->(2)', excluded = (1,2)))

    T_hess = jit(jnp.vectorize(jax.hessian(T), signature='(2)->(2,2)', excluded = (1,2)))
    T_hess_det = jit(jnp.vectorize(lambda x, y, lens_params: jnp.linalg.det(T_hess(x, y, lens_params)), signature='(2)->()', excluded = (1,2)))

    dT_1D = jit(jnp.vectorize(grad(T_1D), signature='()->()', excluded = (1,2)))
    ddT_1D = jit(jnp.vectorize(grad(dT_1D), signature='()->()', excluded = (1,2)))

    dPsi_1D = jit(jnp.vectorize(grad(Psi_1D), signature='()->()', excluded=(1,)))
    ddPsi_1D = jit(jnp.vectorize(grad(dPsi_1D), signature='()->()', excluded=(1,)))
    dPsi_1D_param_free = jit(jnp.vectorize(grad(Psi_1D), signature='(),(n)->()'))
    ddPsi_1D_param_free = jit(jnp.vectorize(grad(dPsi_1D), signature='(),(n)->()'))
    mu_raw = lambda x, lens_params: 1/((1-dPsi_1D(x, lens_params)/x)*(1-ddPsi_1D(x, lens_params)))
    mu = jit(jnp.vectorize(mu_raw, signature='()->()', excluded=(1,)))
    
    @partial(jnp.vectorize, signature = '(),()->()')
    @jit
    def mu_vec(x, lens_params):
        lens_params = jnp.atleast_1d(lens_params)
        return mu_raw(x, lens_params)

    T_funcs = {'T': T, 
               'dT': dT, 
               'dT_norm': dT_norm, 
                'f': f,
                'T_hess_det': T_hess_det,
                'T_1D': T_1D,
                'T_1D_vec': T_1D_vec,
                'dT_1D': dT_1D,
                'ddT_1D': ddT_1D,
                'Psi_1D': Psi_1D,
                'Psi_2D': Psi_2D,
                'dPsi_1D': dPsi_1D,
                'ddPsi_1D': ddPsi_1D,
                'dPsi_1D_param_free': dPsi_1D_param_free,
                'ddPsi_1D_param_free': ddPsi_1D_param_free,
                'mu': mu,
                'mu_vec': mu_vec}

    return T_funcs