import jax
from jax.config import config
config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import grad, hessian, jit
from functools import partial
from .utils import *

@jit
def Psi_plummer(x, lens_params):
    kappa = lens_params[0]
    return kappa / 2 * jnp.log(1 + jnp.linalg.norm(x)**2)

@jit
def Psi_NFW(x, lens_params):
    kappa = lens_params[0]
    x_norm = jnp.linalg.norm(x)
    dim_1 = jnp.ones(x.shape)
    x_safe_low = jnp.where(x_norm<1, x, 0.5*dim_1)
    x_safe_hi = jnp.where(x_norm<1, 2*dim_1, x)
    x_safe_low_norm = jnp.linalg.norm(x_safe_low)
    x_safe_hi_norm = jnp.linalg.norm(x_safe_hi)
    Psi = jnp.where(x_norm<1,
        kappa / 2 * (jnp.log(x_safe_low_norm/2)**2 - jnp.arctanh(jnp.sqrt(1-x_safe_low_norm**2))**2),
        kappa / 2 * (jnp.log(x_safe_hi_norm/2)**2 + jnp.arctan(jnp.sqrt(x_safe_hi_norm**2 - 1))**2))
    return Psi

def Psi_PM(x):
    return jnp.log(jnp.linalg.norm(x))

def make_T_funcs(Psi):

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
    mu = jit(jnp.vectorize(lambda x, lens_params: 1/((1-dPsi_1D(x, lens_params)/x)*(1-ddPsi_1D(x, lens_params))), signature='()->()', excluded=(1,)))

    T_funcs = {'T': T, 
               'dT': dT, 
               'dT_norm': dT_norm, 
                'f': f,
                'T_hess_det': T_hess_det,
                'T_1D': T_1D,
                'dT_1D': dT_1D,
                'ddT_1D': ddT_1D,
                'dPsi_1D': dPsi_1D,
                'ddPsi_1D': ddPsi_1D,
                'dPsi_1D_param_free': dPsi_1D_param_free,
                'ddPsi_1D_param_free': ddPsi_1D_param_free,
                'mu': mu}

    return T_funcs