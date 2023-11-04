import jax
from jax.config import config
config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import grad, hessian, jit
from functools import partial
from .utils import *
from .amplification_factor import *

def get_lens_model(lens_model_name):
    if lens_model_name == 'NFW':
        lm = NFWLens()
    elif lens_model_name == 'gSIS':
        lm = gSISLens()
    elif lens_model_name == 'CIS':
        lm = CISLens()
    else:
        raise ValueError(f'lens model {lens_model_name} not supported')
    return lm

class LensModel:
    def __init__(self):
        pass

    def get_origin_type(self):
        return origin_type_default
    
    def get_y_crit_override(self):
        return y_crit_override_default
    
    def get_x_im_nan_sub(self, dT_1D):
        return x_im_nan_sub_default
    
    def get_irregular_crit_points_dict(self):
        return None
    
    def get_add_to_strong(self):
        return None
    
    def get_override_funcs_dict(self, dT_1D):
        return {'y_crit_override': self.get_y_crit_override(),
                'x_im_nan_sub': self.get_x_im_nan_sub(dT_1D),
                'origin_type': self.get_origin_type(),
                'irregular_crit_points_dict': self.get_irregular_crit_points_dict(),
                'add_to_strong': self.get_add_to_strong()}


class NFWLens(LensModel):

    def __init__(self):
        pass

    def get_Psi(self):
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
        return Psi_NFW
    
class gSISLens(LensModel):
    
    def __init__(self):
        pass

    def get_Psi(self):
        @jit
        def Psi_gSIS(x, lens_params):
            k = lens_params[0]
            return jnp.linalg.norm(x)**(2 - k)/(2 - k)
        return Psi_gSIS
    
    def get_y_crit_override(self):
        @partial(jnp.vectorize, signature = '(),()->()')
        def y_crit_override_gSIS(y_crit, lens_params):
            lens_params = jnp.atleast_1d(lens_params)
            return jax.lax.cond(jnp.abs(lens_params[0] - 1.) < 1e-15, 
                                lambda _: jnp.ones_like(y_crit), 
                                lambda _: jax.lax.cond(lens_params[0] > 1.,
                                                    lambda _: jnp.ones_like(y_crit)*jnp.inf, 
                                                    lambda _: y_crit, 
                                                    None),
                                None)
        return y_crit_override_gSIS
    
    def get_origin_type(self):
        def origin_type_gSIS(lens_params):
            if lens_params[0] < 1.:
                return 'regular'
            else:
                return 'cusp'
        return origin_type_gSIS
    
    def get_x_im_nan_sub(self, dT_1D):
        bisection_1D_var_2D = make_bisection_1D_var_2D()
        bisection_1D_cond_fun = make_bisection_1D_cond_fun(1e-13)
        bisection_1D_step_fun = make_bisection_1D_step_fun(dT_1D)
        @partial(jnp.vectorize, signature = '(3),(),()->(3)')
        def x_im_nan_sub_gSIS(x_im, y0, lens_param):
            lens_params = jnp.atleast_1d(lens_param)
            x_im = jax.lax.cond(jnp.isnan(x_im[0]) & (lens_params[0] > 1.), 
                                lambda x_im: x_im.at[0].set(
                                    bisection_1D_var_2D(dT_1D, 0., 
                                                        -0.1, -1e-14, 
                                                        bisection_1D_cond_fun, 
                                                        bisection_1D_step_fun, 
                                                        y0, lens_params[0])
                                ), 
                                lambda x_im: x_im, 
                                operand = x_im)
            x_im = jax.lax.cond(jnp.isnan(x_im[0]) & (lens_params[0] <= 1.) & (y0 < 1.),
                                lambda x_im: x_im.at[0].set(
                                    bisection_1D_var_2D(dT_1D, 0., 
                                                        (y0-1)*1.5, (y0-1)*0.5, 
                                                        bisection_1D_cond_fun, 
                                                        bisection_1D_step_fun, 
                                                        y0, lens_params[0])
                                ), 
                                lambda x_im: x_im, 
                                operand = x_im)
            x_im = jax.lax.cond(jnp.isnan(x_im[1]) & (lens_params[0] < 1.), 
                                lambda x_im: x_im.at[1].set(
                                    bisection_1D_var_2D(dT_1D, 0., 
                                                        (y0-1)*0.5, -1e-14, 
                                                        bisection_1D_cond_fun, 
                                                        bisection_1D_step_fun, 
                                                        y0, lens_params[0])
                                ), 
                                lambda x_im: x_im, 
                                operand = x_im)
            return x_im

        return x_im_nan_sub_gSIS
    
    def get_irregular_crit_points_dict(self):
        irregular_crit_points_dict = {
            'irregular_crit_points': jnp.array([[1., 1.]]),
            'irregular_y_crit_points': jnp.array([1.]),
            'irregular_x_crit_points': jnp.array([0.]),
            'irregular_lp_crit_points': jnp.array([1.])
        }
        return irregular_crit_points_dict
    
    def get_add_to_strong(self):
        @partial(jnp.vectorize, signature = '(n,2)->(n)')
        def is_crit(point):
            return jnp.abs(point[:,1] - 1.) < 1e-15
        return is_crit
        
class CISLens(LensModel):

    def __init__(self):
        pass

    def get_Psi(self):
        @jit
        def Psi_CIS(x, lens_params):
            x_c = jnp.abs(lens_params[0])
            x_t = jnp.sqrt(x_c**2 + jnp.linalg.norm(x)**2)
            x_c_safe = jnp.where(x_c > 1e-15, x_c, 1e-15)
            Psi = jnp.where(x_c > 1e-15, 
                    x_t + x_c_safe * jnp.log(2 * x_c_safe / (x_t + x_c_safe)), 
                    x_t
                            )
            return Psi
        return Psi_CIS
    
    def get_y_crit_override(self):
        @partial(jnp.vectorize, signature = '(),()->()')
        def y_crit_override_CIS(y_crit, lens_params):
            lens_params = jnp.atleast_1d(lens_params)
            y_crit = jax.lax.cond(lens_params[0] < 1e-15,
                                lambda y_crit: 1.,
                                lambda y_crit: jax.lax.cond(y_crit < 0.,
                                                    lambda y_crit: jnp.ones_like(y_crit)*0., 
                                                    lambda y_crit: y_crit, 
                                                    y_crit),
                                y_crit)
            return y_crit
        return y_crit_override_CIS
    
    def get_origin_type(self):
        def origin_type_CIS(lens_params):
            if jnp.abs(lens_params[0]) > 1e-15:
                origin = 'regular'
            else:
                origin = 'cusp'
            return origin
        return origin_type_CIS
    
    # def get_x_im_nan_sub(self, dT_1D):
    #     bisection_1D_var_2D = make_bisection_1D_var_2D()
    #     bisection_1D_cond_fun = make_bisection_1D_cond_fun(1e-13)
    #     bisection_1D_step_fun = make_bisection_1D_step_fun(dT_1D)
    #     def x_im_nan_sub_CIS(x_im, y0, lens_params):
    #         lens_params = jnp.atleast_1d(lens_params)
    #         x_im = jax.lax.cond(jnp.isnan(x_im[1]) & (lens_params[0] > 0) & (y0 < 1.),
    #                             lambda x_im: x_im.at[1].set(
    #                                 bisection_1D_var_2D(dT_1D, 0., 
    #                                                     -0.1, -1e-14, 
    #                                                     bisection_1D_cond_fun, 
    #                                                     bisection_1D_step_fun, 
    #                                                     y0, lens_params[0])
    #                             ), 
    #                             lambda x_im: x_im, 
    #                             operand = x_im)
    #         return x_im
    #     return x_im_nan_sub_CIS
    
    def get_irregular_crit_points_dict(self):   
        irregular_crit_points_dict = {
                    'irregular_crit_points': jnp.array([[1., 0.]]),
                    'irregular_y_crit_points': jnp.array([1.]),
                    'irregular_x_crit_points': jnp.array([0.]),
                    'irregular_lp_crit_points': jnp.array([0.])
                }
        return irregular_crit_points_dict