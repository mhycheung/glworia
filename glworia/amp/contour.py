import jax
import jax.numpy as jnp
from jax import vmap, jit
from functools import partial
from .root import *
from .utils import *

from typing import List, Tuple, Union, Optional, Dict, Any, Callable

@partial(jit, static_argnums=(0,))
def rk4_step(f, x, h, y, lens_params):
    k1 = f(x, y, lens_params)
    k2 = f(x + h/2*k1, y, lens_params)
    k3 = f(x + h/2*k2, y, lens_params)
    k4 = f(x + h*k3, y, lens_params)
    return h/6*(k1 + 2*k2 + 2*k3 + k4)

@partial(jit, static_argnums=(1, 2, 3, 4, 5))
def contour_int_step_func_full(x_iter, T, dT, dT_norm, f, T_hess_det, h):
    x_old = x_iter[0]
    r = jnp.linalg.norm(x_old)
    u_acc_old = x_iter[2]
    T0 = x_iter[4]
    iter = x_iter[5] + 1
    y = x_iter[7]
    lens_params = x_iter[8]
    kappa = jnp.min(jnp.array([
        jnp.max(jnp.array([100., 1/r])), 
        jnp.max(
            jnp.array([jnp.abs(T_hess_det(x_old, y, lens_params)/dT_norm(x_old, y, lens_params)), 
                       jnp.max(jnp.array([0.1, 1/r]))
                       ]))]
                     ))
    dx_prop = rk4_step(f, x_old, h/kappa, y, lens_params)
    x_prop = x_old + dx_prop
    T_prop = T(x_prop, y, lens_params)
    dx_new = dT(x_prop, y, lens_params)/dT_norm(x_prop, y, lens_params)**2*(T0 - T_prop)
    #FIXME: dx_new could overshoot if near a cusp
    x_new = x_prop + dx_new
    l = jnp.linalg.norm(x_new - x_old)
    dT_norm_mid = dT_norm((x_new + x_old)/2, y, lens_params)
    u_acc_new = u_acc_old + l/dT_norm_mid
    return [x_new, x_old, u_acc_new, u_acc_old, T0, iter, h, y, lens_params]

def make_contour_int_step_func(T, dT, dT_norm, f, T_hess_det, h):
    return lambda x_iter: contour_int_step_func_full(x_iter, T, dT, dT_norm, f, T_hess_det, h)

@jit
def contour_int_cond_func_full(x_iter):
    #FIXME: hardcoded max iter
    return (x_iter[0][1]*x_iter[1][1] >= 0) & (x_iter[5] < 5000)

@partial(jnp.vectorize, signature='(),(2)->(),()', excluded=(2, 3, 4, 5))
@partial(jit, static_argnums=(2, 3))
def contour_int(T0, x_init, cond_func, step_func, y, lens_params):
    u_acc = 0.
    init_val = [x_init, x_init, u_acc, u_acc, T0, 0, 0, y, lens_params]
    x_iter = jax.lax.while_loop(cond_func, step_func, init_val)
    # h = x_iter[6]
    u_final_raw = x_iter[2] - x_iter[3]
    u_final = jnp.abs(x_iter[1][1]/(x_iter[1][1] - x_iter[0][1]))*u_final_raw
    # x_iter_final = [x_iter[1], x_iter[1], x_iter[3], x_iter[3], T0, x_iter[5], h_final, y, lens_params]
    # x_iter = contour_int_step_func_full(x_iter_final, T, dT, dT_norm, f, T_hess_det, h_final)
    return 2*(x_iter[3] + u_final)/(2*jnp.pi), x_iter[5]    

class critical_point:

    def __init__(self, x, type, T_func, mu, y, lens_params, is_cusp = False):
        self.x = x
        self.x0 = x[0]
        self.x1 = x[1]
        self.y = y
        self.y0 = y[0]
        self.y1 = y[1]
        if self.x1 != 0.:
            raise ValueError('x1 != 0 not yet implemented')
        if self.y1 != 0.:
            raise ValueError('y1 != 0 not yet implemented')
        if type not in ['min', 'max', 'sad']:
            raise ValueError('type must be one of "min", "max", "sad"')
        self.type = type
        self.T = T_func(x, y, lens_params)
        self.is_cusp = is_cusp
        self.mu = mu(self.x0, lens_params)

class contour_integral:
    """
    A class for computing the amplification factor by contour integration.

    Attributes:
        y: 2D impact parameter. The second component is ignored for now.
        y0: 1D impact parameter.
        lens_params: The lens parameters. Only one parameter is supported for now.
        T0_min_out_segments: A list of the arrays of time delay values for the contours passing to the right of the minimum image.
        T0_sad_max_segment: The time delay values for the contours passing between the saddle and the maximum image.
        x_im: The image positions in 1D. The order of the images are from the left to the right, i.e. saddle, maximum, minimum.
        mus: The magnification of the images.
        u_min_out: A list of arrays of the time domain amplification for the contours passing to the right of the minimum image, corresponding to the time delays `T0_min_out_segments`.
        u_sad_max: The time domain amplification for the contours passing between the saddle and the maximum image, corresponding to the time delays `T0_sad_max_segment`.
        T_vir: The time delay at the maximum value of the time domain amplification.
    """

    y: jnp.ndarray
    y0: float
    lens_params: jnp.ndarray
    T0_min_out_segments: List[jnp.ndarray]
    T0_sad_max_segment: jnp.ndarray
    x_im: jnp.ndarray
    mus: jnp.ndarray
    u_min_out: List[jnp.ndarray]
    u_sad_max: jnp.ndarray
    T_vir: float

    def __init__(self, x_im: jnp.ndarray, multi_image: bool,
                 T_funcs: Dict[str, Callable], y: jnp.ndarray, lens_params: jnp.ndarray,
                 critical: bool = False, 
                 T0_min_out_segments: Optional[List[jnp.ndarray]] = None,
                 T0_sad_max_segment: Optional[jnp.ndarray] = None,
                 singular: bool = False):
        """
        Parameters:
            x_im: The image positions in 1D. The order of the images are from the left to the right, i.e. saddle, maximum, minimum.
            multi_image: A boolean indicating whether there are multiple images.
            T_funcs: A dictionary of functions derived from the Fermat Potential.
            y: 2D impact parameter. The second component is ignored for now.
            lens_params: The lens parameters. Only one parameter is supported for now.
            critical: A boolean indicating whether the supplied impact parameter lies on the caustic.
            T0_min_out_segments: A list of the arrays of time delay values for the contours passing to the right of the minimum image.
            T0_sad_max_segment: The time delay values for the contours passing between the saddle and the maximum image.
            singular: A boolean indicating whether the lens is singular at the origion.
        """
        
        self.T0_min_out_segments = T0_min_out_segments
        self.multi_image = multi_image
        self.critical = critical
        self.T0_sad_max_segment = T0_sad_max_segment

        self.y = y
        self.y0 = y[0]
        self.lens_params = lens_params
        self.args = [y, lens_params]
        self.args_1D = [self.y0, self.lens_params]

        self.T_1D = T_funcs['T_1D']
        self.mu = T_funcs['mu']
        self.T_images = self.T_1D(x_im, self.y0, self.lens_params)
        self.x_im = x_im
        self.x_im_min = self.x_im[2]
        if singular:
            self.x_im_max = 0.
            self.x_im[1] = 0.
        else:
            self.x_im_max = self.x_im[1]
        self.x_im_sad = self.x_im[0]
        self.T_im_min = self.T_images[2]
        self.T_im_max = self.T_images[1]
        self.T_im_sad = self.T_images[0]

    def compute_mus(self):
        self.mus = self.mu(self.x_im, self.lens_params)
        self.mu_sad, self.mu_max, self.mu_min = tuple(self.mus)

    def set_T0_segments(self, T0_min_out_segments, T0_sad_max_segment = jnp.array([])):
        self.T0_min_out_segments = T0_min_out_segments
        self.T0_sad_max_segment = T0_sad_max_segment

    def get_T_max(self):
        return jnp.max(self.T0_min_out_segments[-1])
    
    def set_T_max(self, T_max):
        self.T_max = T_max

    def get_and_set_T_max(self):
        self.T_max = self.get_T_max()

    def find_T_outer(self, bisect_cond_fun, bisect_step_fun_T_1D):
        self.x_outer =  find_outer_from_T_max(
            self.T_max, self.x_im_min, self.T_im_sad, 
            self.T_im_min, self.T_1D, self.args_1D, bisect_cond_fun,
            bisect_step_fun_T_1D, 10.)
        
    def make_x_inits(self, bisection_1D_v, bisect_cond_fun, bisect_step_fun_T_1D):
        self.x_inits_min_out = make_x_inits_seg(self.T0_min_out_segments+self.T_im_min, 
                         self.x_im_min, self.x_outer, self.args_1D, bisection_1D_v, bisect_cond_fun,
                     bisect_step_fun_T_1D, self.T_1D)
        self.x_inits_sad_max = make_x_inits_seg(self.T0_sad_max_segment+self.T_im_min, 
                         self.x_im_sad, self.x_im_max, self.args_1D, bisection_1D_v, bisect_cond_fun,
                     bisect_step_fun_T_1D, self.T_1D)

    def contour_integrate(self, contour_cond_fun: Callable, contour_step_fun: Callable):
        """
        Computes the contour integral

        Parameters:
            contour_cond_fun: The condition function for terminating the contour integral.
            contour_step_fun: The step function for the contour integral.
        """
        self.u_min_out, self.iter_min_out = contour_int(
            self.T0_min_out_segments + self.T_im_min, self.x_inits_min_out, 
            contour_cond_fun, contour_step_fun, self.y, self.lens_params)
        if self.critical:
            # mu_max = self.mu(self.x_im_max, self.lens_params)
            self.u_sad_max = jnp.zeros_like(self.T0_sad_max_segment)
            self.iter_sad_max = jnp.zeros(len(self.T0_sad_max_segment), dtype = int)
        else:
            self.u_sad_max, self.iter_sad_max = contour_int(
                self.T0_sad_max_segment + self.T_im_min, self.x_inits_sad_max, 
                contour_cond_fun, contour_step_fun, self.y, self.lens_params)
        
    def find_T_for_max_u(self, T0_arr, bisection_1D_v, bisect_cond_fun, bisect_step_fun_T_1D, 
                   contour_cond_fun, contour_step_fun, iters):
        for i in range(iters):
            x_inits = make_x_inits_seg(T0_arr+self.T_im_min, 
                                    self.x_im_min, self.x_outer, self.args_1D, bisection_1D_v, bisect_cond_fun,
                                bisect_step_fun_T_1D, self.T_1D)
            u, iter = contour_int(T0_arr + self.T_im_min, x_inits,
                                    contour_cond_fun, contour_step_fun, self.y, self.lens_params)
            max_u_indx = jnp.argmax(u)
            max_indx = max_u_indx#jnp.max(jnp.array([max_u_indx, max_du2_indx]))
            self.T_vir = T0_arr[max_indx]
            T0_arr = jnp.linspace(T0_arr[int(jnp.max(jnp.array([0, max_indx-1])))], T0_arr[max_indx+1], len(T0_arr))
        return self.T_vir


def find_outer(T0_arr, min_x0, T_sad, T_min, T_1D, args_1D, bisect_cond_fun, bisect_step_fun_T_1D, margin_fac = 10.):
        T0_max = jnp.nanmax(jnp.array([jnp.max(T0_arr - T_min), (T_sad - T_min)*1.5]))
        x_max_over = jnp.sqrt(2 * T0_max)
        x_outer = bisection_1D(T_1D, T0_max, min_x0,
                                x_max_over, bisect_cond_fun, bisect_step_fun_T_1D, args_1D)
        x_outer = x_outer*margin_fac
        return x_outer

@partial(jnp.vectorize, excluded = [1, 2, 3, 4, 5, 6, 7, 8])
def find_outer_from_T_max(T_max, min_x0, T_sad, T_min, T_1D, args_1D, bisect_cond_fun, bisect_step_fun_T_1D, margin_fac = 10.):
        T0_max = jnp.nanmax(jnp.array([T_max, (T_sad - T_min)*1.5]))
        x_max_over = jnp.sqrt(2 * T0_max)
        x_outer = bisection_1D(T_1D, T0_max, min_x0,
                                x_max_over, bisect_cond_fun, bisect_step_fun_T_1D, args_1D)
        x_outer = x_outer*margin_fac
        return x_outer

def x_init_sad_max_routine_full(T0_sad_max, sad_x0, max_x0, T_1D, args_1D, bisection_1D_v, bisect_cond_fun, bisect_step_fun_T_1D):
    # T0_sad_indx = jnp.searchsorted(T0_arr, sad_T)
    # T0_max_indx = jnp.searchsorted(T0_arr, max_T)
    # T0_sad_max = T0_arr[T0_sad_indx:T0_max_indx]

    x0_init_sad_max = bisection_1D_v(T_1D, T0_sad_max, sad_x0,
                                                max_x0,
                                bisect_cond_fun, bisect_step_fun_T_1D, args_1D)
    return x0_init_sad_max


def make_x_init_sad_max_routine(T_1D, bisection_1D_v, bisect_cond_fun, bisect_step_fun_T_1D):
    return lambda T0_sad_max, sad_x0, max_x0, args_1D: x_init_sad_max_routine_full(T0_sad_max, sad_x0, max_x0, T_1D, args_1D, bisection_1D_v, bisect_cond_fun, bisect_step_fun_T_1D)

def x_init_sad_max_return_empty(T0_sad_max, sad_x0, max_x0, args_1D):
    return jnp.nan*jnp.ones(len(T0_sad_max))

@partial(jnp.vectorize, excluded = [1, 2, 3, 4, 5, 6, 7], signature = '(n)->(n,2)')
@partial(jit, static_argnums=(4, 5, 6, 7))
def make_x_inits_seg(T0_arr, from_x0, to_x0, args_1D, bisection_1D_v, bisect_cond_fun,
                     bisect_step_fun_T_1D, T_1D):
    x0_init = bisection_1D_v(T_1D, T0_arr, from_x0, to_x0,
                             bisect_cond_fun, bisect_step_fun_T_1D, args_1D)
    x_init = turn_1D_to_2D(x0_init, 0.)
    return x_init

@partial(jit, static_argnums=(6, 9, 10, 11, 12))
def make_x_inits(T0_arr, min_x0, T0_sad_max, sad_x0, max_x0, x_outer, T_1D, args_1D, multiple_image, bisection_1D_v, bisect_cond_fun, bisect_step_fun_T_1D, x_init_sad_max_routine):
    x0_init_min_out = bisection_1D_v(T_1D, T0_arr, min_x0, x_outer,
                                    bisect_cond_fun, bisect_step_fun_T_1D, args_1D)
    x_init_min_out = turn_1D_to_2D(x0_init_min_out, 0.)

    ops = (T0_sad_max, sad_x0, max_x0, args_1D)
    x0_init_sad_max = jax.lax.cond(multiple_image, x_init_sad_max_routine, x_init_sad_max_return_empty, *ops)
    x0_init_sad_max = turn_1D_to_2D(x0_init_sad_max, 0.)

    return x_init_min_out, x0_init_sad_max

    # if self.multiple_image:
    #     self.T0_sad_indx = jnp.searchsorted(self.T0_arr, self.sad.T)
    #     self.T0_max_indx = jnp.searchsorted(self.T0_arr, self.max.T)
    #     self.T0_sad_max = self.T0_arr[self.T0_sad_indx:self.T0_max_indx]

    #     x0_init_sad_max = bisection_1D_v(self.T_1D, self.T0_sad_max, self.sad.x0,
    #                                                 self.max.x0,
    #                                 bisect_cond_fun, bisect_step_fun_T_1D, self.args_1D)
    #     self.x_init_sad_max = turn_1D_to_2D(x0_init_sad_max, 0.)

def compute_contour_ints_min_out(T0_arr, x_init_min_out, y, lens_params, contour_cond_func, contour_step_func):
    # contour_int_v = make_contour_int_v(self.T, self.dT, 
    #                                    self.dT_norm, self.f,
    #                                    self.T_hess_det, h)
    u_min_out, iter_min_out = contour_int(
        T0_arr,  
        x_init_min_out, 
        contour_cond_func, contour_step_func, y, lens_params)
    return u_min_out, iter_min_out
    
def compute_contour_ints_sad_max_full(T0_arr, T0_sad_max, x_init_sad_max, y, lens_params, contour_cond_func, contour_step_func):
    u_sad_max, iter_sad_max = contour_int(
        T0_sad_max,  
        x_init_sad_max, 
        contour_cond_func, contour_step_func, y, lens_params)
    return u_sad_max, iter_sad_max

def make_compute_contour_ints_sad_max(contour_cond_func, contour_step_func):
    return lambda T0_arr, T0_sad_max, x_init_sad_max, y, lens_params: compute_contour_ints_sad_max_full(T0_arr, T0_sad_max, x_init_sad_max, y, lens_params, contour_cond_func, contour_step_func)

def contour_ints_sad_max_return_0(T0_arr, T0_sad_max, x_init_sad_max, y, lens_params):
    return jnp.zeros(T0_sad_max.shape, dtype=jnp.float64), jnp.zeros(T0_sad_max.shape, dtype = jnp.int64)

@partial(jit, static_argnums=(6, 7, 8))
def compute_contour_ints(T0_arr, T0_sad_max, x_init_min_out, x_init_sad_max, y, lens_params, contour_cond_func, contour_step_func, compute_contour_ints_sad_max):  
    u_min_out, iter_min_out = compute_contour_ints_min_out(T0_arr, x_init_min_out, y, lens_params, contour_cond_func, contour_step_func)
    opts = (T0_arr, T0_sad_max, x_init_sad_max, y, lens_params)
    u_sad_max, iter_sad_max = jax.lax.cond(~jnp.isnan(x_init_sad_max[0,0]), compute_contour_ints_sad_max, contour_ints_sad_max_return_0, *opts)
    return u_min_out, iter_min_out, u_sad_max, iter_sad_max
   
def add_sad_max(u_sum, T0_sad_indx, T0_max_indx, u_sad_max):
    u_sum = u_sum.at[T0_sad_indx:T0_max_indx].add(u_sad_max)
    return u_sum

def pass_sad_max(u_sum, T0_sad_indx, T0_max_indx, u_sad_max):
    return u_sum

def sum_results(T0_arr, u_min_out, u_sad_max, T0_sad_indx, T0_max_indx, multiple_image):
    u_sum = jnp.zeros(len(T0_arr))
    u_sum += u_min_out

    opts = (u_sum, T0_sad_indx, T0_max_indx, u_sad_max)
    u_sum = jax.lax.cond(multiple_image, add_sad_max, pass_sad_max, *opts)

    return u_sum

def make_adaptive_T0_arr(T_images, T_max, long_num, short_num, dense_width = 0.01):
    T0_arr_dense_1 = jnp.linspace(T_images[0]-dense_width/2, T_images[0]+dense_width/2, short_num)+1e-7
    T0_arr_dense_2 = jnp.linspace(T_images[1]-dense_width/2, T_images[1]+dense_width/2, short_num)+1e-7
    T0_arr_dense_1 = T0_arr_dense_1[~jnp.isnan(T0_arr_dense_1)]
    T0_arr_dense_2 = T0_arr_dense_2[~jnp.isnan(T0_arr_dense_2)]
    T0_arr_sparse = jnp.linspace(0, T_max, long_num + 2*short_num - len(T0_arr_dense_1) - len(T0_arr_dense_2)) + 1e-5
    T0_arr = jnp.concatenate((T0_arr_sparse, T0_arr_dense_1, T0_arr_dense_2))
    return jnp.sort(T0_arr)