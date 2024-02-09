import jax.numpy as jnp
from typing import List, Tuple, Union, Optional, Dict, Any, Callable

def interp_F_fft_strong_jnp(t_fft, T0_min_out_interp_full, u_min_out_interp_full, 
                 T0_sad_max_interp, u_interp_sad_max):
    F_fft = jnp.interp(t_fft, T0_min_out_interp_full, u_min_out_interp_full)
    F_fft += jnp.interp(t_fft, T0_sad_max_interp, u_interp_sad_max, left = 0., right = 0.)
    return F_fft

def interp_F_fft_weak_jnp(t_fft, T0_min_out_interp_full, u_min_out_interp_full):
    F_fft = jnp.interp(t_fft, T0_min_out_interp_full, u_min_out_interp_full)
    return F_fft

def amplification_fft_jnp(t_fft, Ft_fft):

    fft_len = len(t_fft)
    dt = t_fft[1] - t_fft[0]
    w_arr = jnp.linspace(0, 2*jnp.pi/dt, num = fft_len)
    Fw_raw = w_arr*jnp.fft.fft(Ft_fft)*dt
    Fw = -jnp.imag(Fw_raw) - 1.j*jnp.real(Fw_raw) + Ft_fft[-1]

    return w_arr, Fw

def F_geom_jnp(ws, T_im, mu_im):
    F = jnp.zeros(len(ws), dtype = jnp.complex128)
    morse_indx = [0.5, 1, 0]
    mu_im = jnp.nan_to_num(mu_im)
    for i in range(3):
        F += jnp.sqrt(jnp.abs(mu_im[i]))*jnp.exp(1.j*(ws*T_im[i] - jnp.pi*morse_indx[i]))
    return F

def smooth_increase_jnp(x, x0, a):
    return (1 + jnp.tanh((x - x0)/a))/2

def smooth_decrease_jnp(x, x0, a):
    return 1 - smooth_increase_jnp(x, x0, a)

def interp_partitions_jnp(w_interp: jnp.ndarray, ws: List[jnp.ndarray], Fs: List[jnp.ndarray], partitions: jnp.ndarray, sigs: jnp.ndarray, T_im: jnp.ndarray, mu_im: jnp.ndarray, origin: str = 'regular') -> Tuple[jnp.ndarray, List[jnp.ndarray]]:
    """
    Patches together the interpolated frequency domain amplitude from different frequency regimes.

    Parameters:
        w_interp: The output frequency array that the interpolated amplitude will be evaluated at.
        ws: A list of frequency arrays that the amplitude is interpolated from.
        Fs: A list of amplitude arrays that the amplitude is interpolated from.
        partitions: The transition frequencies between different frequency regimes. Should have a length one less than `ws` and `Fs`.
        sigs: The width of the transition regions.
        T_im: The time delay of the images.
        mu_im: The magnification of the images.
        origin: The type of the origin. Can be `regular`, `cusp` or `im`.
    
    Returns:
        F_interp: The interpolated amplitude.
        F_interp_raw: A list of the interpolated amplitudes from different frequency regimes.
    """
    if origin not in ['regular', 'im']:
        mu_im.at[1].set(0.)
        T_im.at[1].set(0.)
    F_interp = jnp.zeros_like(w_interp, dtype = jnp.complex128)
    F_interp_raw = [jnp.ones_like(w_interp, dtype=jnp.complex128)]
    for i, (w, F) in enumerate(zip(ws, Fs)):
        F_interp_raw.append(jnp.interp(w_interp, w, F, left = 1., right = 0.))
    F_interp_raw.append(F_geom_jnp(w_interp, T_im, mu_im))
    for i in range(len(partitions)):
        if i == 0:
            F_interp += F_interp_raw[i]*smooth_decrease_jnp(w_interp, partitions[i], sigs[i])
        if i < len(partitions) - 1:
            F_interp += F_interp_raw[i+1]*smooth_increase_jnp(w_interp, partitions[i], sigs[i])\
                        *smooth_decrease_jnp(w_interp, partitions[i+1], sigs[i+1])
        else:
            F_interp += F_interp_raw[i+1]*smooth_increase_jnp(w_interp, partitions[i], sigs[i])
    return F_interp, F_interp_raw