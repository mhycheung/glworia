import jax.numpy as jnp

from glworia.amp.lens_model import get_lens_model
from glworia.amp.amplification_factor import (amplification_computation_prep, 
                                          crtical_curve_interpolants,
                                          compute_F)
from glworia.amp.root import make_crit_curve_helper_func

def test_compute_F(test_F, test_F_settings):
    
    lm_name = test_F_settings['lm_name']
    ws = test_F_settings['ws']
    y = test_F_settings['y']
    lens_params = test_F_settings['lens_params']
    N = test_F_settings['N']
    T0_max = test_F_settings['T0_max']
    param_arr = test_F_settings['param_arr']

    lm = get_lens_model(lm_name)
    Psi = lm.get_Psi()
    T_funcs, helper_funcs = amplification_computation_prep(Psi)
    crit_curve_helper_funcs = make_crit_curve_helper_func(T_funcs)
    crit_funcs = crtical_curve_interpolants(param_arr, T_funcs, crit_curve_helper_funcs)

    F_interp, _, _, _ = compute_F(ws, y, lens_params, T_funcs, helper_funcs, crit_funcs,
              N, T0_max)

    assert jnp.allclose(F_interp, test_F, atol=1e-5)