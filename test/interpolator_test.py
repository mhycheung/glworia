from jax import config
import numpy as np
config.update("jax_enable_x64", True)

from glworia.amp.interpolate import interpolate, interpolate_im
from glworia.amp.load_interp import load_interpolators, F_interp
import json

def test_interpolate(test_interp_settings):
    out_dir = 'test/out/interp'
    interpolate_im(test_interp_settings, out_dir)
    interpolate(test_interp_settings, out_dir)

def test_load_interpolate(test_interp_settings, test_interp_points):
    interpolation_dir_path_test = 'test/out/interp'
    interpolation_dir_path_saved = 'test/data/interp'
    interpolators_test = load_interpolators(interpolation_dir_path_test, **test_interp_settings)
    interpolators_saved = load_interpolators(interpolation_dir_path_saved, **test_interp_settings)
    w_interp = np.linspace(0.001, 1e4, 10**5)
    ys, ls = test_interp_points
    for y in ys:
        for l in ls:
            F_test = F_interp(w_interp, y, l, interpolators_test, test_interp_settings)
            F_saved = F_interp(w_interp, y, l, interpolators_saved, test_interp_settings)
            assert np.allclose(F_test, F_saved, atol=1e-5)
