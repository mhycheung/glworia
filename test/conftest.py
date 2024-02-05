import jax.numpy as jnp
import json
import os

import pytest

@pytest.fixture
def test_F():
    F = jnp.load('test/data/F_interp.npy')
    return F

@pytest.fixture()
def test_F_settings():
    F_settings = {}

    lm_name = 'NFW'
    ws = jnp.linspace(1e-2, 1e3, 10**5)
    y = jnp.array([0.1, 0.])
    lens_params = jnp.array([3.])
    N = 200
    T0_max = 1000.
    param_arr = jnp.linspace(0.1, 10., 100000)

    F_settings['lm_name'] = lm_name
    F_settings['ws'] = ws
    F_settings['y'] = y
    F_settings['lens_params'] = lens_params
    F_settings['N'] = N
    F_settings['T0_max'] = T0_max
    F_settings['param_arr'] = param_arr

    return F_settings

@pytest.fixture
def test_interp_settings():
    with open('test/data/NFW_test.json', 'r') as f:
        interp_settings = json.load(f)
    return interp_settings

@pytest.fixture
def test_interp_points():
    ys = jnp.linspace(0.1, 2.0, 4)
    ls = jnp.linspace(4.0, 8.0, 4)
    return ys, ls

def test_make_out_dir():
    os.makedirs('test/out/interp', exist_ok=True)