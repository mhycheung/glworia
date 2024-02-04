# Constructing an interpolation table

To construct an interpolation table, use the `do_interpolation.py` python script with a configuration file.

```
python scripts/do_interpolation.py -f /path/to/config_file.json -s /path/to/output/directory -im
```

| Option| flag | Description |
| ----------- | -----------  | ----------- |
| `--input-file` | `-f` | Path to configuration file. |
| `--save-dir` | `-s` | Path to directory for saving the interpolation tables. |
| `--image` | `-im` | Also construct an interpolation table for the image magnifications and time delays, in addition to the amplification factor. |
| `--skip-strong` | `-ss` | Skip the strong lensing region in the parameter space. |
| `--skip-weak` | `-sw` | Skip the weak lensing region in the parameter space. |
| `--skip-amplification` | `-sa` | Skip the computation of the amplification factor. |

The configuration file should be in `json` format with the following entries:

| Parameter | Type | Description |
| --------- | ------ | ----------- |
| `lens_param_name` | `str` | Name given to the lens parameter $l$, e.g. `kappa` for the NFW lens or `x_c` for the CIS lens. |
| `y_low` | `float` | Lower limit of the impact parameter $y$ in the interpolation domain. |
| `y_high` | `float` | Upper limit of the impact parameter $y$ in the interpolation domain. |
| `lp_low` | `float` | Lower limit of the lens parameter $l$ in the interpolation domain. |
| `lp_high` | `float` | Upper limit of the lens parameter $l$ in the interpolation domain. |
| `N_grid` | `int` | Number of interpolation nodes to use in each dimension on the $y$-$l$ plane to use for the weak lensing region. |
| `N_grid_strong` | `int` | Number of interpolation nodes to use in each dimension on the $y$-$l$ plane to use for the strong lensing region. |
| `N_crit` | `int` | Number of interpolation nodes on the caustic curve. |
| `N` | `int` | Number of interpolation nodes in dimensionless time $\tau$ to use in each section of the time domain amplification factor. |
| `lens_model_name` | `str` | The name of the lens model, e.g. `NFW`. |
| `crit_lp_N` | `int` | Number of points in $l$ to use for a high-resolution interpolator tracing out the caustic curve in the $y$-$l$ plane. |
| `im_x_init_low` | `float` | Lower limit of the `im_x_init` array of initial guesses for Newton's method when solving for image positions. |
| `im_x_init_high` | `float` | Upper limit of the `im_x_init` array of initial guesses for Newton's method when solving for image positions. |
| `im_x_init_num` | `int` | Number of points in the `im_x_init` array of initial guesses for Newton's method when solving for image positions. |
| `lp_low_im` | `float` | Lower limit of the lens parameter $l$ in the interpolation domain for image related quantities. |
| `lp_high_im` | `float` | Upper limit of the lens parameter $l$ in the interpolation domain for image related quantities. |
| `N_grid_im` | `int` | Number of interpolation nodes to use in each dimension on the $y$-$l$ plane to use for interpolating image related quantities. |
| `N_crit_im` | `int` | Number of interpolation nodes to use on the caustic curve for interpolating image related quantities. |
| `newt_max_iter_im` | `int` | Maximum iterations for Newton's method when solving for the image positions. |
| `crit_lp_im` | `int` | Number of points in $l$ to use for a high-resolution interpolator tracing out the caustic curve in the $y$-$l$ plane, for calculations related to the images. |