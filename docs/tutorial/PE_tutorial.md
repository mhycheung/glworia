# Parameter Estimation with Bilby  

Given an interpolation table of the lensing amplification factor and an unlensed waveform model, the lensed waveform can be rapidly called.

While any value on the $y$-$l$ plane (where $y$ is the impact parameter and $l$ the lens parameter) can be interpolated in principle as long as the value lies within the boundaries of the interpolation table, the interpolated amplification is ill-behaved close to the caustic because the magnification diverges formally.
Therefore, when sampling over the $y$-$l$ plane for parameter estimation, we will ignore the region close to the caustic by assigning it a zero prior probability as implemented in the `glworia.pe.prior` module.
We define such a region as the region between $y_c(l) \pm \delta$, where $y_c(l)$ is the caustic curve in the $y$-$l$ plane, and $\delta$ is a constant. 
For lens models that will reduce to the SIS lens at certain limits (e.g. $x_c \to 0$ for the CIS lens, or $k \to 1$ for gSIS), we also exclude the region close to the $y = 1$ caustic at those limits by setting the prior probability within a small rectangle to be zero.

Example parameter estimation scripts using `bilby` can be found in the `bilby_scripts/` directory in the Git repo.
The scripts can be run with a configuration file.

```
python PE_full.py /path/to/config_file.json
```

Example configuration files can be found in the `config/` directory.
The dictionaries required in the `json` configuration file are as follows:

- `injection_parameters`: the injected parameters for the lensed waveform. These include all the parameters of the unlensed waveform (passed to the waveform generator in `bilby`), in addition to the following lens parameters:

    | Name | Type | Description |
    | ----------- | ----------- | ----------- |
    | `y` | `float` | Injected impact parameter for the lensing set-up |
    | `MLz` | `float` | Injected redshifted lens mass | 
    | `lp` | `float` | Injected lens parameter of the lens |

- `waveform_arguments`: the keyword arguments to be passed to the `bilby.gw.waveform_generator.WaveformGenerator` object in `bilby`.
- `sampler_settings`: the keyword arguments to be passed to the `bilby.run_sampler` function.
- `interpolator_settings`: the settings of the interpolator used, see the tutorial for loading the interpolation tables for more details.
- `prior_settings`: the settings for the priors of the lensing related parameters and `luminosity_distance`. For example, the prior type (`uniform`, `loguniform`) and limits of `MLz` can be specified by `MLz_prior_type`, `MLz_min` and `MLz_max`. For `luminosity_distance`, the prior can be chosen to be `uniformsourceframe`. Additional arguements for masking the region near the caustic are as follows:

    | Name | Type | Description |
    | ----------- | ----------- | ----------- |
    | `crit_mask_settings` | `dict`| Parameters of the masker around the caustic, including `fac`, `cap_high` and `cap_low`. The masked region is defined to be between `y_c - min( y_c * fac, y_c - cap_low )` and `y_c + min( y_c * fac, y_c + cap_high)`. |
    | `mask_boxes` | `list` | Each item is a list `[[y_low, y_high],[l_low, l_high]]`, the boundaries of a rectangular region to exclude in the $y$-$l$ plane |
    
- `misc`: miscellaneous settings as follows:

    | Name | Type | Description |
    | ----------- | ----------- | ----------- |
    | `zero_noise` | `bool` | Whether to perform a zero-noise injection. |
    | `sampling_frequency` | `float` | The sampling rate of the signal in Hertz. |
    | `minimum_frequency` | `float` | Minimum frequency cut-off. |
    | `seed` | `int` | The seed used to generate random numbers for sampling. |
    | `lp_name` | `str` | The name given to the lens parameter, e.g. `kappa` for the NFW lens. |
    | `lp_latex` | `str` | The name of the lens parameter in latex form for making plots |
    | `outdir_ext` | `str` | The name of the output directory |
