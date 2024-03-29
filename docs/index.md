<h1 align="center">
    glworia
</h1>

<h4 align="center"> A python package for gravitational-wave lensing computations including wave-optics effects. </h4>

<p align="center">
    <a href = "https://arxiv.org/abs/2403.13876"><img src="https://img.shields.io/badge/arXiv-2403.13876-b31b1b.svg"></a>
    <a href="https://badge.fury.io/py/glworia"><img src="https://badge.fury.io/py/glworia.svg"></a>
    <a href="https://github.com/mhycheung/glworia/actions/workflows/pytest.yml "><img src="https://github.com/mhycheung/glworia/actions/workflows/pytest.yml/badge.svg"></a>
    <a href="https://github.com/mhycheung/glworia/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg"></a>
    <a href="https://pypi.org/project/glworia/"><img src="https://img.shields.io/pypi/pyversions/glworia"></a>
</p>

## Key Features

* Compute the frequency-dependent lensing amplification factor
* Use your custom lens model, any spherically symmetry lens is supported
* The only function you need to provide is the Fermat potential - `jax` will take care of the rest with auto differentiation!
* Build interpolation tables for your lens model
* Perform Bayesian parameter estimation with `bilby`
* Runs on GPUs

## Installation

```shell
pip install glworia
```

## Usage

Basic usage examples can be found under the Examples tab on the left.

## Parameter estimation results

The full corner plots for the parameter estimation runs shown in the companion paper can be found in the `plots/` directory.
The setting files for these runs are in the `PE_submission_settings/` directory.

## How to Cite
Please cite the methods paper if you used our package to produce results in your publication.
Here is the BibTeX entry:
```
@article{Cheung:2024ugg,
    author = "Cheung, Mark Ho-Yeuk and Ng, Ken K. Y. and Zumalac\'arregui, Miguel and Berti, Emanuele",
    title = "{Probing minihalo lenses with diffracted gravitational waves}",
    eprint = "2403.13876",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    month = "3",
    year = "2024"
}
```

## License

MIT

---

> GitHub [@mhycheung](https://github.com/mhycheung)