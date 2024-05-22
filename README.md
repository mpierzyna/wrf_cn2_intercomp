# Intercomparison of flux, gradient, and variance-based optical turbulence ($C_n^2$) parameterizations

## References

This repository accompanies the work published in Applied Optics as

> Pierzyna, M., Hartogensis, O., Basu, S., and Saathof R. "Intercomparison of flux, gradient, and variance-based optical turbulence ($C_n^2$) parameterizations." _Applied Optics_, vol. 63, no. 16, 2024, pp. E107-E119. [https://doi.org/10.1364/AO.519942]([https://doi.org/10.1364/AO.519942])

The simulated and observed data discussed in our article and utilized by the plotting functions of this repository ([`eval`](eval)) are published on Zenodo

> Pierzyna, M. "Dataset: Intercomparison of flux, gradient, and variance-based optical turbulence ($C_n^2$) parameterizations." In Applied Optics. Zenodo. [https://doi.org/10.5281/zenodo.10966120](https://doi.org/10.5281/zenodo.10966120).

## WRF Simulations

All files required to run the [Weather Research and Forecasting (WRF) model](https://github.com/wrf-model/WRF/releases/tag/v4.5.1) are contained in the [`simulation`](simulation) directory of this repository.
The directory contains subdirectories for each test case discussed in the manuscript.
Each subdirectory contains two files:

- `namelist.wps`: Settings for the WRF Preprocessing System (WPS)
- `namelist.input`: Settings for WRF. Importantly, the configuration of the phyiscs schemes is configured here.

The `myoutfields.txt` ensures that all relevant variables written to the output files when WRF is running.
By default, WRF does not output all variables required for the variance-based parameterization such as TSQ (potential temperature variance) or QKE (twice turbulent kinetic energy).
This file needs to be placed in the same directory as `namelist.input`.

## Post-processing: $C_n^2$ estimation

The codes to post-process the WRF simulations and estimated $C_n^2$ are contained in the [`post_proc`](post_proc) directory of this repository.

Post-processing requires two steps, which are run by executing the `step_XX_*.sh` batch files in order:

1. `step_1_extract_variables.sh`: Extract one vertical cross-section of interest from the full WRF output through horizontal interpolation (see `extract_variables.py`).
2. `step_2a_post_proc_wrf.sh`: Run various post-processing steps such as interpolating the WRF pressure levels to a common altitude level, compute stability parameters, or estimating $C_T^2$ and $C_n^2$ (see `post_proc_wrf.py` and `ct2_cn2_estimation.py`)

The `step_2b_post_proc_obs_cn2.py` needs to be run to estimate $C_T^2$ and $C_n^2$ from data observed at the CESAR site, which are available through the KNMI data platform (KDP, [https://dataplatform.knmi.nl/dataset/?q=Cesar](https://dataplatform.knmi.nl/dataset/?q=Cesar))

## Plotting and evaluation

The codes to assess the accuracy of the $C_n^2$ estimates quantitatively (scores) and qualitatively (plots) are contained in the [`eval`](eval) directory of this repository. 
