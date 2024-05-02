from typing import Dict, List
import datetime

import numpy as np
import xarray as xr
from scipy.signal import butter, sosfiltfilt
import matplotlib.dates as mdates


def butter_filter(y: np.ndarray, fs: float, cutoff_freq: float, order: int) -> np.ndarray:
    """ Apply general Butterworth low-pass filter to ``y``.
    Use only half of ``order`` because ``filtfilt`` performs forward and backward pass doubling the order
    """
    nyq = 0.5 * fs  # Nyquist sampling frequency
    normal_cutoff = cutoff_freq / nyq
    sos = butter(order // 2, normal_cutoff, btype='low', analog=False, output="sos")
    return sosfiltfilt(sos, y)


def get_wrf_smoothed(a: xr.DataArray, cutoff_period_h: float) -> xr.DataArray:
    if "path" in a.dims:
        a = a.mean("path")
    a_smoothed = butter_filter(
        a.to_numpy(),
        fs=1 / (10 * 60),  # 10 min WRF output interval
        cutoff_freq=1 / (cutoff_period_h * 60 * 60),  # cutoff period in hours
        order=3
    )
    a_smoothed = xr.DataArray(a_smoothed, dims=["time"], coords={"time": a["time"]})
    return a_smoothed


def hour_formatter(x: int, pos=0) -> str:
    x: datetime.datetime = mdates.num2date(x)
    _, _, hour = x.month, x.day, x.hour
    return f"{hour:02d}h"


def filter_most(ds: xr.Dataset, blh: xr.DataArray) -> xr.Dataset:
    """Filter out data that is not valid for MOST-based parameterizations."""
    z = ds["z_msl"]  # Instrument height
    slh = blh * .1

    for v in ds:
        if "cn2_w71" in v:
            ds[v] = ds[v].where(z <= slh)

    return ds


def cn2_wrf_to_xlas_obs(ds_wrf: xr.Dataset, blh: xr.DataArray, weighted_mean: bool = True) -> xr.Dataset:
    ds_wrf = ds_wrf.interp(z_msl=80)

    if weighted_mean:
        # Weighted mean computation converts NaN values to zero. Store original NaNs and restore later.
        ds_wrf_is_nan = ds_wrf.isnull().any("path")

        n = ds_wrf.sizes["path"]
        w = xr.DataArray(
            data=np.sin(np.linspace(0, np.pi, n)) ** (7 / 3),
            dims=["path"],
        )
        w = w / w.sum()  # normalize
        ds_wrf = (ds_wrf * w).sum("path")
        ds_wrf = ds_wrf.where(~ds_wrf_is_nan)  # Restore NaN
    else:
        ds_wrf = ds_wrf.mean("path")

    ds_wrf["is_daytime"] = ds_wrf["is_daytime"].astype(bool)  # convert back to bool after above operations
    ds_wrf["is_stable"] = ds_wrf["is_stable"].astype(bool)  # convert back to bool after above operations

    # Only keep data where MOST is valid during unstable conditions
    ds_wrf = filter_most(ds_wrf, blh=blh)
    for v in ds_wrf:
        if "cn2_w71" in v:
            ds_wrf[v] = ds_wrf[v].where(~ds_wrf["is_stable"])

    return ds_wrf


def cn2_wrf_to_las_obs(ds_wrf: xr.Dataset, blh: xr.DataArray) -> xr.Dataset:
    ds_wrf = ds_wrf.isel(path=0).interp(z_msl=10)
    ds_wrf = filter_most(ds_wrf, blh=blh)
    return ds_wrf


def load_xlas_obs(path: str) -> xr.Dataset:
    """  Load XLAS observation data and apply quality checks.

    Attention! Shift of 3h (6 time steps) needs to be applied to data in spring and summer 2022.
    Do this during plotting because interval for shift is not exactly known.
    """
    ds = xr.open_dataset(path)
    ds["valid"] = ds["LAS_nofilt_raw_avg"] > 1000
    ds["Cn2_os"] = ds["Cn2_os"].where(ds["valid"])
    return ds


def load_las_obs(path: str) -> xr.Dataset:
    """ Load LAS observation data and apply quality checks. """
    ds = xr.open_dataset(path)
    ds["valid"] = ds["LAS_nofilt_raw_avg"] > 15000
    ds["Cn2_os"] = ds["Cn2_os"].where(ds["valid"])
    return ds


class Cn2Comparer:
    """ Compare WRF simulations to observations."""
    xlas_wrf_day_only: Dict[str, bool] = {
        "w71": True,  # MOST-based, so not valid for XLAS during night
        "w71_hb16": True,  # MOST-based, so not valid for XLAS during night
        "phys": False,  # TKE based, so valid for all times
    }

    def __init__(self, ds_obs_las: xr.Dataset, ds_obs_xlas: xr.Dataset,
                 ds_wrf_list: List[xr.Dataset], member_mask: List[bool], wrf_labels: List[str],
                 xlas_obs_shift: List = None, blh_key: str = "blh_era5"):
        # Process shifting value
        if xlas_obs_shift is None:
            xlas_obs_shift = [0 for _ in ds_wrf_list]
        else:
            # One shift value per wrf simulation
            assert len(xlas_obs_shift) == len(ds_wrf_list)

        ### OBS: Select subset of observations matching each simulation
        # LAS
        self.ds_obs_las_list = [
            ds_obs_las.sel(time=slice(*ds_wrf.time[[0, -1]].values))
            for ds_wrf in ds_wrf_list
        ]
        # XLAS: Shift values for some simulations according to `xlas_shift`
        self.ds_obs_xlas_list = [
            ds_obs_xlas.shift(time=dt).sel(time=slice(*ds_wrf.time[[0, -1]].values))
            for ds_wrf, dt in zip(ds_wrf_list, xlas_obs_shift)
        ]
        self.ds_obs_list = list(zip(self.ds_obs_las_list, self.ds_obs_xlas_list))  # for convenience

        ### WRF: Match simulated Cn2 to obs (correct height, location, aggregation)
        self.ds_wrf_las_list = [
            cn2_wrf_to_las_obs(
                ds_wrf.isel(member=member_mask),
                blh=ds_wrf[blh_key],
            )
            for ds_wrf in ds_wrf_list
        ]
        self.ds_wrf_xlas_list = [
            cn2_wrf_to_xlas_obs(
                ds_wrf.isel(member=member_mask),
                weighted_mean=True,
                blh=ds_wrf[blh_key],
            )
            for ds_wrf in ds_wrf_list
        ]

        ### Other stuff
        self.n_wrf = len(ds_wrf_list)
        self.wrf_labels = wrf_labels
        self.ds_wrf_list = list(zip(self.ds_wrf_las_list, self.ds_wrf_xlas_list))  # for convenience
