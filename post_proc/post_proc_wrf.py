import xarray as xr
import pathlib
import numpy as np
import numpy.testing as npt
import matplotlib.pyplot as plt
import argparse
import pandas as pd

from misc_tools.netcdf import write_ds_compressed
from misc_tools.sunsetrise import get_daytime_mask
from ct2_cn2_estimation import get_cn2_all
from utils import compute_grad, clip_bowen

LAT_CESAR, LON_CESAR = 51.971, 4.927


def interp_vert_ens(ds: xr.Dataset, diag_plot: bool = False) -> xr.Dataset:
    """Interpolate all variables in ds to common height above mean sea level across all ensemble members.
    Attention! Happens in-place!
    """
    # Average all heights of all members to have common interpolation basis
    print("Interpolating to average height throughout ensemble...")
    z_mean = ds["z"].mean(["path", "time"])

    # Diagnostic plot for spread of heights in ensemble
    if diag_plot:
        fig, ax = plt.subplots()
        dz = (z_mean - z_mean.rename(member="member2")).transpose("bottom_top", ...)
        dz.plot.hist(ax=ax, bins=20)
        fig.show()

    # Average height for all members
    z_mean = z_mean.mean("member")

    # Compute indices of bracketing bottom_top levels
    bt_j = xr.apply_ufunc(
        np.searchsorted,
        ds["z"],
        z_mean.rename(bottom_top="z_msl"),
        kwargs={"side": "left"},
        input_core_dims=[["bottom_top"], ["z_msl"]],
        exclude_dims={"z_msl"},
        output_core_dims=[["z_msl"]],
        vectorize=True,
    )
    bt_i = (bt_j - 1).clip(min=0)
    bt_j = bt_j.clip(max=ds.sizes["bottom_top"] - 1)

    # Get heights for bracketing levels
    z_i = ds["z"].sel(bottom_top=bt_i)
    z_j = ds["z"].sel(bottom_top=bt_j)

    # Get data for bracketing levels
    vars_interp = [v for v in ds if "bottom_top" in ds[v].dims]
    ds_i = ds[vars_interp].sel(bottom_top=bt_i)
    ds_j = ds[vars_interp].sel(bottom_top=bt_j)

    # Interpolate to common height
    ds_interp = ds_i + (z_mean.rename(bottom_top="z_msl") - z_i) * (ds_j - ds_i) / (z_j - z_i)
    for v in vars_interp:
        # Attributes are lost in interpolation. Restore them.
        ds_interp[v].attrs.update(ds[v].attrs)

    # Sanity check interpolation, then drop old z variable
    npt.assert_array_almost_equal(
        ds_interp["z"].std(["member", "time", "path"]),
        0
    )

    # Fill NaN values with original value before interpolation
    ds_interp = ds_interp.fillna(ds[vars_interp].rename(bottom_top="z_msl"))
    ds_interp = ds_interp.assign_coords(
        z_msl=z_mean.rename(bottom_top="z_msl").assign_attrs(units="m", description="Height above mean sea level")
    )

    # Add interpolated variables to original dataset
    ds[vars_interp] = ds_interp
    """ Disable because confusing (redundant) and not used
    ds = ds.assign_coords(
        z_agl=(ds["z_msl"] - ds["HGT"]).assign_attrs(units="m", description="Height above ground level"))
    """
    ds = ds.drop_vars("z")

    return ds


def post_process_ens(ens_root: pathlib.Path, n_warmup: int) -> None:
    # Load ensemble
    print(f"Loading ensemble from {ens_root}...")
    ds_list = [
        xr.open_dataset(f, engine="h5netcdf")
        for f in sorted(ens_root.glob("WRF_base_*"))  # sorting is important!
    ]

    # Add member dimension for merge
    print("Merging...")
    ds_list = [
        ds.expand_dims(dim={"member": [i]})
        for i, ds in enumerate(ds_list)
    ]

    ds = xr.merge(ds_list)
    ds = ds.rename(Time="time")
    ds = ds.drop_vars(["XLAT", "XLONG"])  # Drop because contained in coords south_north and west_east.

    # Drop warmup period
    ds = ds.isel(time=slice(n_warmup, None))

    # Add secondary `member` coordinate with human-readable member names
    member_names = [
        "YSU/MM5",
        "MYJ/Eta",
        "QNSE/QNSE",
        "MYNN2.5/MYNN",
        # "MYNN3/MYNN",
        "BouLac/Eta",
        "GBM/MM5",
        r"E-$\epsilon$/MYNN",
        "ACM2/MM5",
        "UW/MYNN",
    ]
    assert len(member_names) == len(ds_list)
    ds = ds.assign_coords(member_name=xr.DataArray(member_names, dims="member"))

    # Drop member dimension for variables that are constant in time and between members
    print("Dropping constant variables...")
    vars_const = ["HGT", "LANDMASK"]
    for v in vars_const:
        npt.assert_almost_equal(ds[v].std("member").sum(), 0)  # Sanity check
        npt.assert_almost_equal(ds[v].std("time").sum(), 0)  # Sanity check
        ds[v] = ds[v].isel(member=0, time=0).drop_vars(["member", "time"])

    # Interpolate vertically
    ds = interp_vert_ens(ds, diag_plot=True)

    ## Add derived variables
    print("Adding derived variables...")

    # Horizontal wind speed and direction (https://www.eol.ucar.edu/content/wind-direction-quick-reference)
    ds["M"] = np.sqrt(ds["u_met"] ** 2 + ds["v_met"] ** 2)
    ds["M"] = ds["M"].assign_attrs(units="m/s", description="Horizontal wind speed")
    ds["D"] = np.rad2deg(np.arctan2(-ds["u_met"], -ds["v_met"]))
    ds["D"] = ds["D"].assign_attrs(units="deg", description="Horizontal wind direction")

    # Potential temperature gradient. Use T2 as auxiliary variable for bottom level.
    th2 = ds["TH2"]
    th2 = th2.assign_coords(
        z_msl=ds["HGT"] + 2  # terrain height + 2m. Gradient computation happens in MSL coordinates.
    )
    ds["Gamma"] = compute_grad(a=ds["th"], grad_dim="z_msl", a_aux_bottom=th2)
    ds["Gamma"] = ds["Gamma"].assign_attrs(units="K/m", description="Potential temperature gradient")

    # Wind shear. Use artificial zero velocity at z0 (roughness length, assuming log-wind profile)
    # to improve lowest level gradient.
    z0 = ds["Z0"]
    zero_vel = xr.DataArray(
        data=np.zeros_like(z0),
        dims=z0.dims,
        coords={"z_msl": ds["HGT"] + z0}  # here, z0 + terrain is set as height for zero velocity
    )
    dudz = compute_grad(a=ds["u_met"], a_aux_bottom=zero_vel, grad_dim="z_msl")
    dvdz = compute_grad(a=ds["v_met"], a_aux_bottom=zero_vel, grad_dim="z_msl")
    ds["S"] = np.sqrt(dudz ** 2 + dvdz ** 2)
    ds["S"] = ds["S"].assign_attrs(units="1/s", description="Wind shear")

    # SQUARED Bruntt-Vaisala frequency (don't store because not super interesting)
    g = 9.81  # m/s^2
    N_sq = g / ds["th"] * ds["Gamma"]

    # Gradient Richardson number
    ds["Ri_g"] = N_sq / ds["S"] ** 2
    ds["Ri_g"] = ds["Ri_g"].assign_attrs(units="", description="Gradient Richardson number")

    # Bowen ratio
    bo = ds["HFX"] / ds["LH"]

    # Follow Oscars suggestions for clipping
    bo_unclipped = bo.copy()
    bo = clip_bowen(bo)

    # Compute bowen correction factor
    bo_corr = (1 + 0.03 / bo) ** 2

    # Write to disk
    print("Writing to disk...")
    write_ds_compressed(ds, ens_root.parent / f"{ens_root.name}_merged.nc")

    # Estimate Cn2 and add daytime mask for validity of MOST-based Cn2
    print("Estimating Cn2...")
    ds_cn2 = get_cn2_all(ds)
    ds_cn2["bo"] = bo
    ds_cn2["bo_unclipped"] = bo_unclipped
    ds_cn2["bo_corr"] = bo_corr

    # Load ERA5 boundary layer height matching WRF time
    era5_blh = xr.open_mfdataset("../data/ERA5/BLH_*.grb")["blh"]
    era5_blh = era5_blh.sel(latitude=LAT_CESAR, longitude=LON_CESAR, method="nearest")
    era5_blh = era5_blh.interp(time=ds.time)
    ds_cn2["blh_era5"] = era5_blh
    ds_cn2["blh_wrf"] = ds["PBLH"]

    # Add daytime and stability mask
    ds_cn2["is_daytime"] = xr.DataArray(get_daytime_mask(pd.to_datetime(ds.time), lat=52.0, lon=4.9), dims="time")
    ds_cn2["is_stable"] = (ds["HFX"] < 0).median("member").isel(path=0)

    print("Writing to disk...")
    write_ds_compressed(ds_cn2, ens_root.parent / f"{ens_root.name}_merged_cn2.nc")

    print("Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("ens_root", type=pathlib.Path, help="Path to ensemble root directory.")
    parser.add_argument("--n_warmup", type=int, default=6 * 12,
                        help="Number of warmup time steps to drop. Default: 12h at dt=10min -> 72 time steps.")
    args = parser.parse_args()

    post_process_ens(args.ens_root, args.n_warmup)
