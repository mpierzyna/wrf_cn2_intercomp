import argparse
import dataclasses
import pathlib
from typing import Union

import netCDF4
import numpy as np
import wrf
import xarray as xr

if wrf.omp_enabled:
    # Enable multiprocessing
    wrf.omp_set_num_threads(16)
    print(f"OpenMP enabled using {wrf.omp_get_max_threads()} threads.")
else:
    import warnings

    warnings.warn("Your system does not seem to support multiprocessing through OpenMP. "
                  "Only single core performance available.")


@dataclasses.dataclass
class CrossSection:
    """ Cross section between point A(LAT, LON) and point B(LAT, LON) with n interpolation points. """
    p_A: np.ndarray
    p_B: np.ndarray
    n: int

    @property
    def p_interp(self) -> np.ndarray:
        """ Interpolation points between A and B """
        p_interp = np.linspace(0, 1, self.n)
        p_interp = self.p_A + (self.p_B - self.p_A) * p_interp[:, None]
        return p_interp


@dataclasses.dataclass
class WRFEnv:
    WRF_BASE_DIR: pathlib.Path
    RES_DIR: Union[str, pathlib.Path]  # result directory inside WRF_BASE_DIR
    WRF_FILE: Union[str, pathlib.Path]  # wrf file inside result directory

    def __post_init__(self):
        # Convert strs to paths if not yet happend
        if isinstance(self.RES_DIR, str):
            self.RES_DIR = self.WRF_BASE_DIR / self.RES_DIR
        if isinstance(self.WRF_FILE, str):
            self.WRF_FILE = self.RES_DIR / self.WRF_FILE

        # Check that everything exists
        assert self.WRF_BASE_DIR.exists()
        assert self.RES_DIR.exists()
        assert self.WRF_FILE.exists()


def write_ds_compressed(ds: xr.Dataset, output_path: Union[pathlib.Path, str], **kwargs):
    """Write xarray dataset with maxmimum compression on each data variable"""
    ds.to_netcdf(output_path, engine="h5netcdf", encoding={
        var: {
            "zlib": True,
            "complevel": 9
        } for var in ds.data_vars
    }, **kwargs)


def extract_cross_section(wrfenv: WRFEnv, cs: CrossSection, enforce_vars: bool = False) -> xr.Dataset:
    """
    enforce_vars: bool
        Flag deciding if processing should raise exception if not all requested variables are present 
    """
    # Open dataset (lazy loaded)
    print("-----")
    print(f"Processing {wrfenv.WRF_FILE}...")
    wrf_file = netCDF4.Dataset(wrfenv.WRF_FILE)

    # Get interpolation points
    p_interp = cs.p_interp
    print(f"Interpolating fields along {p_interp}")

    # Convert to grid coordinates
    # ATTENTION! Lat and long are inverted here compared to netCDF files so indices are also inverted!
    xy = wrf.ll_to_xy(wrf_file, latitude=p_interp[:, 0], longitude=p_interp[:, 1], meta=False, as_int=False)

    # Split matrix into vectors and assign new dimension to index interpolation path
    x, y = xy
    x = xr.DataArray(x, dims="path")
    y = xr.DataArray(y, dims="path")

    ## Extract variables for defined coordinates
    # Add string of variable names to be extracted.
    # If destaggering needs to happen, add tuple with varname and destagger_dim.
    vars_to_extract = ["z", "HGT", "uvmet", "wa", "th", "tk", "rh", "p", "PBLH", "LANDMASK", "QRAIN", "dbz", "slp",
                       "T2", "TH2", "LH", "HFX", "ZNT", "Z0", "UST"]
    cn2_vars = ["QKE", ("EL_PBL", "bottom_top_stag"), "TSQ", "TKE"]

    # Quick availability check of variables
    vars_to_extract += cn2_vars
    for v in vars_to_extract:
        if isinstance(v, tuple):
            v, _ = v
        if v not in wrf_file.variables:
            msg = f"The requested variable {v} could not be found in {wrfenv.WRF_FILE.name}."
            if enforce_vars:
                raise ValueError(f"{msg} Exiting!")
            else:
                print(f"{msg} Keep going nevertheless.")

    # Start actual extraction
    res = {}
    for v in vars_to_extract:
        # Optionally: Extract dimension along which data will be destaggered
        if isinstance(v, tuple):
            v, destagger_dim = v
        else:
            destagger_dim = None

        # Get variable for all timesteps (ATTENTION! This is heavy on memory!)
        print(f"Reading {v}... ", end="")
        try:
            v_data = wrf.getvar(wrf_file, v, timeidx=wrf.ALL_TIMES)
        except ValueError:
            if enforce_vars:
                raise
            else:
                print("Skipping because it does not exist! ")
                continue

        # Interpolate along path
        print(f"Interpolating horizontally... ", end="")
        v_data = v_data.interp(south_north=y, west_east=x)

        # Now destagger
        if destagger_dim:
            destagger_dim_ind = v_data.dims.index(destagger_dim)
            print(f"Destaggering along {destagger_dim} ({destagger_dim_ind})... ", end="")
            v_data = wrf.destagger(v_data, destagger_dim_ind, meta=True)

        # Serialise object attributes for later netcdf storage
        v_data.attrs["projection"] = v_data.attrs["projection"].proj4()
        v_data = v_data.drop_vars("latlon_coord", errors="ignore")

        # Store it. Maybe write to output file directly to release RAM
        res[v] = v_data
        print(f"Done!")

    # Collect everything into a single dataset
    res_ds = xr.Dataset(res)
    res_ds = res_ds.drop_vars("latlon_coord", errors="ignore")
    res_ds.attrs["path_lat_long"] = p_interp  # Points along which grid cells were selected

    ## Manually post-process some data
    # Split wind velocities into individual variables
    res_ds["u_met"] = res_ds["uvmet"][0]
    res_ds["v_met"] = res_ds["uvmet"][1]
    res_ds = res_ds.drop_vars(["uvmet", "u_v"])

    # Rename vertical wind velocity
    res_ds = res_ds.rename({"wa": "w"})

    return res_ds


if __name__ == "__main__":
    # Cross-section to extract
    cesar_cs = CrossSection(
        # p_A=np.array([51.9702845, 4.9262518]),
        # p_B=np.array([52.0100437, 5.053328]),
        p_A=np.array([51.983, 4.9262518]),
        p_B=np.array([52.0100437, 5.047]),
        n=10
    )

    # Parse arguments
    parser = argparse.ArgumentParser(description="Extract cross-sections from ensemble members")
    parser.add_argument("wrf_root", type=pathlib.Path, help="WRF root directory")
    parser.add_argument("output_dir", type=pathlib.Path, help="Output directory")
    args = parser.parse_args()

    # Set paths
    WRF_ROOT = args.wrf_root
    OUTPUT_DIR = args.output_dir / WRF_ROOT.name
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"Extracting simulation {WRF_ROOT} into {OUTPUT_DIR}...")

    wrfenv = WRFEnv(
        WRF_BASE_DIR=WRF_ROOT,
        RES_DIR=".",
        WRF_FILE=list(sorted(WRF_ROOT.glob("wrfout_10min_d03*")))[0],  # Automatically select d03 10min wrfout
    )

    out_path = OUTPUT_DIR / f"{WRF_ROOT.name}_{wrfenv.WRF_FILE.name}.nc"
    if out_path.exists():
        print(f"{out_path} exists already. Skipping.")
    else:
        ds = extract_cross_section(wrfenv=wrfenv, cs=cesar_cs)
        write_ds_compressed(ds, out_path)
        print(f"Written to {out_path}.")

    print("Done!")
