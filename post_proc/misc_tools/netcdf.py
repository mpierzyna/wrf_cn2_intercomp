import pathlib
from typing import Union
import xarray as xr


def write_ds_compressed(ds: xr.Dataset, output_path: Union[pathlib.Path, str], **kwargs):
    """Write xarray dataset with maxmimum compression on each data variable"""
    ds.to_netcdf(output_path, engine="h5netcdf", encoding={
        var: {
            "zlib": True,
            "complevel": 9
        } for var in ds.data_vars
    }, **kwargs)
