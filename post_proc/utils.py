import numpy as np
import xarray as xr

from misc_tools.math import gradient_nonuniform


def compute_grad(a: xr.DataArray, grad_dim: str, a_aux_bottom: xr.DataArray = None) -> xr.DataArray:
    """Compute gradient of a with respect to z_msl.
    a : xr.DataArray
        Array to compute gradient of.
    grad_dim : str
        Name of dimension along which to compute gradient.
    a_aux_bottom : xr.DataArray
        Auxiliary array with same shape as `a` to support gradient computation at bottom boundary.
    """
    z = a[grad_dim].drop_vars(grad_dim)
    z_orig = None

    if a_aux_bottom is not None:
        # Aux level z might have different shape than z. Expand `z` to `z_aux` shape (after backup).
        z_aux_bottom = a_aux_bottom[grad_dim].drop_vars(grad_dim)
        z_orig = z.copy()
        z = z.expand_dims(z_aux_bottom.sizes)
        if grad_dim not in z_aux_bottom.dims:
            z_aux_bottom = z_aux_bottom.expand_dims({grad_dim: 1})  # Expand aux z to one level of z_key dim

        # Concat aux z to main z
        z = xr.concat([z_aux_bottom, z], dim=grad_dim)

        # Also expand a_aux_bottom to match dim of `a`
        if grad_dim not in a_aux_bottom.dims:
            a_aux_bottom = a_aux_bottom.drop_vars(grad_dim).expand_dims({grad_dim: 1})

        # Concat aux `a` to main `a`
        coords_diff = set(a.coords) - set(a_aux_bottom.coords)  # drop coords that are not in a_aux_bottom for concat
        a = xr.concat([a_aux_bottom, a.drop_vars(coords_diff)], dim=grad_dim)

    # Roll `grad_dim` to first dim to make broadcasting work (after storing original dim order)
    a_dims = a.dims
    if grad_dim in a.coords:
        a = a.drop_vars(grad_dim)
    a = a.transpose(grad_dim, ...)
    z = z.transpose(grad_dim, ...)

    # Compute central difference gradient
    dadz = xr.DataArray(data=np.empty_like(a), dims=a.dims, coords=a.coords)
    dadz[1:-1] = gradient_nonuniform(x=z, y=a, opt=1)  # noqa: Preproc above allows broadcasting of xr.DataArray

    # Compute finite difference gradients at lower and upper boundary
    dadz[0] = (a[1] - a[0]) / (z[1] - z[0])
    dadz[-1] = (a[-1] - a[-2]) / (z[-1] - z[-2])

    # Restore original dim order and coords
    dadz = dadz.transpose(*a_dims)

    # Remove auxiliary index from gradient because it is not physical
    if a_aux_bottom is not None:
        dadz = dadz.isel({grad_dim: slice(1, None)})
        z = z_orig

    # Restore z coordinate
    dadz.coords[grad_dim] = z

    return dadz


def round_timestamp(ds: xr.Dataset) -> xr.Dataset:
    """Round the timestamps of the dataset to the nearest minute."""
    return ds.assign_coords(time=ds.time.dt.round("1min"))


def clip_bowen(bo: xr.DataArray) -> xr.DataArray:
    bo_neg_min, bo_neg_max = -10, -0.1
    bo_pos_min, bo_pos_max = 0.1, 10
    bo = bo.clip(bo_neg_min, bo_pos_max)  # Clip very high and very low values
    bo = xr.where((bo < 0) & (bo > bo_neg_max), bo_neg_max, bo)  # Clip negative values
    bo = xr.where((bo > 0) & (bo < bo_pos_min), bo_pos_min, bo)  # Clip positive values
    return bo
