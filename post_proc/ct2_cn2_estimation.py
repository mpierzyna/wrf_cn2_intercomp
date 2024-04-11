import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from turb_tools.cn2 import he_basu_2016 as hb16
from turb_tools.cn2 import wyngaard_1971 as w71

CT2_ATTRS = {
    "description": "Structure parameter of temperature fluctuations",
    "units": "K^2 m^(-2/3)"
}
CN2_ATTRS = {
    "description": "Structure parameter of refractive index fluctuations",
    "units": "m^(-2/3)"
}


def get_ct2_physical(ds: xr.Dataset) -> xr.DataArray:
    # Get base variables
    qke = ds["QKE"].clip(min=0, max=None)  # Discard negative QKE values
    l_mix = ds["EL_PBL"].clip(min=1e-4, max=None)  # Set minimum mixing length
    var_theta = ds["TSQ"]

    # Energy dissipation rate (m2/s3)
    B1 = 24
    eps = np.power(qke, 3 / 2) / (B1 * l_mix)

    # Temperature variance dissipation ()
    B2 = 15
    chi = np.sqrt(qke) * var_theta / (B2 * l_mix)

    # Compute CT2
    ct2 = 3.2 * chi * np.power(eps, -1 / 3)
    ct2.attrs.update({
        "method": "Physical",
        **CT2_ATTRS,
    })
    return ct2


def get_ct2_most_w71_hb16(ds: xr.Dataset) -> xr.DataArray:
    """ CT2 using HB16 for Ri > 0 (stable) and W71 for Ri <= 0 (unstable)."""
    Ri_g = ds["Ri_g"]
    z = ds["z_msl"] - ds["HGT"]
    Gamma = ds["Gamma"]

    # Use HB16 for Ri > 0. Ri <= 0 will be NaN
    ct_hb16 = hb16.CT2_grad(Ri_g=Ri_g, z=z, Gamma=Gamma)

    # Fill it up with W71
    ct2_w71 = w71.CT2_grad(Ri=Ri_g, z=z, Gamma=Gamma)
    ct2 = ct_hb16.fillna(ct2_w71)
    ct2.attrs.update({
        "method": "HB16 for Ri > 0, W71 for Ri <= 0",
        **CT2_ATTRS,
    })

    return ct2


def get_ct2_most_w71_grad(ds: xr.Dataset) -> xr.DataArray:
    """ CT2 using W71 for all Ri (gradient method). """
    Ri_g = ds["Ri_g"]
    z = ds["z_msl"] - ds["HGT"]
    Gamma = ds["Gamma"]

    ct2_w71 = w71.CT2_grad(Ri=Ri_g, z=z, Gamma=Gamma)
    ct2_w71.attrs.update({
        "method": "Wyngaard 1971 (gradient-based)",
        **CT2_ATTRS,
    })
    return ct2_w71


def get_ct2_most_w71_flux(ds: xr.Dataset) -> xr.DataArray:
    """ CT2 using W71 for all Ri."""
    u_st = ds["UST"]  # (m/s)
    shfx = ds["HFX"] / 1.216e3  # (W/m2 -> K m/s)
    T_mean = ds["tk"]  # Air temperature (K)
    k = 0.4

    L = -u_st ** 3 * T_mean / (k * shfx * 9.81)  # Obukhov length
    theta_st = -shfx / u_st
    z = ds["z_msl"] - ds["HGT"]

    ct2_w71 = w71.CT2_flux(T_st=theta_st, z=z, L=L)
    ct2_w71.attrs.update({
        "method": "Wyngaard 1971 (flux-based)",
        **CT2_ATTRS,
    })
    return ct2_w71


def ct2_to_cn2(ct2: xr.DataArray, ds: xr.Dataset) -> xr.DataArray:
    """Convert CT2 to CN2 using simplified Gladstone relationship."""
    # Get pressure in hPa
    p = ds["p"]  # Pressure (Pa)
    assert p.units == "Pa"
    p_hPa = p / 100  # Pressure (hPa)

    # Get temperature in K
    t_K = ds["tk"]

    # Gladstone
    cn2 = (7.9e-5 * p_hPa / t_K ** 2) ** 2 * ct2
    cn2 = cn2.clip(min=1e-22, max=None)
    cn2.attrs.update({
        **ct2.attrs,
        **CN2_ATTRS,
    })

    return cn2


def clip_Ri(ds: xr.Dataset, diag_plot: bool = False) -> xr.Dataset:
    """Clip Richardson number to 1.5 IQR."""
    Ri_25 = ds["Ri_g"].quantile(0.25)
    Ri_75 = ds["Ri_g"].quantile(0.75)
    Ri_iqr = Ri_75 - Ri_25
    Ri_min = Ri_25 - 1.5 * Ri_iqr
    Ri_max = Ri_75 + 1.5 * Ri_iqr
    ds["Ri_g"] = ds["Ri_g"].clip(Ri_min, Ri_max)

    if diag_plot:
        ds["Ri_g"].plot.hist()
        plt.show()

    return ds


def get_cn2_all(ds: xr.Dataset) -> xr.Dataset:
    """Compute all Cn2 variants."""
    ds = clip_Ri(ds.copy())
    cn2 = xr.Dataset(
        data_vars={
            # "cn2_w71_grad": ct2_to_cn2(get_ct2_most_w71_grad(ds), ds),
            "cn2_w71_hb16_grad": ct2_to_cn2(get_ct2_most_w71_hb16(ds), ds),
            "cn2_w71_flux": ct2_to_cn2(get_ct2_most_w71_flux(ds), ds),
            "cn2_hb15_var": ct2_to_cn2(get_ct2_physical(ds), ds),
        },
        attrs=CN2_ATTRS
    )

    # Clip Cn2
    cn2 = cn2.clip(min=1e-17, max=None)
    return cn2
