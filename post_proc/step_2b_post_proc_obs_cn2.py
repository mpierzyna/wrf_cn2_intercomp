import pathlib
import xarray as xr
import numpy as np

from utils import round_timestamp, compute_grad, clip_bowen
from turb_tools.cn2 import wyngaard_1971 as w71
from turb_tools.cn2 import he_basu_2016 as hb16
from turb_tools import most
from turb_tools.base import theta_p, hypsometric_p, Ri_g, gladstone
from ct2_cn2_estimation import CT2_ATTRS, CN2_ATTRS

# %% Load CESAR data
CESAR_ROOT = pathlib.Path("../data/CESAR")
assert CESAR_ROOT.exists()

ds_meteo_flux = round_timestamp(xr.open_mfdataset(
    (CESAR_ROOT / "Meteo_Flux").glob("cesar_surface_flux_*.nc")  # noqa
)).load()
ds_meteo_tower = round_timestamp(xr.open_mfdataset(
    (CESAR_ROOT / "Meteo_Tower").glob("cesar_tower_meteo_*.nc")  # noqa
)).load()
ds_meteo_surface = round_timestamp(xr.open_mfdataset(
    (CESAR_ROOT / "Meteo_Surface").glob("cesar_surface_meteo_*.nc")  # noqa
)).load()

# %% Pressure from hypsometric equation
tk = ds_meteo_tower["TA"]  # K, air temperature
p_hPa = []

tk_ref = ds_meteo_surface["TA000"]  # K, close to surface temperature
p_hPa_ref = ds_meteo_surface["P0"]  # hPa
z_ref = 0  # m

for z in tk.z:
    tk_i = tk.sel(z=z)
    p_hPa_i = hypsometric_p(T_K=tk_i, z=z, T_ref_K=tk_ref, z_ref=z_ref, p_ref=p_hPa_ref)
    p_hPa.append(p_hPa_i)

    # Current level becomes new reference
    tk_ref = tk_i
    p_hPa_ref = p_hPa_i
    z_ref = z

p_hPa = xr.concat(p_hPa, dim=tk.z)

# %% Compute smoothed bowen ratio
shfx_smoothed = ds_meteo_flux["HSON"].rolling(time=6, center=True).mean()
lhfx_smoothed = ds_meteo_flux["LEED"].rolling(time=6, center=True).mean()
bowen_ratio = shfx_smoothed / lhfx_smoothed

# Follow Oscars suggestions for clipping
bowen_ratio_unclipped = bowen_ratio.copy()
bowen_ratio = clip_bowen(bowen_ratio)

# Compute bowen correction factor
bowen_corr = (1 + 0.03 / bowen_ratio) ** 2

# %% Flux-based W71 CT2
shfx_kin = ds_meteo_flux["HSON"] / 1.216e3  # (W/m^2) to (Km/s)
ust_local = ds_meteo_flux["USTAB"]  # m/s, local
ust_regional = ds_meteo_flux["USTPR"]  # m/s, regional

ct2_flux_local = w71.CT2_flux(
    T_st=-shfx_kin / ust_local,
    z=tk.z,
    L=most.L(u_st=ust_local, shfx_msK=shfx_kin, T_K=tk)
)
ct2_flux_regional = w71.CT2_flux(
    T_st=-shfx_kin / ust_regional,
    z=tk.z,
    L=most.L(u_st=ust_regional, shfx_msK=shfx_kin, T_K=tk)
)

# %% Gradient-based W71 CT2
theta_K = theta_p(T_K=tk, p=p_hPa)  # p_ref in hPa, so pressure also given in hPa
Gamma = compute_grad(a=theta_K, grad_dim="z")

# Get wind components from wind speed and direction.
M = ds_meteo_tower["F"]
D = np.deg2rad(ds_meteo_tower["D"])

# Drop lowest level because it is nan anyway
M = M.isel(z=M.z > 2)
D = D.isel(z=D.z > 2)

# Compute components
u = -np.sin(D) * M
v = -np.cos(D) * M

# Assume zero velocity at roughness height
z0 = 1e-2  # depends on wind direction, but for simplicity, we keep it constant.
zero_vel = xr.DataArray(
    data=np.zeros((M.sizes["time"], 1)),
    dims=["time", "z"],
    coords={"z": [z0]}
)
u = xr.concat([u, zero_vel], dim="z")
v = xr.concat([v, zero_vel], dim="z")

# Compute gradients and wind shear
dudz = compute_grad(a=u, grad_dim="z")
dvdz = compute_grad(a=v, grad_dim="z")
S = np.sqrt(dudz ** 2 + dvdz ** 2)

# Remove assumed zero_velocity again
S = S.isel(z=S.z > 2)

# Combine Gamma and S to Richardson number
Ri = Ri_g(T_K=tk, Gamma=Gamma, S=S)

ct2_w71 = w71.CT2_grad(Ri=Ri, z=Gamma.z, Gamma=Gamma)
ct2_hb16 = hb16.CT2_grad(Ri_g=Ri, z=z, Gamma=Gamma)  # Returns NaN when Ri <= 0 (unstable)
ct2_grad = ct2_hb16.fillna(ct2_w71)  # Combine W71 and HB16

# %% Convert all to Cn2, too
cn2_flux_local = gladstone(CT2=ct2_flux_local, P_hPa=p_hPa, T_K=tk)
cn2_flux_regional = gladstone(CT2=ct2_flux_regional, P_hPa=p_hPa, T_K=tk)
cn2_grad = gladstone(CT2=ct2_grad, P_hPa=p_hPa, T_K=tk)

# %% Store in combined dataset
ds = xr.Dataset(
    data_vars={
        "ct2_w71_flux_local": ct2_flux_local.assign_attrs(CT2_ATTRS),
        "ct2_w71_flux_regional": ct2_flux_regional.assign_attrs(CT2_ATTRS),
        "ct2_w71_hb16_grad": ct2_grad.assign_attrs(CT2_ATTRS),
        "cn2_w71_flux_local": cn2_flux_local.assign_attrs(CN2_ATTRS),
        "cn2_w71_flux_regional": cn2_flux_regional.assign_attrs(CN2_ATTRS),
        "cn2_w71_hb16_grad": cn2_grad.assign_attrs(CN2_ATTRS),
        "bo": bowen_ratio,
        "bo_unclipped": bowen_ratio_unclipped,
        "bo_corr": bowen_corr,
    },
    coords={
        "z": tk.z,
        "time": tk.time,
    },
)
ds.to_netcdf("../data/CESAR/cesar_cn2_ct2.nc")
