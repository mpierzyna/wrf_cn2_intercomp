import numpy as np
from scipy.interpolate import splrep, splev

f3_Ri_data = np.array([
    [-2, -1.8, -1.6, -1.4, -1.2, -1.0, -.8, -.6, -.4, -.2, -.1, -.08, -.06, -.04, -.02,
     0, .02, .04, .06, .08, .1, .12, .14, .16, .18, .2],
    [3.62, 3.5, 3.37, 3.22, 3.06, 2.89, 2.68, 2.44, 2.14, 1.75, 1.48, 1.42, 1.35, 1.27, 1.19,
     1.09, 0.81, 0.63, 0.5, 0.39, 0.3, 0.22, 0.15, 0.098, 0.051, 0.015],
])
f3_Ri_tck = splrep(
    x=f3_Ri_data[0],
    y=f3_Ri_data[1],
    s=0
)


def f3_Ri(Ri_new):
    # Do not extrapolate
    return splev(Ri_new, f3_Ri_tck, ext=3)


def CT2_flux(*, T_st, z, L):
    """ Flux-based Wyngaard et al (1971) with updated constants by Andreas (1989)"""
    # DO NOT clip L in computation. Should be done outside.
    # L = np.clip(L, a_min=np.percentile(L, 2.5), a_max=np.percentile(L, 97.5))
    zeta = z / L
    zeta_neg = np.where(zeta < 0, zeta, np.nan)  # unstable
    zeta_pos = np.where(zeta >= 0, zeta, np.nan)  # stable
    # from numpy docs: Where True, yield x, otherwise yield y.
    g3 = np.where(
        zeta < 0,  # unstable
        4.9 * np.power(1 - 6.1 * zeta_neg, -2 / 3),  # unstable
        4.9 * (1 + 2.2 * np.power(zeta_pos, 2 / 3)),  # stable
    )

    try:
        # If zeta is an xarray DataArray, convert g to xarray DataArray for proper broadcasting
        import xarray as xr
        if isinstance(zeta, xr.DataArray):
            g3 = xr.DataArray(data=g3, dims=zeta.dims, coords=zeta.coords)
    except ImportError:
        pass

    return T_st ** 2 * z ** (-2 / 3) * g3


def CT2_grad(*, Ri, z, Gamma):
    """ Eq 9 from Wyngaard 1971 """
    try:
        # Spline interpolation converts data to numpy array.
        # Convert back to xarray if necessary for proper broadcasting.
        import xarray as xr
        if isinstance(Ri, xr.DataArray):
            f3 = xr.DataArray(data=f3_Ri(Ri), dims=Ri.dims, coords=Ri.coords)
        else:
            f3 = f3_Ri(Ri)
    except ImportError:
        f3 = f3_Ri(Ri)

    return np.power(z, 4 / 3) * Gamma ** 2 * f3


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    ri = np.linspace(-3, .5)
    plt.plot(
        ri,
        f3_Ri(ri)
    )
    plt.scatter(
        f3_Ri_data[0],
        f3_Ri_data[1]
    )
    plt.yscale("log")
    plt.show()
