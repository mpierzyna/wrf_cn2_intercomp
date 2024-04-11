import numpy as np


def CT2_grad(*, Ri_g, z, Gamma):
    """ Gradient-based (Ri_g!) DNS regression model from He and Basu (2016).
    Only valid for Ri > 0. For Ri <= 0, use GRADIENT-BASED (!!) Wyngaard (1971).
    """
    g_T = 0.05 + 1.02 * np.exp(-14.49 * Ri_g)

    try:
        import xarray as xr
        if isinstance(g_T, xr.DataArray):
            # xarray
            g_T = xr.where(Ri_g <= 0, np.nan, g_T)
        else:
            # numpy
            g_T = np.where(Ri_g <= 0, np.nan, g_T)  # Expression only valid for Ri_g > 0
    except ImportError:
        # numpy
        g_T = np.where(Ri_g <= 0, np.nan, g_T)  # Expression only valid for Ri_g > 0

    return g_T * np.power(z, 4 / 3) * Gamma ** 2
