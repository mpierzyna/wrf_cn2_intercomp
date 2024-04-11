from typing import Tuple

import numpy as np


def T_st(*, u_st, shfx_msK):
    """ Characterstic temperature in K """
    return - shfx_msK / u_st


def u_st(*, uw, vw):
    """ Friction velocity in m/s """
    return np.power(uw ** 2 + vw ** 2, 1 / 4)


def Ri_g(*, T_K, Gamma, S, g=9.81):
    """ Gradient Richardson number """
    return (g / T_K) * Gamma / S ** 2


def gladstone(*, CT2, P_hPa, T_K, a=7.9e-5, beta=None):
    """ Gladstone relation: CT2 -> Cn2 """
    Cn2 = (a * P_hPa / T_K ** 2) ** 2 * CT2
    if beta is not None:
        Cn2 *= (1 + 0.03 / beta) ** 2
    return Cn2


def gladstone_inverse(*, Cn2, P_hPa, T_K, a=7.9e-5, beta=None):
    """ Inverse Gladstone relation: Cn2 -> CT2 """
    CT2 = Cn2 / ((a * P_hPa / T_K ** 2) ** 2)
    if beta is not None:
        CT2 /= (1 + 0.03 / beta) ** 2

    return CT2


def S(*, dudz, dvdz):
    """ Mean wind shear in 1/s """
    return np.sqrt(dudz ** 2 + dvdz ** 2)


def theta_z(*, T_K, z, Gamma_d=9.8):
    """ Potential temperature based on height z and dry adiabatic lapse rate"""
    return T_K + Gamma_d * z


def theta_p(*, T_K, p, p_ref=1000, gamma=1.4):
    """ Potential temperature from T (Kelvin) at pressure level p
    with reference pressure p_ref (typically 1000 hPa per definition) """
    return T_K * np.power((p_ref / p), (gamma - 1) / gamma)


def hypsometric_p(*, T_K, z, T_ref_K, z_ref, p_ref, R_s=287., g=9.81):
    """ Compute pressure in height z at temperature T_K assuming exponential atmosphere.
    Reference values of temperature and pressure at level z_ref have to be known.
    """
    T_mean = (T_K + T_ref_K) / 2  # in K
    return p_ref * np.exp(
        - (z - z_ref) * g / R_s / T_mean
    )


def cn2_pdf_distance(cn2_obs: np.ndarray, cn2_mod: np.ndarray) -> Tuple[float, float, float, float, float]:
    """
    Compute Distance Metrics from Cn2 PDFs.
    From Sukanta, translated with ChatGPT

    Parameters
    ----------
    cn2_obs : np.ndarray
        Observed Cn2 values
    cn2_mod : np.ndarray
        Modeled Cn2 values

    Returns
    -------
    Tuple[float, float, float, float, float]
        e1 : float
            Shannon entropy of cn2_obs
        e2 : float
            Shannon entropy of cn2_mod
        d_k : float
            Kantorovich Distance between cn2_obs and cn2_mod
        d_ks : float
            Kolmogorov-Smirnov Statistic between cn2_obs and cn2_mod
        d_b : float
            Bhattacharyya Distance between cn2_obs and cn2_mod
    """
    # Reshape the arrays if they are 1D
    if cn2_obs.ndim == 1:
        cn2_obs = cn2_obs.reshape(-1, 1)
    if cn2_mod.ndim == 1:
        cn2_mod = cn2_mod.reshape(-1, 1)

    # Compute the logarithm
    obs = np.log10(cn2_obs)
    mod = np.log10(cn2_mod)

    # Define the edges of the histograms
    x = np.arange(-22, -12.5, 0.5)

    # Compute the histograms, probabilities and cumulative sums
    nn1 = np.histogram(obs, bins=x)[0]
    p1 = nn1 / nn1.sum()
    c1 = np.cumsum(p1)

    nn2 = np.histogram(mod, bins=x)[0]
    p2 = nn2 / nn2.sum()
    c2 = np.cumsum(p2)

    # Compute the Shannon entropy
    e1 = -np.nansum(p1 * np.log2(p1))
    e2 = -np.nansum(p2 * np.log2(p2))

    # Compute the Kantorovich Distance
    # d_k = kantorovich(p1, p2, 0)
    d_k = np.nan

    # Compute the Kolmogorov-Smirnov Statistic
    d_ks = np.max(np.abs(c1 - c2))

    # Compute the Bhattacharyya Distance
    d_b = -np.log(np.nansum(np.sqrt(p1 * p2)))

    return e1, e2, d_k, d_ks, d_b
