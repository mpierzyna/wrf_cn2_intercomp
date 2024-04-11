import numpy as np


def L(*, u_st, T_K, shfx_msK, g=9.81, k=.4):
    """ Obukhov length in m """
    return - u_st ** 3 * T_K / (k * g * shfx_msK)


def CT2(*, T_st, z, f_zeta):
    """ General MOST expression to obtain CT2 where f_zeta is the similarity function.
    In particular, f_zeta is f(zeta) = f(z/L) where L is the Obukhov length.
    See Savage (2009) for large collection of f functions.
    """
    return T_st ** 2 * np.power(z, -2 / 3) * f_zeta
