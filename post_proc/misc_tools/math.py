import numpy as np
from scipy.interpolate import splrep, splev


def gradient_nonuniform(x: np.ndarray, y: np.ndarray, opt: int) -> np.ndarray:
    """
    Calculate the numerical derivative of y with respect to x, using three different methods.

    Parameters:
    x: numpy array of input x values
    y: numpy array of input y values
    opt : integer (1,2,3)
        1 => central difference with unequal spacing
        2 => spline
        3 => smoothing cubic spline

    Returns:
    The derivative evaluated at x[1:-1] positions.
    """
    if opt == 1:
        f1 = x[1:-1] - x[:-2]
        f2 = x[2:] - x[1:-1]
        a = f2 / f1
        num = y[2:] + (a ** 2 - 1) * y[1:-1] - a ** 2 * y[:-2]
        den = a * (a + 1) * f1
        return num / den
    elif opt == 2:
        pp = splrep(x, y)
        return splev(x[1:-1], pp, der=1)
    else:
        pp = splrep(x, y, s=0.0)
        return splev(x[1:-1], pp, der=1)
