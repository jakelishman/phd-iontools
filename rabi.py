"""
Provides functions for calculting the Rabi frequencies of transitions.  The
default single call is to
    relative_rabi(lamb_dicke, n1, n2)
which returns the strength of the `n1 <-> n2` transition relative to the base
Rabi frequency.  To get the concrete frequency, simply multiply by the base Rabi
frequency.  The modified Rabi frequency is also available at `rabi_mod()`, and
the underlying Laguerre polynomials can be accessed with `laguerre()`.

You can convert between Rabi frequencies and modified Rabi frequencies using the
`rabi_mod_from_rabi()` function, but beware that you need to have a concrete
Rabi frequency, not a relative value.

If you need to calculate a lot of related Rabi frequencies, you should use the
`relative_rabi_range()` and `rabi_mod_range()` functions which operate in linear
time (the naive implementation would be quadratic in the maximum `n` value
used).  These use an underlying recurrence relation in the Laguerre polynomials,
which is also available at `laguerre_range()`.
"""

__all__ = ["laguerre", "laguerre_range", "rabi_mod_from_rabi",
           "relative_rabi", "relative_rabi_range", "rabi_mod", "rabi_mod_range"]

import numpy as np

def laguerre(n: int, a: float, x: float) -> float:
    """laguerre(n : int >= 0, a : float, x : float) -> res : float

    Calculate the Laguerre polynomial result L_n^a(x), which is equivalent to
    Mathematica's LaguerreL[n, a, x].
    """
    if n == 0:
        return 1
    elif n == 1:
        return 1 + a - x
    # use a recurrence relation calculation for speed and accuracy
    # ref: http://functions.wolfram.com/Polynomials/LaguerreL3/17/01/01/01/
    l_2, l_1 = 1, 1 + a - x
    for m in range(2, n + 1):
        l_2, l_1 = l_1, ((a + 2*m - x - 1) * l_1 - (a + m - 1) * l_2) / m
    return l_1

def laguerre_range(n_start: int, n_end: int, a: float, x: float) -> np.ndarray:
    """laguerre_range(n_start, n_end, a, x) -> np.array(dtype=np.float64)

    Use the recurrence relation for nearest-neighbour in n of the Laguerre
    polynomials to calculate
        [laguerre(n_start, a, x),
         laguerre(n_start + 1, a, x),
         ...,
         laguerre(n_end - 1, a, x)]
    in linear time of `n_end` rather than quadratic.

    The time is linear in `n_end` not in the difference, because the initial
    calculation of `laguerre(n_start, a, x)` times linear time proportional to
    `n_start`, then each additional term takes another work unit.

    Reference: http://functions.wolfram.com/Polynomials/LaguerreL3/17/01/01/01/
    """
    if n_start >= n_end:
        return np.array([])
    elif n_start == n_end - 1:
        return np.array([laguerre(n_start, a, x)])
    out = np.empty((n_end - n_start, ), dtype=np.float64)
    out[0] = laguerre(n_start, a, x)
    out[1] = laguerre(n_start + 1, a, x)
    for n in range(2, n_end - n_start):
        out[n] = ((a + 2*n - x - 1) * out[n - 1] - (a + n - 1) * out[n - 2]) / n
    return out

def rabi_mod_from_rabi(detuning: float, rabis: float) -> float:
    """
    Convert a set of Rabi frequencies into modified Rabi frequencies.  The Rabi
    frequencies must be concrete values, not the relative values returned by the
    other functions in this module.

    Works on singletons or arrays.
    """
    if detuning == 0.0:
        return rabis
    else:
        return np.sqrt(detuning ** 2 + rabis ** 2)

def relative_rabi(lamb_dicke: float, n1: int, n2: int) -> float:
    """
    Get the relative Rabi frequency of a transition coupling motional levels
    `n1` and `n2` with a given Lamb--Dicke parameter.  The actual Rabi frequency
    will be the return value multiplied by the base Rabi frequency.
    """
    ldsq = lamb_dicke * lamb_dicke
    out = np.exp(-0.5 * ldsq) * (lamb_dicke ** abs(n1 - n2))
    out = out * laguerre(min(n1, n2), abs(n1 - n2), ldsq)
    fact = 1.0
    for n in range(1 + min(n1, n2), 1 + max(n1, n2)):
        fact = fact * n
    return out / np.sqrt(fact)

def relative_rabi_range(lamb_dicke: float, n_start: int, n_end: int, diff: int)\
        -> np.ndarray:
    """
    Get a range of Rabi frequencies in linear time of `n_end`.  The
    calculation of a single Rabi frequency is linear in `n`, so the naive
    version of a range is quadratic.  This method is functionally equivalent
    to
        np.array([rabi(n, n + diff) for n in range(n_start, n_end)])
    but runs in linear time."""
    if diff < 0:
        n_start = n_start + diff
        n_end = n_end + diff
        diff = -diff
    if n_start >= n_end:
        return np.array([])
    elif n_start == n_end - 1:
        return np.array([relative_rabi(lamb_dicke, n_start, n_start + diff)])
    ldsq = lamb_dicke * lamb_dicke
    const = np.exp(-0.5*ldsq) * lamb_dicke**diff
    lag = laguerre_range(n_start, n_end, diff, ldsq)
    fact = np.empty_like(lag)
    # the np.arange must contain a float so that the `.prod()` call doesn't
    # use fixed-length integers and overflow the factorial calculation.
    fact[0] = 1.0 / np.arange(n_start + 1.0, n_start + diff + 1).prod()
    for i in range(1, n_end - n_start):
        fact[i] = fact[i - 1] * (n_start + i) / (n_start + i + diff)
    return const * lag * np.sqrt(fact)

def rabi_mod(detuning: float, lamb_dicke: float, base_rabi: float,
             n1: int, n2: int) -> float:
    """
    Calculate the modified Rabi frequency coupling the two motional levels
    `n1` and `n2`.
    """
    return rabi_mod_from_rabi(detuning,
                              base_rabi * relative_rabi(lamb_dicke, n1, n2))

def rabi_mod_range(detuning: float, lamb_dicke: float, base_rabi: float,
                   n_start: int, n_end: int, diff: int)\
        -> np.ndarray:
    """
    Get a range of modified Rabi frequencies in linear time.  This is
    functionally equivalent to
        np.array([rabi_mod_range(n, n+diff) for n in range(n_start, n_end)])
    but runs in linear time.

    If you already have an array of the Rabi frequencies, use
    `rabi_mod_from_rabi` to avoid recalculating them.
    """
    return rabi_mod_from_rabi(
            detuning,
            base_rabi * relative_rabi_range(lamb_dicke, n_start, n_end, diff))
