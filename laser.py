"""Provides the Laser class to store settings for lasers operating sideband
transitions, and tools for calculating the relevant Rabi frequencies of
different transitions."""

import numpy as np

def laguerre(n, a, x):
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

def laguerre_range(n_start, n_end, a, x):
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

class Laser:
    """Stores information about the laser used in an experiment, and provides
    methods for calculating frequencies associated with transitions."""
    def __init__(self, detuning, lamb_dicke, base_rabi):
        """Arguments:
        detuning: float in Hz --
            The detuning of the laser from perfect transitions.
        lamb_dicke: float > 0 --
            The value of the Lamb-Dicke parameter for this laser interaction
            with an ion in the trap.
        base_rabi: float in Hz --
            The value of the base Rabi frequency of the laser transition.  This
            is used to calculate the actual Rabi frequencies of all transitions
            at all motional levels."""
        self.detuning = detuning
        self.lamb_dicke = lamb_dicke
        self.base_rabi = base_rabi

    def rabi_mod_from_rabi(self, rabis):
        """Convert a set of Rabi frequencies into modified Rabi frequencies.
        Works on singletons or arrays."""
        if self.detuning == 0.0:
            return rabis
        else:
            return np.sqrt(self.detuning ** 2 + rabis ** 2)

    def rabi(self, n1, n2):
        """rabi(n1 : int >= 0, n2 : int >= 0) -> float in Hz

        Get the Rabi frequency of a transition coupling motional levels `n1` and
        `n2`."""
        ldsq = self.lamb_dicke * self.lamb_dicke
        out = np.exp(-0.5 * ldsq) * (self.lamb_dicke ** abs(n1 - n2))
        out = out * laguerre(min(n1, n2), abs(n1 - n2), ldsq)
        fact = 1.0
        for n in range(1 + min(n1, n2), 1 + max(n1, n2)):
            fact = fact * n
        return self.base_rabi * out / np.sqrt(fact)

    def rabi_range(self, n_start, n_end, diff):
        """rabi_range(n_start : int >= 0, n_end >= 0, diff : int)
        -> np.array of float in Hz.

        Get a range of Rabi frequencies in linear time of `n_end`.  The
        calculation of a single Rabi frequency is linear in `n`, so the naive
        version of a range is quadratic.  This method is functionally equivalent
        to
            np.array([self.rabi(n, n + diff) for n in range(n_start, n_end)])
        but runs in linear time."""
        if diff < 0:
            n_start = n_start + diff
            n_end = n_end + diff
            diff = -diff
        if n_start >= n_end:
            return np.array([])
        elif n_start == n_end - 1:
            return np.array([self.rabi(n_start, n_start + diff)])
        ldsq = self.lamb_dicke * self.lamb_dicke
        const = np.exp(-0.5*ldsq) * self.lamb_dicke**diff * self.base_rabi
        lag = laguerre_range(n_start, n_end, diff, ldsq)
        fact = np.empty_like(lag)
        # the np.arange must contain a float so that the `.prod()` call doesn't
        # use fixed-length integers and overflow the factorial calculation.
        fact[0] = 1.0 / np.arange(n_start + 1.0, n_start + diff + 1).prod()
        for i in range(1, n_end - n_start):
            fact[i] = fact[i - 1] * (n_start + i) / (n_start + i + diff)
        return const * lag * np.sqrt(fact)

    def rabi_mod(self, n1, n2):
        """rabi_mod(n1 : int >= 0, n2 : int >= 0) -> float in Hz

        Calculate the modified Rabi frequency coupling the two motional levels
        `n1` and `n2`."""
        return self.rabi_mod_from_rabi(self.rabi(n1, n2))

    def rabi_mod_range(self, n_start, n_end, diff):
        """rabi_mod_range(n_start : int >= 0, n_end : int >= 0, diff : int)
        -> np.array of float in Hz

        Get a range of modified Rabi frequencies in linear time.  This is
        functionally equivalent to
            np.array([rabi_mod_range(n, n+diff) for n in range(n_start, n_end)])
        but runs in linear time.

        If you already have an array of the Rabi frequencies, use
        `rabi_mod_from_rabi` to avoid recalculating them."""
        return self.rabi_mod_from_rabi(self.rabi_range(n_start, n_end, diff))
