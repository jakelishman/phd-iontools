"""Provides the Sideband class for calculation of time-evolution operators for
laser sidebands and their derivatives."""

__all__ = ["Sideband"]

import numpy as np

def ladder_indices(ns, change, quadrant):
    """Return an object which slices out the non-zero elements of a ladder
    operator from a 2D matrix of the Fock states coupled to a two-level system.
    For example,
        ladder_indices(3, 1, (0, 0)) -> [(1, 0), (2, 1)]
    corresponds to the creation operator on the space spanned by {|0>, |1>, |2>}
    and slices out the elements |1e><0e| and |2e><1e| in order.

    Arguments:
    ns: int > 0 -- The number of Fock states considered
    change: int -- The change in number (e.g. 1 -> creation, -1 -> annihilation)
    quadrant: int * int --
        The coordinate of the top-left value in the relevant quadrant, e.g.
            |e><g| has quadrant (0, ns)
            |g><e| has quadrant (ns, 0).

    Returns:
    numpy slice object -- An object which will slice out the correct indices out
    of a matrix of shape `(2 * ns, 2 * ns)`.

    Remarks:
    The result of this function can be converted for use with a coupled qubit
    space by doing (e.g.) `ladder_indices(ns, -1) + (0, ns)` to get the
    `sigma_+ * a` indices.  This is because of numpy's broadcasting rules cause
    the addition to apply to every set of indices in the object.
    """
    return tuple(np.array([(quadrant[0] + max(0, change) + n,
                            quadrant[1] + max(0, -change) + n)
                          for n in range(ns - abs(change))], dtype=np.intp).T)

def diagonal_indices(start, stop):
    """diagonal_indices(start : int, stop : int) -> ndarry slice object

    Get a slice object which returns the diagonal elements with indices in
    `range(start, stop)`.  For example, `diagonal_indices(2, 5)` slices out the
    indicies `[(2, 2), (3, 3), (4, 4)]`."""
    return (np.arange(start, stop), np.arange(start, stop))

class Sideband:
    """Calculate time-evolution operators and their derivatives for single-ion
    operations on a particular (arbitrary) sideband.

    All matrices are ordered to be correct when applied to a vector in the order
        [ |e0>, |e1>, ... |e(ns - 1)>, |g0>, |g1>, ... |g(ns-1)> ]."""

    # Largely the aim is to precalculate as much as possible so that subsequent
    # calculations of the matrices are very fast.  The creation of this class
    # ought to be linear in `ns`, as should all matrix-returning functions, if
    # we ignore the potentially quadratic dependency of matrix (c)allocation.

    def __init__(self, ns, order, laser):
        """Arguments:
        ns: int > 0 -- The number of motional states to consider.
        order: int --
            The order of the sideband this operator should represent.  For
            example, 0 -> carrier, -1 -> first red, 1 -> first blue, -2 ->
            second red, and so on.
        laser: laser.Laser -- The laser settings to use for this transition."""
        self.ns = ns
        self.order = order
        self.detuning = laser.detuning
        if order > 0:
            self.__const_indices = diagonal_indices(0, abs(order))
            self.__ee_indices = diagonal_indices(abs(order), ns)
            self.__gg_indices = diagonal_indices(ns, 2 * ns)
        else:
            self.__const_indices = diagonal_indices(ns, ns + abs(order))
            self.__ee_indices = diagonal_indices(0, ns)
            self.__gg_indices = diagonal_indices(ns + abs(order), 2 * ns)
        self.__eg_indices = ladder_indices(ns, order, (0, ns))
        self.__ge_indices = ladder_indices(ns, -order, (ns, 0))
        self.__rabi = laser.rabi_range(0, ns, abs(order))
        self.__rabi_mod = laser.rabi_mod_from_rabi(self.__rabi)
        self.__off_diag_len = self.ns - abs(order)
        self.__eg_pre = -1.0j * np.exp(0.5j * np.pi * abs(order))\
                        * self.__rabi[:self.__off_diag_len]\
                        / self.__rabi_mod[:self.__off_diag_len]
        self.__ee_du_dt_pre = -0.5 * self.__rabi**2 / self.__rabi_mod
        self.__sin = np.empty_like(self.__rabi_mod)
        self.__cos = np.empty_like(self.__rabi_mod)
        self.__phase_time = 1.0
        self.__phase_phi = 1.0
        self.__phase_tot = 1.0
        self.__last_params = (None, None)

    def __update_if_required(self, time, phase):
        """Update the common elements for use in the matrix-returning
        functions."""
        if (time, phase) == self.__last_params:
            return
        self.__last_params = (time, phase)
        self.__sin = np.sin(self.__rabi_mod * 0.5 * time)
        self.__cos = np.cos(self.__rabi_mod * 0.5 * time)
        self.__phase_time = np.exp(-0.5j * self.detuning * time)
        self.__phase_phi = np.exp(-1j * phase)
        self.__phase_tot = self.__phase_time * self.__phase_phi

    def u(self, time, phase):
        """u(time : float in s, phase : float in rad) -> 2D np.array of complex

        Get the matrix form of the time-evolution operator corresponding to this
        transition.  The result is ordered such that it should be applied to a
        vector [|e0>, |e1>, ..., |g0>, |g1>, ...]."""
        self.__update_if_required(time, phase)
        ee = self.__cos + 1j * self.detuning * self.__sin / self.__rabi_mod
        ee = ee * self.__phase_time
        out = np.zeros((2 * self.ns, 2 * self.ns), dtype=np.complex128)
        out[self.__const_indices] = 1.0
        out[self.__eg_indices] = self.__sin[:self.__off_diag_len]\
                                 * self.__eg_pre * self.__phase_tot
        out[self.__ge_indices] = -np.conj(out[self.__eg_indices])
        out[self.__ee_indices] = ee[:len(self.__ee_indices[0])]
        out[self.__gg_indices] = np.conj(ee[:len(self.__gg_indices[0])])
        return out

    def du_dt(self, time, phase):
        """du_dt(time : float in s, phase : float in rad)
        -> 2D np.array of complex

        Get the matrix form of the partial derivative of the time-evolution
        operator with respect to time."""
        self.__update_if_required(time, phase)
        ee = self.__ee_du_dt_pre * self.__phase_time * self.__sin
        eg = 0.5 * self.__eg_pre * self.__phase_tot * (\
                self.__rabi_mod * self.__cos - 1j * self.detuning * self.__sin\
             )[:self.__off_diag_len]
        out = np.zeros((2 * self.ns, 2 * self.ns), dtype=np.complex128)
        out[self.__ee_indices] = ee[:len(self.__ee_indices[0])]
        out[self.__gg_indices] = np.conj(ee[:len(self.__gg_indices[0])])
        out[self.__eg_indices] = eg
        out[self.__ge_indices] = -np.conj(eg)
        return out

    def du_dphi(self, time, phase):
        """du_dt(time : float in s, phase : float in rad)
        -> 2D np.array of complex

        Get the matrix form of the partial derivative of the time-evolution
        operator with respect to phase."""
        self.__update_if_required(time, phase)
        eg = -1j*self.__eg_pre*self.__sin[:self.__off_diag_len]*self.__phase_tot
        out = np.zeros((2 * self.ns, 2 * self.ns), dtype=np.complex128)
        out[self.__eg_indices] = eg
        out[self.__ge_indices] = -np.conj(eg)
        return out
