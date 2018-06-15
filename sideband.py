"""
Provides the Sideband class for calculation of time-evolution operators for
laser sidebands and their derivatives.  This should typically be accessed via
the top-level package, so as `iontools.Sideband`, rather than needing to come
into this module.
"""

__all__ = ["Sideband"]

import numpy as np
import qutip
from . import rabi as _rabi

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

# 2-level system projectors of |e><e| and |g><g|.  They are created like this to
# ensure that the state is in the correct order to match the sigma_z operator.
_proj_ee = (qutip.qeye(2) + qutip.sigmaz()) / 2
_proj_gg = (qutip.qeye(2) - qutip.sigmaz()) / 2
_proj_eg = qutip.sigmap()
_proj_ge = qutip.sigmam()

class Sideband:
    """
    Calculate time-evolution operators and their derivatives for single-ion
    operations on a particular (arbitrary) sideband.  Off-resonant excitations
    are ignored, which requires that the trap frequency is much larger than the
    base Rabi frequency, which is in turn much larger than the detuning of the
    laser.

    The main method is `Sideband.u()`, which calculates the time-evolution
    operator for a specified time and phase.  The `Sideband.du_dt()` and
    `Sideband.du_dphi()` methods are available for calculating the derivatives
    as well.

    The internals of this class are messy and not very enlightening due to the
    precalculation of various factors for speed reasons.

    Members:
    ns: int >= 0 -- The number of motional states being considered.
    order: int -- The order of the sideband.  `order` < 0 => red, etc.
    detuning: float in rad -- The detuning of the laser being used.
    """

    # Largely the aim is to precalculate as much as possible so that subsequent
    # calculations of the matrices are very fast.  The creation of this class
    # ought to be linear in `ns`, as should all matrix-returning functions.

    def __init__(self, ns, order, lamb_dicke, base_rabi, detuning=0.0):
        """
        Arguments:
        ns: int > 0 -- The number of motional states to consider.
        order: int --
            The order of the sideband this operator should represent.  For
            example, 0 -> carrier, -1 -> first red, 1 -> first blue, -2 ->
            second red, and so on.
        lamb_dicke: float -- The Lamb-Dicke parameter of the laser used.
        base_rabi: float in Hz -- The base Rabi frequency of the laser used.
        detuning: float in Hz -- The detuning of the laser from transition.
        """
        self.ns = ns
        self.order = order
        self.__lamb_dicke = lamb_dicke
        self.__base_rabi = base_rabi
        self.__detuning = detuning
        if order > 0:
            self.__const_ind = (0, abs(order))
            self.__ee_ind = (abs(order), ns)
            self.__gg_ind = (ns, 2 * ns)
        else:
            self.__const_ind = (ns, ns + abs(order))
            self.__ee_ind = (0, ns)
            self.__gg_ind = (ns + abs(order), 2 * ns)
        self.__off_diag_len = self.ns - abs(order)
        self.__update_prefactors()
        self.__sin = np.empty_like(self.__rabi_mod)
        self.__cos = np.empty_like(self.__rabi_mod)
        self.__diag = np.empty(2 * self.ns, dtype=np.complex128)
        self.__phase_time = 1.0
        self.__phase_phi = 1.0
        self.__phase_tot = 1.0
        self.__last_params = (None, None)

    def __repr__(self):
        if self.order == 0:
            desc = "carrier"
        else:
            desc = ("red" if self.order < 0 else "blue") + f" {abs(self.order)}"
        return "\n".join([
            f"{self.__class__.__name__} of order {self.order} ({desc})",
            f"  order      = {self.order}",
            f"  ns         = {self.ns}",
            f"  detuning   = {self.detuning}",
            f"  Lamb-Dicke = {self.lamb_dicke}",
            f"  base Rabi  = {self.base_rabi}",
        ])

    def __update_prefactors(self):
        self.__rabi = self.__base_rabi\
            * _rabi.relative_rabi_range(self.__lamb_dicke,
                                        0, self.ns,
                                        abs(self.order))
        self.__rabi_mod = _rabi.rabi_mod_from_rabi(self.__detuning, self.__rabi)
        self.__eg_pre = -1.0j * np.exp(0.5j * np.pi * abs(self.order))\
                        * self.__rabi[:self.__off_diag_len]\
                        / self.__rabi_mod[:self.__off_diag_len]
        self.__ee_du_dt_pre = -0.5 * self.__rabi**2 / self.__rabi_mod
        self.__force_update = True

    @property
    def detuning(self):
        return self.__detuning
    @detuning.setter
    def detuning(self, detuning):
        self.__detuning = detuning
        self.__update_prefactors()

    @property
    def lamb_dicke(self):
        return self.__lamb_dicke
    @lamb_dicke.setter
    def lamb_dicke(self, lamb_dicke):
        self.__lamb_dicke = lamb_dicke
        self.__update_prefactors()

    @property
    def base_rabi(self):
        return self.__base_rabi
    @base_rabi.setter
    def base_rabi(self, base_rabi):
        self.__base_rabi = base_rabi
        self.__update_prefactors()

    def update_multiple_parameters(self, detuning=None, lamb_dicke=None,
                                   base_rabi=None):
        """
        Update multiple parameters at once.   This is much more efficient than
        updating an individual parameter three times over, because each
        parameter update causes the internal prefactors to be recomputed.  Using
        this function allows us to only do that once.

        Arguments:
        detuning (kw): float in Hz -- The new detuning from resonance to use.
        lamb_dicke (kw): float -- The new Lamb--Dicke parameter to use.
        base_rabi (kw): float in Hz -- The new base Rabi frequency to use.
        """
        if detuning is not None:
            self.__detuning = detuning
        if lamb_dicke is not None:
            self.__lamb_dicke = lamb_dicke
        if base_rabi is not None:
            self.__base_rabi = base_rabi
        if detuning is None and lamb_dicke is None and base_rabi is None:
            return
        self.__update_prefactors()

    def with_ns(self, ns):
        """with_ns(ns: int) -> Sideband

        Return a new `Sideband` object with the same properties, but considering
        a different range of motional states."""
        return type(self)(ns, self.order, self.lamb_dicke, self.base_rabi,
                          detuning=self.detuning)

    def __update_if_required(self, time, phase):
        """Update the common elements for use in the matrix-returning
        functions."""
        if (not self.__force_update) and (time, phase) == self.__last_params:
            return
        self.__last_params = (time, phase)
        self.__sin = np.sin(self.__rabi_mod * 0.5 * time)
        self.__cos = np.cos(self.__rabi_mod * 0.5 * time)
        self.__phase_time = np.exp(-0.5j * self.detuning * time)
        self.__phase_phi = np.exp(-1j * phase)
        self.__phase_tot = self.__phase_time * self.__phase_phi
        self.__force_update = False

    def u(self, time: float, phase: float) -> qutip.Qobj:
        """u(time: float in s, phase: float in rad) -> operator

        Get the matrix form of the time-evolution operator corresponding to this
        transition.  The result is ordered such that it should be applied to a
        vector [|e0>, |e1>, ..., |g0>, |g1>, ...]."""
        self.__update_if_required(time, phase)
        ee = self.__cos + 1j * self.detuning * self.__sin / self.__rabi_mod
        ee = ee * self.__phase_time
        self.__diag[self.__const_ind[0]:self.__const_ind[1]] = 1.0
        self.__diag[self.__ee_ind[0]:self.__ee_ind[1]] =\
            ee[:self.__ee_ind[1] - self.__ee_ind[0]]
        self.__diag[self.__gg_ind[0]:self.__gg_ind[1]] =\
            np.conj(ee[:self.__gg_ind[1] - self.__gg_ind[0]])
        eg = self.__sin[:self.__off_diag_len] * self.__eg_pre * self.__phase_tot
        ge = -np.conj(eg)
        return   qutip.tensor(_proj_ee, qutip.qdiags(self.__diag[:self.ns], 0))\
               + qutip.tensor(_proj_gg, qutip.qdiags(self.__diag[self.ns:], 0))\
               + qutip.tensor(_proj_eg, qutip.qdiags(eg, -self.order))\
               + qutip.tensor(_proj_ge, qutip.qdiags(ge, self.order))

    def du_dt(self, time: float, phase: float) -> qutip.Qobj:
        """du_dt(time : float in s, phase : float in rad) -> operator

        Get the matrix form of the partial derivative of the time-evolution
        operator with respect to time."""
        self.__update_if_required(time, phase)
        ee = self.__ee_du_dt_pre * self.__phase_time * self.__sin
        eg = 0.5 * self.__eg_pre * self.__phase_tot * (\
                self.__rabi_mod * self.__cos - 1j * self.detuning * self.__sin\
             )[:self.__off_diag_len]
        self.__diag[self.__const_ind[0]:self.__const_ind[1]] = 0.0
        self.__diag[self.__ee_ind[0]:self.__ee_ind[1]] =\
            ee[:self.__ee_ind[1] - self.__ee_ind[0]]
        self.__diag[self.__gg_ind[0]:self.__gg_ind[1]] =\
            np.conj(ee[:self.__gg_ind[1] - self.__gg_ind[0]])
        return qutip.tensor(_proj_ee, qutip.qdiags(self.__diag[0:self.ns], 0))\
               + qutip.tensor(_proj_gg, qutip.qdiags(self.__diag[self.ns:], 0))\
               + qutip.tensor(_proj_eg, qutip.qdiags(eg, -self.order))\
               + qutip.tensor(_proj_ge, qutip.qdiags(-np.conj(eg), self.order))

    def du_dphi(self, time: float, phase: float) -> qutip.Qobj:
        """du_dphi(time: float in s, phase: float in rad) -> operator

        Get the matrix form of the partial derivative of the time-evolution
        operator with respect to phase."""
        self.__update_if_required(time, phase)
        eg = -1j*self.__eg_pre*self.__sin[:self.__off_diag_len]*self.__phase_tot
        return qutip.tensor(_proj_eg, qutip.qdiags(eg, -self.order))\
               + qutip.tensor(_proj_ge, qutip.qdiags(-np.conj(eg), self.order))
