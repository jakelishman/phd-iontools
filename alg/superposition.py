"""
Implementation of an analytic algorithm to evolve a system from |g0> into an
arbitrary superposition that can be expressed as a finite sum
    |x> = sum_n (c_gn |gn> + c_en |en>)
where the c are arbitrary complex coefficients, using only first-order
sidebands in the Lamb--Dicke regime.

Throughout, the target specifier in function definitions is a dictionary
defining the state to be populated, just like what would be passed to
`iontools.state.create()`.
"""

__all__ = ["Result", "single", "shortest", "all"]

from .. import Sideband as _Sideband
from .. import Sequence as _Sequence
from .. import state as _state
import numpy as _np
import qutip as _qutip

class Result:
    """
    A class which holds the result of the superposition pulse creation
    algorithm.

    Members:
    times: np.array of float in s > 0 --
        The times each of the pulses should be applied for.

    phases: np.array of float in rad on (-pi, pi] --
        The phases each of the pulses should be applied at.  This is the phase
        shift relative to the original zero point of the transition - it does
        not include the effects on the phase that might have been caused by any
        previous pulse.

    orders: np.array of {-1, 0, 1} --
        The orders of the sidebands that should be applied.  The first order in
        the list is the first pulse that should be applied.

    laser: iontools.Laser -- The laser that was used to generate the times.

    total_time: float in s -- The total time taken for the pulse sequence.

    sequence: iontools.Sequence --
        A Sequence class representing the pulse sequence, where the operator can
        be found setting arbitrary times and phases on the pulses.  Useful for
        investigating the fidelity under errors.

    op: 2D np.array of complex --
        An operator matrix representing the unitary transformation that is
        applied.
    """
    def __init__(self, times, phases, laser, sidebands):
        self.times = times
        self.phases = phases
        self.orders = _np.array([sideband.order for sideband in sidebands])
        self.laser = laser
        self.total_time = sum(times)
        self.sequence = _Sequence(sidebands)
        if len(sidebands) is 0:
            self.op = _qutip.tensor(_qutip.qeye(2), _qutip.qeye(2))
        else:
            self.op = self.sequence.op(times, phases)

def _is_populated(state, internal, n, tol):
    return _np.abs(_state.element(state, f"{internal}{n}")) > tol

def _both_populated(state, n, tol):
    return _is_populated(state, "g", n, tol)\
           and _is_populated(state, "e", n, tol)

def _paired_states(state, cur_n, out_of, order):
    if out_of is "g":
        return _state.element(state, [f"g{cur_n}", f"e{cur_n + order}"])
    else:
        return _state.element(state, [f"g{cur_n - order}", f"e{cur_n}"])

def single(target, laser, orders=None, tol=1e-11):
    if orders is not None:
        orders = orders[::-1]
    state = target if isinstance(target, qutip.Qobj) else _state.create(target)
    cur_n = _state.max_populated_n(state)
    ns = cur_n + 1
    times, phases, sidebands = [], [], []
    rabi_carrier = laser.rabi_range(0, cur_n + 1, 0)
    rabi_red = laser.rabi_range(0, cur_n, 1)
    count = 0
    while cur_n > 0 or _is_populated(state, "e", 0, tol):
        if cur_n == 0:
            order = 0
            out_of = "e"
        elif _both_populated(state, cur_n, tol):
            order = 0
            if orders is None or orders[count + 1] == -1:
                out_of = "e"
            else:
                out_of = "g"
        elif _is_populated(state, "e", cur_n, tol):
            order = 1
            out_of = "e"
        else:
            order = -1
            out_of = "g"
        if orders is not None and count >= len(orders):
            raise ValueError("The chosen pulse sequence was not long enough.\n"
                             + str(orders))
        elif orders is not None and order != orders[count]:
            raise ValueError("Could not follow the desired pulse sequence.  "
                             + f"Was asked to use {orders[count]}, but had to "
                             + f"use {order} instead.")
        rabi = rabi_carrier[cur_n] if order == 0 else rabi_red[cur_n - 1]
        g, e = _paired_states(state, cur_n, out_of, order)
        # Make sure the time here is always negative.  Since we take the
        # absolute values of the coefficients, atan2 will always be positive, so
        # we just have to make sure we take the negative sin branch.
        if out_of == "g":
            phase = .5 * _np.pi * (abs(order) + 1) + _np.angle(g) - _np.angle(e)
            time = -2.0 * _np.arctan2(_np.abs(g), _np.abs(e)) / rabi
        else:
            phase = .5 * _np.pi * (abs(order) - 1) + _np.angle(g) - _np.angle(e)
            time = -2.0 * _np.arctan2(_np.abs(e), _np.abs(g)) / rabi
        sideband = _Sideband(ns, order, laser)
        times.append(time)
        phases.append(phase)
        sidebands.append(sideband)
        state = sideband.u(time, phase) * state
        cur_n = cur_n - abs(order)
        count = count + 1
    times = -_np.array(times[::-1])
    phases = _np.array(phases[::-1])
    sidebands = _np.array(sidebands[::-1])
    return Result(times, phases, laser, sidebands)
