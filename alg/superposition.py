"""
Implementation of an analytic algorithm to evolve a system from |g0> into an
arbitrary superposition that can be expressed as a finite sum
    |x> = sum_n (c_gn |gn> + c_en |en>)
where the c are arbitrary complex coefficients, using only first-order
sidebands in the Lamb--Dicke regime.

Throughout, the target specifier in function definitions is a dictionary
defining the state to be populated, just like what would be passed to
`iontools.state.create()`.

The `single()` function is currently the only one implemented, which should be
sufficient to find a pulse sequence to target any reasonable superposition
inside the electronic decoherence time.

The superposition finder functions return objects of the type
`iontools.alg.superposition.Result`.  More help is available in the docstring of
that class (or instances of it).
"""

__all__ = ["Result", "single", "shortest", "all"]

from .. import Sideband as _Sideband
from .. import Sequence as _Sequence
from .. import state as _state
from .. import rabi as _rabi
import numpy as _np
import qutip as _qutip

_SI_TIME_UNITS = {1.0: "s", 1e-3: "ms", 1e-6: "us", 1e-12: "ns"}

def _bound(angle):
    angle = _np.fmod(angle, 2 * _np.pi)
    if angle <= -_np.pi:
        return angle + 2 * _np.pi
    elif angle > _np.pi:
        return angle - 2 * _np.pi
    else:
        return angle

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

    lamb_dicke: float -- The Lamb--Dicke parameter used.

    base_rabi: float in Hz -- The base Rabi frequency of the laser used.

    total_time: float in s -- The total time taken for the pulse sequence.

    sequence: iontools.Sequence --
        A Sequence class representing the pulse sequence, where the operator can
        be found setting arbitrary times and phases on the pulses.  Useful for
        investigating the fidelity under errors.

    op: 2D np.array of complex --
        An operator matrix representing the unitary transformation that is
        applied.
    """
    def __init__(self, times, phases, lamb_dicke, base_rabi, sidebands):
        self.times = times
        self.phases = _np.array([_bound(phase - phases[0]) for phase in phases])
        self.orders = _np.array([sideband.order for sideband in sidebands])
        self.lamb_dicke = lamb_dicke
        self.base_rabi = base_rabi
        self.total_time = sum(times)
        self.sequence = _Sequence(sidebands)
        if len(sidebands) is 0:
            self.op = _qutip.tensor(_qutip.qeye(2), _qutip.qeye(2))
        else:
            self.op = self.sequence.op(times, phases)

    def __repr__(self):
        ttime = f"{self.total_time}s"
        for lim, unit in _SI_TIME_UNITS.items():
            if self.total_time >= lim:
                rescale = self.total_time / lim
                ttime = f"{rescale:.3f}{unit}"
                break
        return "\n".join([
            f"Superposition sequence result, which would take {ttime}.",
            f"  times  = {repr(list(self.times))}",
            f"  phases = {repr(list(self.phases))}",
            f"  orders = {repr(list(self.orders))}",
        ])

def _is_populated(state: _qutip.Qobj, internal: str, n: int, tol: float) -> bool:
    """Is this particular internal and motional level populated?"""
    return _np.abs(_state.element(state, f"{internal}{n}")) > tol

def _both_populated(state: _qutip.Qobj, n: int, tol: float) -> bool:
    """Are both internal levels populated at this motional level?"""
    return _is_populated(state, "g", n, tol)\
           and _is_populated(state, "e", n, tol)

def _paired_states(state: _qutip.Qobj, cur_n: int, out_of: str, order: int):
    """Return a 2-tuple of the complex coefficients of the coupled ground and
    excited states (in that order) for the parameters specified."""
    if out_of is "g":
        return _state.element(state, [f"g{cur_n}", f"e{cur_n + order}"])
    else:
        return _state.element(state, [f"g{cur_n - order}", f"e{cur_n}"])

def single(target, lamb_dicke: float, base_rabi: float,
           orders=None, tol=1e-11) -> Result:
    """
    Find a single pulse sequence that will evolve the state |g0> into the target
    state specified in the parameters.  The phases returned are in the
    "theoretical" picture, where there is a separate laser beam for each driven
    transition, and they all start with zero phase.

    Arguments:
    target: dict | qutip.Qobj --
        The target state at the end of the pulse sequence.  Can either be
        specified as a list of populated elements as would be passed to
        `iontools.state.create()`, or a `qutip.Qobj` that has already been
        created.

    lamb_dicke: float --
        The Lamb--Dicke parameter of the laser used to create the pulses.

    base_rabi: float in Hz --
        The base Rabi frequency of the laser used to create the pulses.

    orders: optional list of int --
        The specific sidebands that should be used in order of how they would be
        applied.  It is a `ValueError` to pass an impossible list of orders.
        Each element is the order of the sideband that should be applied in that
        position.

        The list of orders passed can only be as long as the shortest possible
        sequence.  For example, to create the superposition
            {'g0': 1, 'g1': 1, 'g2': 1}
        the valid lists of orders are
            [c, {b, r}, c, {b, r}, c, r].

    tol: optional float --
        The tolerance to use for calculating if a state is unoccupied.

    Returns:
    iontools.alg.superposition.Result --
        The `Result` class that holds the results.

    Raises:
    ValueError --
        If the `orders` list is impossible to achieve.
    """
    if orders is not None:
        orders = orders[::-1]
    state: _qutip.Qobj = target if isinstance(target, _qutip.Qobj)\
                         else _state.create(target)
    cur_n = _state.max_populated_n(state)
    ns = cur_n + 1
    times, phases, sidebands = [], [], []
    rabi_carrier = base_rabi\
                   * _rabi.relative_rabi_range(lamb_dicke, 0, cur_n + 1, 0)
    rabi_red = base_rabi * _rabi.relative_rabi_range(lamb_dicke, 0, cur_n, 1)
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
                             + f"For pulse number {len(orders) - count} I was"
                             + f" asked to use {orders[count]}, but I expected"
                             + f" to see {order}.")
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
        sideband = _Sideband(ns, order, lamb_dicke, base_rabi)
        times.append(time)
        phases.append(phase)
        sidebands.append(sideband)
        state = sideband.u(time, phase) * state
        cur_n = cur_n - abs(order)
        count = count + 1
    times = -_np.array(times[::-1])
    phases = _np.array(phases[::-1])
    sidebands = _np.array(sidebands[::-1])
    return Result(times, phases, lamb_dicke, base_rabi, sidebands)
