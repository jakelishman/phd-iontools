"""
Provides the `Sequence` class, though this is actually exposed as part of the
top-level package.  For help, see the documentation of that class, or the
top-level package.
"""

import numpy as np
import qutip
import functools
from . import Sideband

__all__ = ["Sequence"]

def no_derivatives(*args, **kwargs):
    raise NotImplementedError("Derivatives are not enabled.")

class _DummySideband(Sideband):
    def __init__(self):
        self.ns = 1
        self.order = 0
        self.__base = qutip.tensor(qutip.qeye(2), qutip.qeye(2))
        self.__d_base = self.__base - self.__base # 0 op of same size
        self.u = lambda *args: self.__base
        self.du_dt = lambda *args: self.__d_base
        self.du_dphi = lambda *args: self.__d_base

def _interleave(args_before):
    """
    If the arguments to a function are supplied as (time, phase), interleave
    them into a single `params` array and pass them on.

    `args_before` is the number of arguments before the `time`/`params`
    argument, and then the parameters specifiers should be the last argument (in
    the case of `params`), or the last two arguments (in the case of `times`,
    `phases`).
    """
    def ret(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if len(args) == args_before + 2:
                return func(*args)
            else:
                phases = args[args_before + 2]
                params = np.empty((2 * len(args[args_before + 1]),),
                                  dtype=np.float64)
                params[0::2] = args[args_before + 1]
                params[1::2] = phases
                return func(*args[:args_before + 1], params, **kwargs)
        return wrapper
    return ret

class Sequence:
    """
    This class provides an interface for working with a sequence of individual
    `Sideband`s.  It can efficiently calculate the effective operator of the
    whole pulse sequence using the `Sequence.op()` method, and the derivatives
    are available (if they were not explicitly disabled) in the
    `Sequence.d_op()` method.

    The internals of this class are not particularly enlightening, since they
    are optimised numerics to calculate the operator and the derivatives in a
    manner linear in the number of pulses.

    Members:
    ns: int >= 0 -- The number of motional states considered.

    pulses: np.array of `Sideband` --
        The `Sideband`s that make up the sequence.  `pulses[0]` will be the
        first pulse that is applied to any given state.

    motional_change: int >= 0 --
        The maximum possible motional change that can be brought about by the
        sequence.  To be safe, you should have enough motional states under
        consideration that there will never be "leakage" out of the top state.
    """

    def __init__(self, *pulses, derivatives=True):
        """
        Sequence(pulse0, pulse1, ...) == Sequence([pulse0, pulse1, ...])

        Arguments:
        pulses: sideband.Sideband --
            The pulse sequence to apply.  They should be arranged in the same
            order as they are applied, so the first pulse in the argument list
            is the first pulse applied to the system.
        derivatives (kw): bool --
            Whether to calculate derivatives of the sideband with each update.
        """
        if len(pulses) is 1 and not isinstance(pulses[0], qutip.Qobj):
            try:
                pulses = list(pulses[0])
            except TypeError:
                pass
        if len(pulses) is 0:
            # insert dummy sideband to return identity matrix
            pulses = [_DummySideband()]
        if any(map(lambda pulse: pulses[0].ns - pulse.ns, pulses)):
            raise ValueError("Not all pulses have the same number of 'ns'.")
        self.ns = pulses[0].ns
        self.pulses = np.array(pulses)
        self.motional_change = sum(map(lambda x: abs(x.order), pulses))
        self.__last_params = None
        self.__op = None
        self.derivatives = derivatives
        self.__force_update = False
        # Perform branching only once at instantiation, rather than at each call
        # to any function.
        if derivatives:
            self.__d_op = np.empty(2 * len(pulses), dtype=qutip.Qobj)
            self.__cache = np.empty(len(pulses), dtype=qutip.Qobj)
            self.__updater = self.__update_all
        else:
            self.d_op = no_derivatives
            self.__updater = self.__update_only_u

    def __repr__(self):
        if len(self.pulses) is 1 and isinstance(self.pulses[0], _DummySideband):
            npulses = 0
            orders = []
        else:
            npulses = len(self.pulses)
            orders = [x.order for x in self.pulses]
        plural = "" if npulses is 1 else "s"
        return "\n".join([
            f"{self.__class__.__name__} containing {npulses} pulse{plural}.",
            f"  orders      = {orders}",
            f"  ns          = {self.ns}",
            f"  derivatives = {self.derivatives}",
        ])

    @classmethod
    def from_orders(cls, orders, lamb_dicke, base_rabi,
                    detuning=0.0, ns=None, derivatives=True):
        """
        Sequence.from_orders(orders, laser) -> Sequence

        Create a `Sequence` class from a specified list of orders, rather than
        from pre-created `Sideband` classes.  This is just a convenience method.

        Arguments:
        orders: array-like of int --
            The orders of the pulses to apply, where the ordering of the list is
            the same as the ordering of how the pulses would be applied, i.e.
            the first element of the list is the first pulse applied.

        lamb_dicke: float --
            The Lamb-Dicke parameter to use for each of the sidebands.

        base_rabi: float in Hz --
            The base Rabi frequency to use for each of the sidebands.

        detuning (kw): float in Hz --
            The detuning from resonance of each of the sidebands.

        ns (kw): int > 0 --
            The number of motional states to consider for each pulse.  If this
            is not specified, then the minimum number required to safely apply
            the pulse sequence to (|g> + |e>)|0> will be used.

        derivatives (ks): bool --
            Whether to calculate derivatives of the pulse sequence.
        """
        if ns is None:
            ns = 1 + sum(np.abs(orders))
        pulses = [Sideband(ns, order, lamb_dicke, base_rabi, detuning)
                  for order in orders]
        return cls(pulses, derivatives=derivatives)

    def set_sideband_parameters(self, sideband: int, detuning=None,
                                lamb_dicke=None, base_rabi=None):
        """
        Update the sideband at position `sideband` to have the new parameters
        specified.  This will force an update to `op()` and `d_op()` on the next
        call to either of them, even if the parameters passed are the same.
        """
        self.pulses[sideband].update_multiple_parameters(detuning,
                                                         lamb_dicke,
                                                         base_rabi)
        self.__force_update = True

    def set_all_sideband_parameters(self, detuning=None, lamb_dicke=None,
                                    base_rabi=None):
        """
        Update the sideband parameters for all of the pulses in the sequence.
        This will force `op()` and `d_op()` to update on the next call to either
        of them, even if the parameters passed are the same.
        """
        for i in range(len(self.pulses)):
            self.set_sideband_parameters(i, detuning, lamb_dicke, base_rabi)

    def with_ns(self, ns):
        """with_ns(ns: int) -> Sequence

        Return a new `Sequence` object with the same properties, but considering
        a different number of motional levels."""
        return type(self)([pulse.with_ns(ns) for pulse in self.pulses])

    def __update_if_required(self, params):
        """Update all the necessary operators, if the parameters have
        changed."""
        if (not self.__force_update)\
           and np.array_equal(params, self.__last_params):
            return
        self.__force_update = False
        self.__updater(params)

    def __update_all(self, params):
        """Update the operators, but only if the parameters passed differ from
        the last calculation."""
        self.__op = qutip.tensor(qutip.qeye(2), qutip.qeye(self.ns))
        cur = self.__op
        self.__d_op[:] = self.__op
        for i in range(len(self.pulses) - 1):
            (t, phi) = params[2*i : 2*i + 2]
            self.__cache[i] = self.pulses[i].u(t, phi)
            self.__op = self.__cache[i] * self.__op
            self.__d_op[2*i] = self.pulses[i].du_dt(t, phi) * self.__d_op[2*i]
            self.__d_op[2*i + 1] = self.pulses[i].du_dphi(t, phi)\
                                   * self.__d_op[2 * i + 1]
            self.__d_op[2*i + 2] = self.__op
            self.__d_op[2*i + 3] = self.__op
        (t, phi) = params[-2:]
        self.__cache[-1] = self.pulses[-1].u(t, phi)
        self.__op = self.__cache[-1] * self.__op
        self.__d_op[-2] = self.pulses[-1].du_dt(t, phi) * self.__d_op[-2]
        self.__d_op[-1] = self.pulses[-1].du_dphi(t, phi) * self.__d_op[-1]
        for i in range(len(self.pulses) - 1, 0, -1):
            (t, phi) = params[2*i : 2*i + 2]
            cur = cur * self.__cache[i]
            self.__d_op[2 * i - 2] = cur * self.__d_op[2 * i - 2]
            self.__d_op[2 * i - 1] = cur * self.__d_op[2 * i - 1]
        self.__last_params = np.array(params)

    def __update_only_u(self, params):
        """Update only the `u` operator, and don't calculate the derivatives."""
        self.__op = qutip.tensor(qutip.qeye(2), qutip.qeye(self.ns))
        for i, pulse in enumerate(self.pulses):
            self.__op = pulse.u(*params[2*i : 2*i + 2]) * self.__op
        self.__last_params = np.array(params)

    @_interleave(0)
    def op(self, params, force=False):
        """
        op(params) -> matrix: qutip.Qobj (operator)
        op(times, phases) -> matrix: qutip.Qobj (operator)

        Return the operator matrix for the sequence for the pulse lengths and
        phases specified by `params`.  This can be called as either of the two
        ways specified (for convenience).

        Arguments:
        params: np.array of alternating `time`, `phase`
        where
            time: float in s > 0,
            phase: float in rad on (-pi, pi] --

            A sequence of the parameters for each of the pulses in the sequence.
            These should be given in the same order as the pulses were given to
            instantiate the class.

        times: np.array of float in s > 0 --
            A sequence of the times for each of the pulses in the sequence.
            These should be given in the same order as the pulses were given to
            instantiate the class.

        phases: np.array of float on (-pi, pi] --
            A sequence of the phases for each of the pulses in the sequence.
            These should be given in the same order as the pulses were given to
            instantiate the class.

        Returns:
        2D np.array of complex --
            The matrix form of the entire evolution due to the pulse
            sequence."""
        self.__force_update = self.__force_update or force
        self.__update_if_required(params)
        return self.__op

    @_interleave(0)
    def d_op(self, params, force=False):
        """
        d_op(params) -> matrices: 1D np.array of qutip.Qobj (operator)
        d_op(times, phases) -> matrices: 1D np.array of qutip.Qobj (operator)

        Returns an array of matrices corresponding the derivatives of the
        whole sequence matrix with respect to each of the parameters in turn.

        Arguments:
        params: np.array of alternating `time`, `phase`
        where
            time: float in s > 0,
            phase: float in rad on (-pi, pi] --

            A sequence of the parameters for each of the pulses in the sequence.
            These should be given in the same order as the pulses were given to
            instantiate the class.

        times: np.array of float in s > 0 --
            A sequence of the times for each of the pulses in the sequence.
            These should be given in the same order as the pulses were given to
            instantiate the class.

        phases: np.array of float on (-pi, pi] --
            A sequence of the phases for each of the pulses in the sequence.
            These should be given in the same order as the pulses were given to
            instantiate the class.

        Returns:
        3D np.array of complex (as a stack of 2D matrices) --
            Each of the derivatives of the full pulse sequence with respect to
            the parameters in the `params` argument.  The derivatives will
            always be ordered as `time[0]`, `phase[0]`, `time[1]`, etc,
            regardless of which form the parameters were passed in.
        """
        self.__force_update = self.__force_update or force
        self.__update_if_required(params)
        return self.__d_op

    @_interleave(1)
    def trace(self, state: qutip.Qobj, params):
        """
        trace(state: qutip.Qobj, params: np.array) -> generator of qutip.Qobj
        trace(state: qutip.Qobj, times, phases) -> generator of qutip.Qobj

        Return a generator which yields the state of the system after each of
        the pulses in the sequence have been applied, including the initial and
        end states.
        """
        yield state
        for i, pulse in enumerate(self.pulses):
            state = pulse.u(*params[2*i:2*i+2]) * state
            yield state
