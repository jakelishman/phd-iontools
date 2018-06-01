import numpy as np
import qutip
from .  import Sideband

__all__ = ["Sequence"]

def no_derivatives(*args, **kwargs):
    raise NotImplementedError("Derivatives are not enabled.")

class Sequence:
    def __init__(self, *pulses, derivatives=True):
        """
        Sequence(pulse1, pulse2, ...) == Sequence([pulse1, pulse2, ...])

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
            raise ValueError("Must have at least one pulse.")
        if any(map(lambda pulse: pulses[0].ns - pulse.ns, pulses)):
            raise ValueError("Not all pulses have the same number of 'ns'.")
        self.ns = pulses[0].ns
        self.pulses = np.array(pulses)
        self.motional_change = sum(map(lambda x: abs(x.order), pulses))
        self.__last_params = None
        self.__op = None
        # Perform branching only once at instantiation, rather than at each call
        # to any function.
        if derivatives:
            self.__d_op = np.empty(2 * len(pulses), dtype=qutip.Qobj)
            self.__cache = np.empty(len(pulses), dtype=qutip.Qobj)
            self.__updater = self.__update_all
        else:
            self.d_op = no_derivatives
            self.__updater = self.__update_only_u

    def with_ns(self, ns):
        """with_ns(ns: int) -> Sequence

        Return a new `Sequence` object with the same properties, but considering
        a different number of motional levels."""
        return type(self)([pulse.with_ns(ns) for pulse in self.pulses])

    def __update_if_required(self, params):
        """Update all the necessary operators, if the parameters have
        changed."""
        if np.array_equal(params, self.__last_params):
            return
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

    def op(self, times, phases=None):
        """
        op(params) -> matrix: 2D numpy.array of complex
        op(times, phases) -> matrix: 2D numpy.array of complex

        Return the operator matrix for the sequence for the pulse lengths and
        phases specified by `params`.

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
        if phases is not None:
            params = np.empty((len(times) + len(phases)), dtype=np.float64)
            params[0::2] = times
            params[1::2] = phases
        else:
            params = times
        self.__update_if_required(params)
        return self.__op

    def d_op(self, times, phases=None):
        """
        d_op(params) -> matrices: 3D numpy.array of complex
        d_op(times, phases) -> matrices: 3D numpy.array of complex

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
        if phases is not None:
            params = np.empty((times.size + phases.size,), dtype=np.float64)
            params[0::2] = times
            params[1::2] = phases
        else:
            params = times
        self.__update_if_required(params)
        return self.__d_op

    def trace(self, params, state):
        """trace(params: np.array of float, state: qutip.Qobj)
        -> generator of qutip.Qobj

        Return a generator which yields the state of the system after each of
        the pulses in the sequence have been applied, including the initial and
        end states."""
        yield state
        for i, pulse in enumerate(self.pulses):
            state = pulse.u(*params[2*i:2*i+2]) * state
            yield state
