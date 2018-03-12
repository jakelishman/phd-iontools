import numpy as np

class Sequence:
    def __init__(self, *pulses):
        """Arguments:
        pulses: sideband.Sideband --
            The pulse sequence to apply.  They should be arranged in the same
            order as they are applied, so the first pulse in the argument list
            is the first pulse applied to the system."""
        self.pulses = pulses
        self.__last_params = None
        self.__op = None
        self.__d_op = np.empty((2*len(pulses), 2*pulses[0].ns, 2*pulses[0].ns),
                               dtype=np.complex128)

    def __update_if_required(self, params):
        """Update the operators, but only if the parameters passed differ from
        the last calculation."""
        if np.array_equal(params, self.__last_params):
            return
        self.__op = np.eye(2 * self.pulses[0].ns, dtype=np.complex128)
        self.__d_op[:] = self.__op
        for i in range(len(self.pulses) - 1):
            (t, phi) = params[2*i : 2*i + 2]
            np.dot(self.pulses[i].u(t, phi), self.__op, out=self.__op)
            self.__d_op[2 * i] = np.dot(self.pulses[i].du_dt(t, phi),
                                        self.__d_op[2 * i])
            self.__d_op[2 * i + 1] = np.dot(self.pulses[i].du_dphi(t, phi),
                                            self.__d_op[2 * i + 1])
            self.__d_op[2 * i + 2] = self.__op.copy()
            self.__d_op[2 * i + 3] = self.__op.copy()
        (t, phi) = params[-2:]
        np.dot(self.pulses[-1].u(t, phi), self.__op, out=self.__op)
        self.__d_op[-2] = np.dot(self.pulses[-1].du_dt(t, phi), self.__d_op[-2])
        self.__d_op[-1] = np.dot(self.pulses[-1].du_dphi(t, phi),
                                 self.__d_op[-1])
        # This somewhat relies on the caching behaviour of the Sideband
        # class---we could store the values of the matrices the first time we
        # computed them, but there should be no need since the class does it if
        # we don't change the parameters inbetween.
        cur = np.eye(2 * self.pulses[0].ns, dtype=np.complex128)
        for i in range(len(self.pulses) - 1, 0, -1):
            (t, phi) = params[2*i : 2*i + 2]
            np.dot(cur, self.pulses[i].u(t, phi), out=cur)
            self.__d_op[2 * i - 2] = np.dot(cur, self.__d_op[2 * i - 2])
            self.__d_op[2 * i - 1] = np.dot(cur, self.__d_op[2 * i - 1])
        self.__last_params = params

    def op(self, params):
        """Return the operator matrix for the sequence for the pulse lengths and
        phases specified by `params`.

        Arguments:
        params: np.array of alternating `time`, `phase`
        where
            time: float in s > 0,
            phase: float in rad on (-pi, pi] --

            A sequence of the parameters for each of the pulses in the sequence.
            These should be given in the same order as the pulses were given to
            instantiate the class.

        Returns:
        2D np.array of complex --
            The matrix form of the entire evolution due to the pulse
            sequence."""
        self.__update_if_required(params)
        return np.copy(self.__op)

    def d_op(self, params):
        """Returns an array of matrices corresponding the derivatives of the
        whole sequence matrix with respect to each of the parameters in turn.

        Arguments:
        params: np.array of alternating `time`, `phase`
        where
            time: float in s > 0,
            phase: float in rad on (-pi, pi] --

            A sequence of the parameters for each of the pulses in the sequence.
            These should be given in the same order as the pulses were given to
            instantiate the class.

        Returns:
        3D np.array of complex (as a stack of 2D matrices) --
            Each of the derivatives of the full pulse sequence with respect to
            the parameters in the `params` argument.  They will be in the same
            order as how `params` was passed."""
        self.__update_if_required(params)
        return np.copy(self.__d_op)
