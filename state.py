import numpy as np

__all__ = ["State"]

class State:
    def __init__(self, data):
        self.ns = data.shape[0] // 2
        self.vec = data

    def __repr__(self):
        return "State vector with data:\n{}".format(self.vec.__repr__())

    @staticmethod
    def create(dict, ns=1):
        """create(dict, ns=1) -> State

        Create a normalised state which considers at least `ns` motional states,
        with the populations filled by the specifications in `dict`.

        Arguments:
        dict: dictionary of state_name * value
        with
            state_name: ("g" | "e") + string(int) --
                The identifier of the state to be filled, for example the string
                "g3" is the state in the ground state and the n=3 motional
                level.
            value: complex -- The relative value of that state.

        ns (optional): int --
            The minimum number of motional states to consider.  The resulting
            vector will always be at least twice the length of `ns`, but could
            be longer if there is a greater requirement in `dict`.

        Returns:
        np.array of complex --
            The state vector with all of the excited elements starting with n=0
            followed by all the ground elements.  The state vector is
            normalised.

        Examples:
        > create({"g0": 1.0})
        array([0.0, 1.0])

        > create({"e2": 1.0, "g3": 1.0j})
        array([0.0, 0.0, 0.707..., 0.0, 0.0, 0.0, 0.0, 0.707j])

        > create({"g1": 1.0}, ns=3)
        array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0])"""
        if not dict:
            raise TypeError("At least one populated state must be specified.")
        for (key, _) in dict.items():
            ns = max(ns, 1 + int(key[1:]))
        out = np.zeros(2 * ns, dtype=np.complex128)
        for (key, val) in dict.items():
            qubit = ns if key[0] is "g" else 0
            n = int(key[1:])
            out[qubit + n] = val
        return State(out / np.linalg.norm(out))
