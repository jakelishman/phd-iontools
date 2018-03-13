import numpy as np
import qutip
from qutip import Qobj

__all__ = ["create"]

def create(dict, ns=1):
    """create(dict, ns=1) -> qutip.Qobj

    Create a normalised state which considers at least `ns` motional states,
    with the populations filled by the specifications in `dict`.

    Arguments:
    dict: dictionary of state_name * value
    with
        state_name: ("g" | "e") + string(int) --
            The identifier of the state to be filled, for example the string
            "g3" is the state in the ground state and the n=3 motional level.
        value: complex -- The relative value of that state.

    ns (optional): int --
        The minimum number of motional states to consider.  The resulting vector
        will always be at least twice the length of `ns`, but could be longer if
        there is a greater requirement in `dict`.

    Returns:
    qutip.Qobj --
        The state vector with all of the excited elements starting with n=0
        followed by all the ground elements.  The state vector is normalised.

    Examples:
    > create({"g0": 1.0}).data
    array([0.0, 1.0])

    > create({"e2": 1.0, "g3": 1.0j}).data
    array([0.0, 0.0, 0.707..., 0.0, 0.0, 0.0, 0.0, 0.707j])

    > create({"g1": 1.0}, ns=3).data
    array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0])"""
    if not dict:
        raise TypeError("At least one populated state must be specified.")
    for (key, _) in dict.items():
        ns = max(ns, 1 + int(key[1:]))
    e = np.zeros(ns, dtype=np.complex128)
    g = np.zeros_like(e)
    for (key, val) in dict.items():
        qubit = g if key[0] is "g" else e
        qubit[int(key[1:])] = val
    out = qutip.tensor(qutip.basis(2, 0), Qobj(e))\
          + qutip.tensor(qutip.basis(2, 1), Qobj(g))
    return out.unit()
