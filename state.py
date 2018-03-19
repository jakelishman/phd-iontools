import numpy as np
import qutip

__all__ = ["create", "ns", "element", "extend", "populated_elements",
           "qubit_projectors"]

def create(populated, ns=1, normalise=True):
    """create(populated, ns=1, normalise=True) -> qutip.Qobj
    Create a normalised state which considers at least `ns` motional states,
    with the populations filled by the specifications in `dict`.

    Arguments:
    populated: (dictionary | iterable) of state_name * value
    with
        state_name: ("g" | "e") + string(int) --
            The identifier of the state to be filled, for example the string
            "g3" is the state in the ground state and the n=3 motional level.
        value: complex -- The relative value of that state.

    ns (optional): int --
        The minimum number of motional states to consider.  The resulting vector
        will always be at least twice the length of `ns`, but could be longer if
        there is a greater requirement in `dict`.

    normalise (optional): bool --
        Whether to normalise the state before returning it.  Defaults to `True`.

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
    if isinstance(populated, dict):
        if not populated:
            raise TypeError("At least one populated state must be specified.")
        pairs = populated.items
    else:
        if hasattr(populated, "__len__") and len(populated) is 0:
            raise TypeError("At least one populated state must be specified.")
        base = list(populated)
        pairs = lambda: base
    for (key, _) in pairs():
        ns = max(ns, 1 + int(key[1:]))
    e = np.zeros(ns, dtype=np.complex128)
    g = np.zeros_like(e)
    for (key, val) in pairs():
        qubit = g if key[0] is "g" else e
        qubit[int(key[1:])] = val
    out = qutip.tensor(qutip.basis(2, 0), qutip.Qobj(e))\
          + qutip.tensor(qutip.basis(2, 1), qutip.Qobj(g))
    return out.unit() if normalise else out

def ns(state):
    """ns(state: qutip.Qobj) -> int

    Return the number of motional states tracked in the Fock basis for the given
    state."""
    return state.dims[0][1]

def extend(state, ns_):
    """extend(state: qutip.Qobj, ns_: int) -> qutip.Qobj

    Extend the number of motional states considered in the two-level system
    `state` to be exactly `ns_`.  Throws ValueError instead of truncating the
    state if `ns_` is less than is currently used in `state`."""
    cur_ns = ns(state)
    if ns_ is cur_ns:
        return state.copy()
    elif ns_ < cur_ns:
        raise ValueError("Can't reduce the size of the state.")
    data = state.full().reshape(2 * cur_ns)
    out = np.zeros(2 * ns_, dtype=np.complex128)
    out[:cur_ns] = data[:cur_ns]
    out[ns_:ns_ + cur_ns] = data[cur_ns:]
    return qutip.Qobj(out, dims=[[2, ns_], [1, 1]])

def element(state, els):
    """element(state, els) -> complex or np.array of complex

    Return either the single complex value of the single element `els`, or an
    array of each of the elements `els` in order.

    Arguments:
    state: qutip.Qobj -- the state vector (ket) to extract the elements from.
    els: array_like -- the element or list of elements to extract.

    Returns:
    complex or np.array of complex -- the extracted elements in order."""
    ns_ = ns(state)
    idx = lambda s: (0 if s[0] is "e" else ns_) + int(s[1:])
    if isinstance(els, str):
        return state.full().flat[idx(els)]
    return state.full().flat[[idx(el) for el in els]]

def populated_elements(state):
    """populated_elements(state: qutip.Qobj) -> np.array of string * complex

    Given a state vector of a 2-level system tensor a Fock state, return a list
    of (name * value) pairs for each of the populated states.  `name` is a
    string of the qubit followed by the Fock level (e.g. "g0" or "e10"), and
    `value` is the complex value that was in the state."""
    digits = len(str(state.dims[0][1] - 1))
    return np.array([
        (qubit + str(i), val) \
        for qubit, proj in zip(["e", "g"], qubit_projectors(state)) \
        for i, val in enumerate((proj * state).full().flat) \
        if abs(val) > 1e-11
    ], dtype=np.dtype([("element", "U{}".format(digits+1)), ("value", "c16")]))

def qubit_projectors(target):
    """qubit_projectors(target: qutip.Qobj | int) -> qutip.Qobj * qutip.Qobj

    Given either an example state (2-level system tensor Fock basis) or a number
    of motional states to be considered in the Fock basis, return a pair of the
    projectors onto the "e" and "g" subspaces respectively."""
    ns = target.dims[0][1] if isinstance(target, qutip.Qobj) else target
    return (qutip.tensor(qutip.basis(2, 0), qutip.qeye(ns)).dag(),
            qutip.tensor(qutip.basis(2, 1), qutip.qeye(ns)).dag())
