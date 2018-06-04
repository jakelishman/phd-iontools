"""
Tools for working with single ions in an analytical manner.  This packages
contains tools for simulating single ion dynamics analytically, whether inside
or outside the Lamb--Dicke regime.  Everything in this module is done in terms
of `qutip.Qobj`s.

The main assumption, however, is that off-resonant interactions are completely
suppressed.  This requires that the trap frequency must be much larger than the
base Rabi frequency, which in turn must be much larger than any detuning away
from a particular transition.

The `Sideband` class is the main tool for working with the interaction of a
single sideband with an ion.  The `Sequence` class is used for working with
whole discrete pulse sequences of these sidebands.  A `Laser` contains the
necessary operating parameters to calculate the operators for these classes, and
the `state` module, including in particular `state.create()` are used for
creating the `qutip.Qobj` kets corresponding to the trapped ion states.

Also included in the `alg` package are analytical algorithms for calculating
interesting pulse sequences in the trapped ions.
"""

from .laser import *
from .sideband import *
from .sequence import *
from . import state
from . import alg

__all__ = laser.__all__ + sideband.__all__ + sequence.__all__ + ["state", "alg"]
