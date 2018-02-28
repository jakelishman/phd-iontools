import numpy as np

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
    return (np.arange(start, stop), np.arange(start, stop))

def laguerre(n, a, x):
    """laguerre(n : int >= 0, a : float, x : float) -> res : float

    Calculate the Laguerre polynomial result L_n^a(x), which is equivalent to
    Mathematica's LaguerreL[n, a, x].
    """
    if n == 0:
        return 1
    elif n == 1:
        return 1 + a - x
    # use a recurrence relation calculation for speed and accuracy
    # ref: http://functions.wolfram.com/Polynomials/LaguerreL3/17/01/01/01/
    l_2, l_1 = 1, 1 + a - x
    for m in range(2, n + 1):
        l_2, l_1 = l_1, ((a + 2*m - x - 1) * l_1 - (a + m - 1) * l_2) / m
    return l_1

def laguerre_range(n_start, n_end, a, x):
    """laguerre_range(n_start, n_end, a, x) -> np.array(dtype=np.float64)

    Use the recurrence relation for nearest-neighbour in n of the Laguerre
    polynomials to calculate
        [laguerre(n_start, a, x),
         laguerre(n_start + 1, a, x),
         ...,
         laguerre(n_end - 1, a, x)]
    in linear time of `n_end` rather than quadratic.

    Reference: http://functions.wolfram.com/Polynomials/LaguerreL3/17/01/01/01/
    """
    if n_start >= n_end:
        return np.array([])
    elif n_start == n_end - 1:
        return np.array([laguerre(n_start, a, x)])
    out = np.empty((n_end - n_start, ), dtype=np.float64)
    out[0] = laguerre(n_start, a, x)
    out[1] = laguerre(n_start + 1, a, x)
    for n in range(2, n_end - n_start):
        out[n] = ((a + 2*n - x - 1) * out[n - 1] - (a + n - 1) * out[n - 2]) / n
    return out

class Laser:
    def __init__(self, detuning, lamb_dicke, base_rabi):
        self.detuning = detuning
        self.lamb_dicke = lamb_dicke
        self.base_rabi = base_rabi
        self.rabi_mod_from_rabi =\
                (lambda rabis: rabis) if detuning == 0.0\
                else lambda rabis: np.sqrt(detuning ** 2 + rabis ** 2)

    def rabi(self, n1, n2):
        ldsq = self.lamb_dicke * self.lamb_dicke
        out = np.exp(-0.5 * ldsq) * (self.lamb_dicke ** abs(n1 - n2))
        out = out * laguerre(min(n1, n2), abs(n1 - n2), ldsq)
        fact = 1.0
        for n in range(1 + min(n1, n2), 1 + max(n1, n2)):
            fact = fact * n
        return self.base_rabi * out / np.sqrt(fact) 

    def rabi_range(self, n_start, n_end, diff):
        if diff < 0:
            n_start = n_start + diff
            n_end = n_end + diff
            diff = -diff
        if n_start >= n_end:
            return np.array([])
        elif n_start == n_end - 1:
            return np.array([self.rabi(n_start, n_start + diff)])
        ldsq = self.lamb_dicke * self.lamb_dicke
        const = np.exp(-0.5*ldsq) * self.lamb_dicke**diff * self.base_rabi
        lag = laguerre_range(n_start, n_end, diff, ldsq)
        fact = np.empty_like(lag)
        fact[0] = 1.0 / np.arange(n_start + 1.0, n_start + diff + 1).prod()
        for i in range(1, n_end - n_start):
            fact[i] = fact[i - 1] * (n_start + i) / (n_start + i + diff)
        return const * lag * np.sqrt(fact)

    def rabi_mod(self, n1, n2):
        return self.rabi_mod_from_rabi(self.rabi(n1, n2))

    def rabi_mod_range(self, n_start, n_end, diff):
        return self.rabi_mod_from_rabi(self.rabi_range(n_start, n_end, diff))

class Sideband:
    def __init__(self, ns, order, laser):
        self.__ns = ns
        self.__order = order
        self.__detuning = laser.detuning
        if order > 0:
            self.__const_indices = diagonal_indices(0, abs(order))
            self.__ee_indices = diagonal_indices(abs(order), ns)
            self.__gg_indices = diagonal_indices(ns, 2 * ns)
        else:
            self.__const_indices = diagonal_indices(ns, ns + abs(order))
            self.__ee_indices = diagonal_indices(0, ns)
            self.__gg_indices = diagonal_indices(ns + abs(order), 2 * ns)
        self.__eg_indices = ladder_indices(ns, order, (0, ns))
        self.__ge_indices = ladder_indices(ns, -order, (ns, 0))
        self.__rabi = laser.rabi_range(0, ns, abs(order))
        self.__rabi_mod = laser.rabi_mod_from_rabi(self.__rabi)
        self.__off_diag_len = self.__ns - abs(order)
        self.__eg_pre = -1.0j * np.exp(0.5j * np.pi * abs(order))\
                        * self.__rabi[:self.__off_diag_len]\
                        / self.__rabi_mod[:self.__off_diag_len]
        self.__ee_du_dt_pre = -0.5 * self.__rabi**2 / self.__rabi_mod
        self.__sin = np.empty_like(self.__rabi_mod)
        self.__cos = np.empty_like(self.__rabi_mod)
        self.__phase_time = 1.0
        self.__phase_phi = 1.0
        self.__phase_tot = 1.0
        self.__last_params = (None, None)

    def __update_if_required(self, time, phase):
        if (time, phase) == self.__last_params:
            return
        self.__last_params = (time, phase)
        self.__sin = np.sin(self.__rabi_mod * 0.5 * time)
        self.__cos = np.cos(self.__rabi_mod * 0.5 * time)
        self.__phase_time = np.exp(-0.5j * self.__detuning * time)
        self.__phase_phi = np.exp(-1j * phase)
        self.__phase_tot = self.__phase_time * self.__phase_phi

    def u(self, time, phase):
        self.__update_if_required(time, phase)
        ee = self.__cos + 1j * self.__detuning * self.__sin / self.__rabi_mod
        ee = ee * self.__phase_time

        out = np.zeros((2 * self.__ns, 2 * self.__ns), dtype=np.complex128)
        out[self.__const_indices] = 1.0
        out[self.__eg_indices] = self.__sin[:self.__off_diag_len]\
                                 * self.__eg_pre * self.__phase_tot
        out[self.__ge_indices] = -np.conj(out[self.__eg_indices])
        out[self.__ee_indices] = ee[:len(self.__ee_indices[0])]
        out[self.__gg_indices] = np.conj(ee[:len(self.__gg_indices[0])])
        return out

    def du_dt(self, time, phase):
        self.__update_if_required(time, phase)
        ee = self.__ee_du_dt_pre * self.__phase_time * self.__sin
        eg = 0.5 * self.__eg_pre * self.__phase_tot * (\
                self.__rabi_mod*self.__cos - 1j*self.__detuning*self.__sin\
             )[:self.__off_diag_len]

        out = np.zeros((2 * self.__ns, 2 * self.__ns), dtype=np.complex128)
        out[self.__ee_indices] = ee[:len(self.__ee_indices[0])]
        out[self.__gg_indices] = np.conj(ee[:len(self.__gg_indices[0])])
        out[self.__eg_indices] = eg
        out[self.__ge_indices] = -np.conj(eg)
        return out

    def du_dphi(self, time, phase):
        self.__update_if_required(time, phase)
        eg = -1j*self.__eg_pre*self.__sin[:self.__off_diag_len]*self.__phase_tot
        out = np.zeros((2 * self.__ns, 2 * self.__ns), dtype=np.complex128)
        out[self.__eg_indices] = eg
        out[self.__ge_indices] = -np.conj(eg)
        return out
