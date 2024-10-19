import numpy as np
import mpmath as mp
from lib.matrix_manipulation import *
from lib.renormalization.utils import interaction


class CubicIsingModel:

    def __init__(self, energy, field, component, field_direction='x',
                 aferro_concentration=0, lattice_size=500):

        self.b = 3 # Rescaling factor
        self.d = 3 # Dimension
        self.m = self.b ** (self.d - 1) # Bond-moving multiplier

        self.J = mp.mpf(energy)
        self.H = mp.mpf(field)
        self.n = component
        self.u = field_direction

        self.p = aferro_concentration
        self.N = lattice_size

        def cubic_ising_transfer_matrix(j, h, n, u):

            if n == 1:
                if u == 'x':
                    t = mp.matrix([[mp.exp(j + 2*h), mp.exp(-j)],
                                   [mp.exp(-j),      mp.exp(j - 2*h)]])

            elif n == 2:
                if u == 'x':
                    t = mp.matrix([[mp.exp(j + 2*h), mp.exp(-j),      mp.exp(h),  mp.exp(h)],
                                   [mp.exp(-j),      mp.exp(j - 2*h), mp.exp(-h),  mp.exp(-h)],
                                   [mp.exp(h),       mp.exp(-h),       mp.exp(j),  mp.exp(-j)],
                                   [mp.exp(h),       mp.exp(-h),       mp.exp(-j), mp.exp(j)]])

                if u == 'xy':
                    t = mp.matrix([[mp.exp(j + 2*h), mp.exp(-j),      mp.mpf(2 * h),   mp.mpf(1)],
                                   [mp.exp(-j),      mp.exp(j - 2*h), mp.mpf(1),       mp.exp(-2 * h)],
                                   [mp.exp(2 * h),   mp.mpf(1),       mp.exp(j + 2*h), mp.exp(-j)],
                                   [mp.mpf(1),       mp.exp(-2 * h),  mp.exp(-j),      mp.exp(j - 2*h)]])

            elif n == 3:
                if u == 'x':
                    t = mp.matrix([[mp.exp(j + 2*h), mp.exp(-j),      mp.exp(h),  mp.exp(h),  mp.exp(h),  mp.exp(h)],
                                   [mp.exp(-j),      mp.exp(j - 2*h), mp.exp(-h), mp.exp(-h), mp.exp(-h), mp.exp(-h)],
                                   [mp.exp(h),       mp.exp(-h),      mp.exp(j),  mp.exp(-j), mp.mpf(1),  mp.mpf(1)],
                                   [mp.exp(h),       mp.exp(-h),      mp.exp(-j), mp.exp(j),  mp.mpf(1),  mp.mpf(1)],
                                   [mp.exp(h),       mp.exp(-h),      mp.mpf(1),  mp.mpf(1),  mp.exp(j),  mp.exp(-j)],
                                   [mp.exp(h),       mp.exp(-h),      mp.mpf(1),  mp.mpf(1),  mp.exp(-j), mp.exp(j)]])

                if u == 'xy':
                    t = mp.matrix([[mp.exp(j + 2*h), mp.exp(-j),      mp.exp(2 * h),   mp.mpf(1),       mp.exp(h),  mp.exp(h)],
                                   [mp.exp(-j),      mp.exp(j - 2*h), mp.mpf(1),       mp.exp(-2 * h),  mp.exp(-h), mp.exp(-h)],
                                   [mp.exp(2 * h),   mp.mpf(1),       mp.exp(j + 2*h), mp.exp(-j),      mp.exp(h),  mp.exp(h)],
                                   [mp.mpf(1),       mp.exp(-2 * h),  mp.exp(-j),      mp.exp(j - 2*h), mp.exp(-h), mp.exp(-h)],
                                   [mp.exp(h),       mp.exp(-h),      mp.exp(h),       mp.exp(-h),      mp.exp(j),  mp.exp(-j)],
                                   [mp.exp(h),       mp.exp(-h),      mp.exp(h),       mp.exp(-h),      mp.exp(-j), mp.exp(j)]])

                if u == 'xyz':
                    t = mp.matrix([[mp.exp(j + 2*h), mp.exp(-j),      mp.exp(2 * h),   mp.mpf(1),        mp.exp(2 * h),   mp.mpf(1)],
                                   [mp.exp(-j),      mp.exp(j - 2*h), mp.mpf(1),       mp.exp(-2 * h),   mp.mpf(1),       mp.exp(-2 * h)],
                                   [mp.exp(2 * h),   mp.mpf(1),       mp.exp(j + 2*h), mp.exp(-j),       mp.exp(2 * h),   mp.mpf(1)],
                                   [mp.mpf(1),       mp.mpf(-2 * h),  mp.exp(-j),      mp.exp(j - 2*h),  mp.mpf(1),       mp.exp(-2 * h)],
                                   [mp.exp(2 * h),   mp.mpf(1),       mp.exp(2 * h),   mp.mpf(1),        mp.exp(j + 2*h), mp.exp(-j)],
                                   [mp.mpf(1),       mp.mpf(-2 * h),  mp.mpf(1),       mp.exp(-2 * h),   mp.exp(-j),      mp.exp(j - 2*h)]])

            elif n == 4:
                t = mp.matrix([[mp.exp(j), mp.exp(-j), mp.mpf(1), mp.mpf(1), mp.mpf(1), mp.mpf(1), mp.mpf(1), mp.mpf(1)],
                               [mp.exp(-j), mp.exp(j), mp.mpf(1), mp.mpf(1), mp.mpf(1), mp.mpf(1), mp.mpf(1), mp.mpf(1)],
                               [mp.mpf(1), mp.mpf(1), mp.exp(j), mp.exp(-j), mp.mpf(1), mp.mpf(1), mp.mpf(1), mp.mpf(1)],
                               [mp.mpf(1), mp.mpf(1), mp.exp(-j), mp.exp(j), mp.mpf(1), mp.mpf(1), mp.mpf(1), mp.mpf(1)],
                               [mp.mpf(1), mp.mpf(1), mp.mpf(1), mp.mpf(1), mp.exp(j), mp.exp(-j), mp.mpf(1), mp.mpf(1)],
                               [mp.mpf(1), mp.mpf(1), mp.mpf(1), mp.mpf(1), mp.exp(-j), mp.exp(j), mp.mpf(1), mp.mpf(1)],
                               [mp.mpf(1), mp.mpf(1), mp.mpf(1), mp.mpf(1), mp.mpf(1), mp.mpf(1), mp.exp(j), mp.exp(-j)],
                               [mp.mpf(1), mp.mpf(1), mp.mpf(1), mp.mpf(1), mp.mpf(1), mp.mpf(1), mp.exp(-j), mp.exp(j)]])

            elif n == 5:
                t = mp.matrix([[mp.exp(j), mp.exp(-j), mp.mpf(1), mp.mpf(1), mp.mpf(1), mp.mpf(1), mp.mpf(1), mp.mpf(1), mp.mpf(1), mp.mpf(1)],
                               [mp.exp(-j), mp.exp(j), mp.mpf(1), mp.mpf(1), mp.mpf(1), mp.mpf(1), mp.mpf(1), mp.mpf(1), mp.mpf(1), mp.mpf(1)],
                               [mp.mpf(1), mp.mpf(1), mp.exp(j), mp.exp(-j), mp.mpf(1), mp.mpf(1), mp.mpf(1), mp.mpf(1), mp.mpf(1), mp.mpf(1)],
                               [mp.mpf(1), mp.mpf(1), mp.exp(-j), mp.exp(j), mp.mpf(1), mp.mpf(1), mp.mpf(1), mp.mpf(1), mp.mpf(1), mp.mpf(1)],
                               [mp.mpf(1), mp.mpf(1), mp.mpf(1), mp.mpf(1), mp.exp(j), mp.exp(-j), mp.mpf(1), mp.mpf(1), mp.mpf(1), mp.mpf(1)],
                               [mp.mpf(1), mp.mpf(1), mp.mpf(1), mp.mpf(1), mp.exp(-j), mp.exp(j), mp.mpf(1), mp.mpf(1), mp.mpf(1), mp.mpf(1)],
                               [mp.mpf(1), mp.mpf(1), mp.mpf(1), mp.mpf(1), mp.mpf(1), mp.mpf(1), mp.exp(j), mp.exp(-j), mp.mpf(1), mp.mpf(1)],
                               [mp.mpf(1), mp.mpf(1), mp.mpf(1), mp.mpf(1), mp.mpf(1), mp.mpf(1), mp.exp(-j), mp.exp(j), mp.mpf(1), mp.mpf(1)],
                               [mp.mpf(1), mp.mpf(1), mp.mpf(1), mp.mpf(1), mp.mpf(1), mp.mpf(1), mp.mpf(1), mp.mpf(1), mp.exp(j), mp.exp(-j)],
                               [mp.mpf(1), mp.mpf(1), mp.mpf(1), mp.mpf(1), mp.mpf(1), mp.mpf(1), mp.mpf(1), mp.mpf(1), mp.exp(-j), mp.exp(j)]])

            else:
                raise TransferMatrixError("Only 1, 2, and 3 can be given as number of components.")

            return t

        self.transfer_matrix = cubic_ising_transfer_matrix(self.J, self.H, self.n, self.u)

        def cubic_ising_transfer_matrices(j, h, n, u, p, N):

            ferro  = cubic_ising_transfer_matrix(j, h, n, u)
            aferro = cubic_ising_transfer_matrix(-j, h, n, u)

            return [ferro for _ in range(int((1 - p) * N))] + [aferro for _ in range(int(p * N))]

        self.transfer_matrices = cubic_ising_transfer_matrices(self.J, self.H, self.n, self.u, self.p, self.N)


class IsingRenormalizationGroup(CubicIsingModel):

    def __init__(self, energy, field, component, field_direction, aferro_concentration, lattice_size):
        CubicIsingModel.__init__(self, energy, field, component, field_direction, aferro_concentration, lattice_size)

    def _bond_moving(self, matrices):
        t = matrices[0]
        n = len(matrices)

        for i in range(n - 1):
            t = mp_multiply(t, matrices[i + 1])

        return normalizer(t)

    def _decimation(self, matrices):
        t = matrices[0] * matrices[1]
        t = normalizer(t)
        t = t * matrices[2]

        return normalizer(t)

    def renormalize(self, matrices):

        np.random.seed(19)

        num = self.b * self.m # 27
        N = len(matrices)

        renormalized = []
        for k in range(N):

            random = []
            for _ in range(num):
                i = np.random.randint(1, N)
                random.append(matrices[i])

            # Renormalize
            dm = []
            for i in range(0, len(random) - 1, 3):
                dm.append(self._decimation(random[i:i + 3]))
            r = self._bond_moving(dm)
            renormalized.append(r)

        return renormalized

    def _deviator_transfer_matrix(self, j):
        # Creates a zero-field transfer matrix
        J = mp.mpf(j)

        n = 2 * self.n
        t = mp.matrix(n)
        diag_element_counter = 0

        for i in range(n):
            for j in range(n):

                if i == j:
                    diag_element_counter += 1
                    t[i, j] = mp.exp(J)

                    if j + 1 <= n - 1 and diag_element_counter % 2 == 1:
                        t[i, j + 1] = mp.exp(-J)

                    if j - 1 >= 0 and diag_element_counter % 2 == 0:
                        t[i, j - 1] = mp.exp(-J)

                else:
                    if t[i, j] == 0:
                        t[i, j] = mp.exp(0)

        return normalizer(t)

    def renormalize_chaos(self, matrices, deviation=1e-3):
        np.random.seed(19)

        num = self.b * self.m # 27

        N = len(matrices)
        e = self._deviator_transfer_matrix(j = deviation)
        renormalized = []
        for k in range(N):

            random = []
            for _ in range(num):

                i = np.random.randint(1, N)
                random.append(matrices[i])

            # Renormalize
            dm = []
            for i in range(0, len(random) - 1, 3):
                dm.append(self._decimation(random[i:i + 3]))

            r = self._bond_moving(dm)
            renormalized.append(r)

            if k == 19: # Arbitrary index
                normal = r
                deviated_random = []
                for i in range(num):
                    t = mp_multiply(random[i], e)
                    deviated_random.append(t)

                # Renormalize deviated bond
                dm = []
                for i in range(0, len(deviated_random) - 1, 3):
                    dm.append(self._decimation(deviated_random[i:i + 3]))
                deviated = self._bond_moving(dm)

        return renormalized, interaction(normal), interaction(deviated)

    def phase(self):

        #limsup = 0.95 * lattice_size
        #liminf = 0.05 * lattice_size

        matrices = self.transfer_matrices
        phase = "undetermined"
        e = 1e-3
        RG = 20 # RG iteration number

        if self.n == 1:
            for k in range(RG):

                matrices = self.renormalize(matrices)

                t = matrix_average(matrices)
                tsum = np.sum(t)
                diagsum = np.sum(main_diagonal(t))
                offdiagsum = np.sum(off_diagonal(t))

                if 4 - e < tsum < 4 + e:
                    phase = "disorder"
                    break

                # Main diagonal 1, others 0 (zero field)
                elif 2 - e < t[0, 0] + t[1, 1] < 2 + e and t[0, 1] + t[1, 0] < e:
                    phase = "ferro"
                    break

                # Anti diagonal 1, others 0 (zero field)
                elif 2 - e < t[0, 1] + t[1, 0] < 2 + e and t[0, 0] + t[1, 1] < e:
                    phase = "antiferro"
                    break

                # First element 1, others 0
                elif 1 - e < t[0, 0] < 1 + e and t[0, 1] + t[1, 0] + t[1, 1] < e:
                    phase = "x_aligned"
                    break

                # Last element 1, others 0
                elif 1 - e < t[1, 1] < 1 + e and t[0, 1] + t[1, 0] + t[0, 0] < e:
                    phase = "x_antialigned"
                    break

                else:
                    phase = "unknown"

        if self.n == 2:
            for k in range(RG):

                matrices = self.renormalize(matrices)

                t = matrix_average(matrices)
                tsum = np.sum(t)
                diagsum = np.sum(main_diagonal(t))
                offdiagsum = np.sum(off_diagonal(t))

                if 16 - e < tsum < 16 + e:
                    phase = "disorder"
                    break

                elif 4 - e < diagsum < 4 + e and offdiagsum < e:
                    phase = "ferro"
                    break

                elif 1 - e < tsum < 1 + e and 1 - e < t[0, 0] < 1 + e:
                    phase = "x_aligned"
                    break

                elif 4 - e < tsum < 4 + e and 4 - e < t[0, 0] + t[0, 2] + t[2, 0] + t[2, 2] < 4 + e:
                    phase = "xy_aligned"
                    break

                elif 2 - e < tsum < 2 + e and 2 - e < t[0, 0] + t[2, 2] < 2 + e:
                    phase = "xory_aligned"
                    break

                elif 2 - e < tsum < 2 + e and 2 - e < t[0, 2] + t[2, 0] < 2 + e:
                    phase = "xy_alternating"
                    break

                elif 2 - e < tsum < 2 + e and 2 - e < t[0, 1] + t[1, 0] < 2 + e:
                    phase = "x_antiferro"
                    break

                elif 4 - e < tsum < 4 + e and 4 - e < t[0, 1] + t[1, 0] + t[2, 3] + t[3, 2] < 4 + e:
                    phase = "xory_antiferro"
                    break

                else:
                    phase = "unknown"

        if self.n == 3:
            for k in range(RG):

                matrices = self.renormalize(matrices)

                t = matrix_average(matrices)
                tsum = np.sum(t)
                diagsum = np.sum(main_diagonal(t))
                offdiagsum = np.sum(off_diagonal(t))

                if 36 - e < tsum < 36 + e:
                    phase = "disorder"
                    break

                elif 6-e < diagsum < 6+e and offdiagsum < e:
                    phase = "ferro"
                    break

                elif 1-e < tsum < 1+e and 1-e < t[0, 0] < 1+e:
                    phase = "x_aligned"
                    break

                elif 4-e < tsum < 4+e and 4-e < t[0, 0] + t[0, 2] + t[2, 0] + t[2, 2] < 4+e:
                    phase = "xy_aligned"
                    break

                elif 9 - e < tsum < 9 + e and 9 - e < t[0, 0] + t[0, 2] + t[0, 4] + t[2, 0] + t[2, 2] + t[2, 4] + t[4, 0] + t[4, 2] + t[4, 4] < 9 + e:
                    phase = "xyz_aligned"
                    break

                elif 2 - e < tsum < 2 + e and 2 - e < t[0, 0] + t[2, 2] < 2 + e:
                    phase = "xory_aligned"
                    break

                elif 3 - e < tsum < 3 + e and 3 - e < t[0, 0] + t[2, 2] + t[4, 4] < 3 + e:
                    phase = "xoryorz_aligned"
                    break

                elif 2 - e < tsum < 2 + e and 2 - e < t[0, 2] + t[2, 0] < 2 + e:
                    phase = "xy_alternating"
                    break

                elif 2 - e < tsum < 2 + e and 2 - e < t[0, 1] + t[1, 0] < 2 + e:
                    phase = "x_antiferro"
                    break

                elif 4 - e < tsum < 4 + e and 4 - e < t[0, 1] + t[1, 0] + t[2, 3] + t[3, 2] < 4 + e:
                    phase = "xory_antiferro"
                    break

                elif 6 - e < tsum < 6 + e and 6 - e < t[0, 1] + t[1, 0] + t[2, 3] + t[3, 2] + t[4, 5] + t[5, 4] < 6 + e:
                    phase = "xoryorz_antiferro"
                    break

                elif 12 - e < tsum < 12 + e and 12 - e < t[0, 0] + t[0, 1] + t[1, 0] + t[1, 1] + t[2, 2] + t[2, 3] + t[3, 2] + t[3, 3] + t[4, 4] + t[4, 5] + t[5, 4] + t[5, 5] < 12 + e:
                    phase = "nematic"
                    break

                else:
                    phase = "unknown"

        return k, phase