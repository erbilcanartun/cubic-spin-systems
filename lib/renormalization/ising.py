import numpy as np
import mpmath as mp
from lib.matrix_manipulation import *

class CubicIsingModel:

    def __init__(self, energy, field, component, field_direction='x', aferro_concentration=0, lattice_size=500):

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
                                   [mp.exp(-j), mp.exp(j - 2*h)]])

            elif n == 2:
                if u == 'x':
                    t = mp.matrix([[mp.exp(j + 2*h), mp.exp(-j), mp.mpf(1), mp.mpf(1)],
                                   [mp.exp(-j), mp.exp(j - 2*h), mp.mpf(1), mp.mpf(1)],
                                   [mp.mpf(1), mp.mpf(1), mp.exp(j), mp.exp(-j)],
                                   [mp.mpf(1), mp.mpf(1), mp.exp(-j), mp.exp(j)]])

                if u == 'xy':
                    t = mp.matrix([[mp.exp(j + 2*h), mp.exp(-j), mp.mpf(1), mp.mpf(1)],
                                   [mp.exp(-j), mp.exp(j - 2*h), mp.mpf(1), mp.mpf(1)],
                                   [mp.mpf(1), mp.mpf(1), mp.exp(j + 2*h), mp.exp(-j)],
                                   [mp.mpf(1), mp.mpf(1), mp.exp(-j), mp.exp(j - 2*h)]])

            elif n == 3:
                if u == 'x':
                    t = mp.matrix([[mp.exp(j + 2*h), mp.exp(-j), mp.mpf(1), mp.mpf(1), mp.mpf(1), mp.mpf(1)],
                                   [mp.exp(-j), mp.exp(j - 2*h), mp.mpf(1), mp.mpf(1), mp.mpf(1), mp.mpf(1)],
                                   [mp.mpf(1), mp.mpf(1), mp.exp(j), mp.exp(-j), mp.mpf(1), mp.mpf(1)],
                                   [mp.mpf(1), mp.mpf(1), mp.exp(-j), mp.exp(j), mp.mpf(1), mp.mpf(1)],
                                   [mp.mpf(1), mp.mpf(1), mp.mpf(1), mp.mpf(1), mp.exp(j), mp.exp(-j)],
                                   [mp.mpf(1), mp.mpf(1), mp.mpf(1), mp.mpf(1), mp.exp(-j), mp.exp(j)]])

                if u == 'xy':
                    t = mp.matrix([[mp.exp(j + 2*h), mp.exp(-j), mp.mpf(1), mp.mpf(1), mp.mpf(1), mp.mpf(1)],
                                   [mp.exp(-j), mp.exp(j - 2*h), mp.mpf(1), mp.mpf(1), mp.mpf(1), mp.mpf(1)],
                                   [mp.mpf(1), mp.mpf(1), mp.exp(j + 2*h), mp.exp(-j), mp.mpf(1), mp.mpf(1)],
                                   [mp.mpf(1), mp.mpf(1), mp.exp(-j), mp.exp(j - 2*h), mp.mpf(1), mp.mpf(1)],
                                   [mp.mpf(1), mp.mpf(1), mp.mpf(1), mp.mpf(1), mp.exp(j), mp.exp(-j)],
                                   [mp.mpf(1), mp.mpf(1), mp.mpf(1), mp.mpf(1), mp.exp(-j), mp.exp(j)]])

                if u == 'xyz':
                    t = mp.matrix([[mp.exp(j + 2*h), mp.exp(-j), mp.mpf(1), mp.mpf(1), mp.mpf(1), mp.mpf(1)],
                                   [mp.exp(-j), mp.exp(j - 2*h), mp.mpf(1), mp.mpf(1), mp.mpf(1), mp.mpf(1)],
                                   [mp.mpf(1), mp.mpf(1), mp.exp(j + 2*h), mp.exp(-j), mp.mpf(1), mp.mpf(1)],
                                   [mp.mpf(1), mp.mpf(1), mp.exp(-j), mp.exp(j - 2*h), mp.mpf(1), mp.mpf(1)],
                                   [mp.mpf(1), mp.mpf(1), mp.mpf(1), mp.mpf(1), mp.exp(j + 2*h), mp.exp(-j)],
                                   [mp.mpf(1), mp.mpf(1), mp.mpf(1), mp.mpf(1), mp.exp(-j), mp.exp(j - 2*h)]])

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
        t = normalizer(t)

        return t

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