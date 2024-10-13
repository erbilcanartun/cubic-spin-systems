import numpy as np
import mpmath as mp
from lib.matrix_manipulation import *

class CubicClockModel:

    def __init__(self, energyJ, energyK, bondmoving_method='old'):

        self.b = 3 # Rescaling factor
        self.d = 3 # Dimension
        self.m = self.b**(self.d - 1) # Bond-moving multiplier

        self.q = 6
        
        self.J = energyJ
        self.K = energyK

        self.method = bondmoving_method

        def cubic_clock_transfer_matrix(j, k, method):

            J, K = mp.mpf(j), mp.mpf(k)
            
            if method == 'old':
                
                t = mp.matrix([[mp.exp(J + K),  mp.exp(J),      mp.mpf(J),      mp.mpf(-J),     mp.mpf(-J),     mp.mpf(-J - K)],
                               [mp.exp(J),      mp.exp(J + K),  mp.mpf(J),      mp.mpf(-J),     mp.mpf(-J - K), mp.mpf(-J)],
                               [mp.mpf(J),      mp.mpf(J),      mp.exp(J + K),  mp.exp(-J - K), mp.mpf(-J),     mp.mpf(-J)],
                               [mp.mpf(-J),     mp.mpf(-J),     mp.exp(-J - K), mp.exp(J + K),  mp.mpf(J),      mp.mpf(J)],
                               [mp.mpf(-J),     mp.mpf(-J - K), mp.mpf(-J),     mp.mpf(J),      mp.exp(J + K),  mp.exp(J)],
                               [mp.mpf(-J - K), mp.mpf(-J),     mp.mpf(-J),     mp.mpf(J),      mp.exp(J),      mp.exp(J + K)]])
                
            if method == 'new':
                
                vacancy = (self.q - 1) ** (1 / (2 * self.d))
                vacancy_ = (self.q - 1) ** (1 / self.q)
                
                t = mp.matrix([[mp.exp(J + K),  mp.exp(J),      mp.mpf(J),      mp.mpf(-J),     mp.mpf(-J),     mp.mpf(-J - K), vacancy],
                               [mp.exp(J),      mp.exp(J + K),  mp.mpf(J),      mp.mpf(-J),     mp.mpf(-J - K), mp.mpf(-J),     vacancy],
                               [mp.mpf(J),      mp.mpf(J),      mp.exp(J + K),  mp.exp(-J - K), mp.mpf(-J),     mp.mpf(-J),     vacancy],
                               [mp.mpf(-J),     mp.mpf(-J),     mp.exp(-J - K), mp.exp(J + K),  mp.mpf(J),      mp.mpf(J),      vacancy],
                               [mp.mpf(-J),     mp.mpf(-J - K), mp.mpf(-J),     mp.mpf(J),      mp.exp(J + K),  mp.exp(J),      vacancy],
                               [mp.mpf(-J - K), mp.mpf(-J),     mp.mpf(-J),     mp.mpf(J),      mp.exp(J),      mp.exp(J + K),  vacancy],
                               [vacancy,        vacancy,        vacancy,        vacancy,        vacancy,        vacancy,        vacancy_]]
            return t

        self.transfer_matrix = cubic_clock_transfer_matrix(self.J, self.K, self.method)


class PottsRenormalizationGroup(CubicClockModel):

    def __init__(self, energyJ, energyK, bondmoving_method='old':
        CubicClockModel.__init__(self, energyJ, energyK, bondmoving_method)
    
    def _bond_moving(self, transfer_matrix):

        n = len(transfer_matrix)
        T = mp.matrix(n)
        
        for i in range(n):
            for j in range(n):
                T[i, j] = transfer_matrix[i, j] ** self.m
        
        return T

    def _decimation(self, transfer_matrix):
        
        T = transfer_matrix * transfer_matrix
        for i in range(self.b - 2):
            T = T * transfer_matrix
            
        return T

def renormalize(self, transfer_matrix, iteration=20):
    
    T = normalizer(transfer_matrix)

    for i in range(iteration):
    
        T = self._bond_moving(T)
        T = self._decimation(T)
        T = normalizer(T)
        
    return T

def phase(self, transfer_matrix, max_iter=20):
    
    n = len(transfer_matrix)
    t = normalizer(transfer_matrix)
    e = 1e-3
    
    for i in range(10):
        t = self._bond_moving(t)
        t = self._decimation(t)
        t = normalizer(t)
    
    for i in range(max_iter):
        t = self._bond_moving(t)
        t = self._decimation(t)
        t = normalizer(t)

        tsum = np.sum(t)
        diagsum = np.sum(main_diagonal(t))
        offdiagsum = np.sum(off_diagonal(t))

        if 36-e < tsum < 36+e:
            phase = "D"
            break

        elif 2-e < tsum < 2+e and 2-e < t[2, 5] + t[5, 2] < 2+e:
            phase = "OA"
            break

        elif 6-e < diagsum < 6+e and offdiagsum < e:
            phase = "OB"
            break

        elif 2-e < tsum < 2+e and 2-e < t[2, 3] + t[3, 2] < 2+e:
            phase = "OC"
            break

        elif 4-e < tsum < 4+e and 4-e < t[0, 1] + t[1, 0] + t[4, 5] + t[5, 4] < 4+e:
            phase = "OD"
            break

        elif 4-e < tsum < 4+e and 4-e < t[0, 0] + t[1, 1] + t[4, 4] + t[5, 5] < 4+e:
            phase = "OE"
            break

        elif 12-e < tsum < 12+e and 12-e < t[0, 0] + t[0, 1] + t[1, 0] + t[1, 1] + t[2, 2] + t[2, 3] + t[3, 2] + t[3, 3] + t[4, 4] + t[4, 5] + t[5, 4] + t[5, 5] < 12+e:
            phase = "OF"
            break

        elif tsum < e and 8-e < t[0, 3] + t[1, 3] + t[3, 0] + t[3, 1] + t[2, 4] + t[2, 5] + t[4, 2] + t[5, 2] < 8+e and -8-e < t[0, 2] + t[1, 2] + t[2, 0] + t[2, 1] + t[3, 4] + t[3, 5] + t[4, 3] + t[5, 3] < -8+e:
            phase = "OG"
            break
            
        elif 4-e < tsum < 4+e and 4-e < t[2, 2] + t[2, 3] + t[3, 2] + t[3, 3] < 4+e:
            phase = "OH"
            break
            
        elif 2-e < tsum < 2+e and 2-e < t[2, 2] + t[3, 3] < 2+e:
            phase = "OI"
            break
            
        elif tsum < e and -2-e < t[2, 2] + t[3, 3] < -2+e and 2-e < t[2, 3] + t[3, 2] < 2+e:
            phase = "OJ"
            break
            
        elif tsum < e and 2-e < t[2, 2] + t[3, 3] < 2+e and -2-e < t[2, 3] + t[3, 2] < -2+e:
            phase = "OK"
            break
            
        elif 8-e < tsum < 8+e and 8-e < t[0, 0] + t[0, 1] + t[1, 0] + t[1, 1] + t[4, 4] + t[4, 5] + t[5, 4] + t[5, 5] < 8+e:
            phase = "OL"
            break
            
        elif 6-e < tsum < 6+e and 6-e < t[0, 1] + t[1, 0] + t[2, 3] + t[3, 2] + t[4, 5] + t[5, 4] < 6+e:
            phase = "OM"
            break

        else:
            phase = "X"

    return t, phase









