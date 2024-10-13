import numpy as np
import mpmath as mp
from lib.matrix_manipulation import *

class CubicPottsModel:

    def __init__(self, energyJ, energyK, state_q, data_type="npf"):

        self.b = 3 # Rescaling factor
        self.d = 3 # Dimension
        self.m = self.b**(self.d - 1) # Bond-moving multiplier
        
        self.q = state_q
        self.J = energyJ
        self.K = energyK

        self.dtype = data_type

        def cubic_potts_transfer_matrix(J, K, q, dtype):
            
            n = int(q)
            
            if dtype == "npf":

                T = np.zeros([n, n])
                for i in range(n):
                    for j in range(n):
                        if i == j:
                            T[i][j] = np.exp(J + K)
                        elif i + j == n - 1:
                            T[i][n - 1 - i] = np.exp(-K)
                        else:
                            T[i][j] = 1

            if dtype == "mpf":

                J_, K_ = mp.mpf(J), mp.mpf(K)
                T = mp.matrix(n)
                for i in range(n):
                    for j in range(n):
                        if i == j:
                            T[i, j] = mp.exp(J_ + K_)
                        elif i + j == n - 1:
                            T[i, n - 1 - i] = mp.exp(-K_)
                        else:
                            T[i, j] = mp.mpf(1)
            return T

        self.transfer_matrix = cubic_potts_transfer_matrix(self.J, self.K, self.q, self.dtype)


class PottsRenormalizationGroup(CubicPottsModel):

    def __init__(self, energyJ, energyK, state_q, data_type="npf"):
        CubicPottsModel.__init__(self, energyJ, energyK, state_q, data_type)
    
    def _bond_moving(self, transfer_matrix):
        
        n = len(transfer_matrix)
        
        if type(transfer_matrix) == (np.ndarray or list):
            T = transfer_matrix ** self.m

        if type(transfer_matrix) == mp.matrix:
            T = mp.matrix(n)
            for i in range(n):
                for j in range(n):
                    T[i, j] = transfer_matrix[i, j] ** self.m
        return T
    
    def _decimation(self, transfer_matrix):
        
        if type(transfer_matrix) == (np.ndarray or list):
            
            T = np.dot(transfer_matrix, transfer_matrix)
            for i in range(self.b - 2):
                T = np.dot(T, transfer_matrix)
        
        if type(transfer_matrix) == mp.matrix:
            
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
        T = normalizer(transfer_matrix)
    
        for i in range(max_iter):

            T = self._bond_moving(T)
            T = self._decimation(T)
            T = normalizer(T)
            
            eps = 1e-3
            n2 = n ** 2
    
            tsum = np.sum(T)
            diagsum = np.sum(main_diagonal(T))
            antidiagsum = np.sum(anti_diagonal(T))
            offdiagsum = np.sum(off_diagonal(T))
            offantidiagsum = np.sum(off_anti_diagonal(T))
    
            if type(T) == (np.ndarray or list):
    
                    if tsum > n2 - eps and tsum < n2 + eps:
                        phase = "D" # All elements 1
                        break
    
                    elif diagsum > n - eps and offdiagsum < eps:
                        phase = "OA" # Only main diagonal
                        break
    
                    elif antidiagsum > n - eps and offantidiagsum < eps:
                        phase = "OB" # Only anti-diagonal
                        break
                        
                    elif diagsum + antidiagsum > 2*n - eps and T[0][2] < eps:
                        phase = "OC" # Only main anti-diagonal
                        break
    
                    else: # ?
                        phase = "OX"
    
    
            if type(T) == mp.matrix:
    
                    if tsum > n2 - eps and tsum < n2 + eps:
                        phase = "D" # All elements 1
                        break
    
                    elif diagsum > n - eps and offdiagsum < eps:
                        phase = "OA" # Only main diagonal 1
                        break
    
                    elif antidiagsum > n - eps and offantidiagsum < eps:
                        phase = "OB" # Only anti-diagonal 1
                        break
                        
                    elif diagsum + antidiagsum > 2*n - eps and T[0, 2] < eps:
                        phase = "OC" # Only main diagonal + anti-diagonal 1
                        break
    
                    else:
                        phase = "OX" # ?
    
        return T, phase