import numpy as np
import mpmath as mp

def normalizer(x):
    return x / np.amax(x)

def main_diagonal(x):
    
    n = len(x)
    
    if type(x) == (np.ndarray or list):
        diag = np.array([])
        for i in range(n):
            for j in range(n):
                if i == j:
                    diag = np.append(diag, x[i][j])
                    
    if type(x) == mp.matrix:
        diag = np.array([])
        for i in range(n):
            for j in range(n):
                if i == j:
                    diag = np.append(diag, x[i, j])
    return diag

def anti_diagonal(x):
    
    n = len(x)
    
    if type(x) == (np.ndarray or list):
        antidiag = np.array([])
        for i in range(n):
                antidiag = np.append(antidiag, x[i][n - 1 - i])
                
    if type(x) == mp.matrix:
        antidiag = np.array([])
        for i in range(n):
                antidiag = np.append(antidiag, x[i, n - 1 - i])

    return antidiag

def off_diagonal(x):
    return np.extract(1 - np.eye(len(x)), x)

def off_anti_diagonal(x):
    
    n = len(x)
    a = np.ones((n, n))
    np.fill_diagonal(np.fliplr(a), 0)
    
    return np.extract(a, x)