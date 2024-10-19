import numpy as np
import mpmath as mp

def normalizer(x):
    return x / np.amax(x)

def mp_multiply(t1, t2):

    n = len(t1)
    n2 = len(t2)
    if n != n2:
        raise ValueError("Size of t1 and t2 must be the same.")

    t = mp.matrix(n)

    for i in range(n):
        for j in range(n):
            t[i, j] = t1[i, j] * t2[i, j]

    return normalizer(t)

def matrix_average(matrices):

    n = len(matrices[0])
    N = len(matrices)
    t0 = mp.matrix(n)

    for i in range(n):
        for j in range(n):
            c = 0
            for t in matrices:
                c += t[i, j]
            t0[i, j] = c / N
    return t0

def main_diagonal(x):

    n = len(x)

    #if type(x) == (np.ndarray or list):
    #    diag = np.array([])
    #    for i in range(n):
    #        for j in range(n):
    #            if i == j:
    #                diag = np.append(diag, x[i][j])

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