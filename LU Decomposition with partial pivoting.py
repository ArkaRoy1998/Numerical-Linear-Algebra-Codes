import numpy as np


# Helper Function 1
def Pivot_Matrix(M):  # Function to calculate the Pivot Matrix P for PA = LU
    m = len(M)
    identity_matrix = np.eye(m)  # Identity matrix of size m
    P = np.eye(m)
    pivot_array = []  # Array to keep track of pivots

    for j in range(m - 1):
        max_row_index = j
        max_abs_value = abs(M[j][j])
        for i in range(j + 1, m):
            abs_value = abs(M[i][j])
            if abs_value > max_abs_value:
                max_abs_value = abs_value
                max_row_index = i

        if max_abs_value == 0:
            return None, None

        pivot_array.append(max_row_index)  # Storing pivot index

        if j != max_row_index:
            # Swap the rows
            P[[j, max_row_index]] = P[[max_row_index, j]]

    return P, pivot_array


# Helper Function 2
def matrix_multiplication(A, B):  # Function to multiply two matrices
    n = len(A)
    C = [[0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]

    return C


# main function
def lu_decomposition(A, b):  # Performs an LU Decomposition of A (which must be square)
    #                                                                       into PA = LU. The function returns P,
    #                                                                       L ,U and x where Ax = b
    n = len(A)
    L = np.zeros((n, n))
    np.fill_diagonal(L, 1)  # make the diagonal elements of L as 1
    P, pivot_array = Pivot_Matrix(A)
    if P is None:
        print("Unable to find nonzero pivot")
        return

    PA = matrix_multiplication(P, A)
    Pb = np.dot(P, b)  # required to compute Ly = Pb
    U = PA

    for k in range(0, n - 1):
        for j in range(k + 1, n):
            L[j][k] = U[j][k] / U[k][k]
            for i in range(k, n):
                U[j][i] = U[j][i] - (L[j][k] * U[k][i])

    # Solve for Ly = Pb
    y = np.zeros(n)
    for i in range(n):
        y[i] = Pb[i]
        for j in range(i):
            y[i] -= L[i][j] * y[j]

    # Solve for Ux = y
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = y[i]
        for j in range(i + 1, n):
            x[i] -= U[i][j] * x[j]
        x[i] /= U[i][i]

    return L, U, P, x


A1 = [[1, 2], [3, 4]]
b1 = [5, 6]

print(lu_decomposition(A1, b1))
