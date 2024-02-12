import numpy as np


def Tridiagonal_LU(A, n):
    b = np.diag(A)  # stores diagonal elements of the matrix A
    a = np.diag(A, k=-1)# stores lower diagonal elements of the matrix A
    a_shifted = np.zeros(len(a)+1)
    a_shifted[1:] = a

    # Make the element at the first index 0
    a_shifted[0] = 0
    c = np.diag(A, k=1)  # stores the upper diagonal elements of the matrix A
    l = np.zeros(n)  # Declaration of n-1 size array to store the lower diagonal elements of L
    v = np.zeros(n)  # Declaration of n size array to store the main diagonal elements of U
    v[0] = b[0]

    # Compute lk and vk for k=2 to n
    for i in range(1, n):
        l[i] = a_shifted[i] / v[i - 1]

        v[i] = b[i] - l[i] * c[i - 1]
        



    L = np.zeros((n, n))
    U = np.zeros((n, n))

    np.fill_diagonal(L, 1)
    np.fill_diagonal(L[1:], l[1:])

    np.fill_diagonal(U, v[:n])
    np.fill_diagonal(U[:, 1:], c)

    return L, U


# Example usage:
A = np.array([[1, 4, 0, 0],
              [3, 4, 1, 0],
              [0, 2, 3, 4],
              [0, 0, 1, 3]])
n = 4
L, U = Tridiagonal_LU(A, n)
print("L:")
print(L)
print("U:")
print(U)
