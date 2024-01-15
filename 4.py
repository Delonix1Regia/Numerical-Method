import numpy as np

import numpy as np
from prettytable import PrettyTable

def gauss_jordan(A, B):
    augmented_matrix = np.column_stack((A, B))
    n = len(A)

    table = PrettyTable()
    table.field_names = [f"x{i+1}" for i in range(n)] + ["B"]

    table.add_row(list(A[0]) + [B[0]])
    table.add_row(list(A[1]) + [B[1]])
    table.add_row(list(A[2]) + [B[2]])
    table.add_row(["-" for _ in range(n)] + ["-"])

    for i in range(n):
        pivot_row = i
        for j in range(i + 1, n):
            if abs(augmented_matrix[j, i]) > abs(augmented_matrix[pivot_row, i]):
                pivot_row = j

        augmented_matrix[[i, pivot_row]] = augmented_matrix[[pivot_row, i]]
        pivot = augmented_matrix[i, i]
        augmented_matrix[i] = augmented_matrix[i] / pivot

        for k in range(n):
            if k != i:
                factor = augmented_matrix[k, i]
                augmented_matrix[k] -= factor * augmented_matrix[i]

        table.add_row([round(val, 4) for val in augmented_matrix[i]])

        if i < n - 1:
            table.add_row(["-" for _ in range(n)] + ["-"])

    solution = augmented_matrix[:, -1]

    return solution, table



def lu_decomposition(A):
    n = len(A)
    L = np.zeros((n, n))
    U = np.zeros((n, n))
    steps = []

    for i in range(n):
        step = {"Step": i + 1, "L": None, "U": None}
   
        for k in range(i, n):
            sum = 0
            for j in range(i):
                sum += (L[i][j] * U[j][k])
            U[i][k] = A[i][k] - sum

        step["U"] = np.copy(U)
       
        for k in range(i, n):
            if i == k:
                L[i][i] = 1
            else:
                sum = 0
                for j in range(i):
                    sum += (L[k][j] * U[j][i])
                L[k][i] = (A[k][i] - sum) / U[i][i]

        step["L"] = np.copy(L)
        steps.append(step)

    return L, U, steps


def gauss_seidel(A, b, x0, max_iter=100, tol=0.001):
    n = len(A)
    x = x0.copy()

    # Membuat tabel PrettyTable untuk menampilkan iterasi
    table = PrettyTable()
    table.field_names = [f"x{i+1}" for i in range(n)]

    for k in range(max_iter):
        row = [round(val, 4) for val in x]
        table.add_row(row)

        x_old = x.copy()
        for i in range(n):
            sum1 = sum(A[i][j] * x[j] for j in range(i))
            sum2 = sum(A[i][j] * x_old[j] for j in range(i + 1, n))
            x[i] = (b[i] - sum1 - sum2) / A[i][i]

        # Mengecek kriteria konvergensi
        if all(abs(x[i] - x_old[i]) < tol for i in range(n)):
            break

    return x, table


A = np.array([[11, 1, 1],
              [1, 11, 1],
              [1, 1, 11]])

B = np.array([13, 13, 13])


x0 = [0, 0, 0]


solution, iteration_table = gauss_jordan(A, B)

print("Tabel Iterasi Gauss-Jordan:")
print(iteration_table)
print('Penyelesaian dengan metode Gauss-Jordan:', solution)


print('Penyelesaian dengan metode Gauss-Seidel:', gauss_seidel(A, B, x0))

L, U, lu_steps = lu_decomposition(A)

print("Matriks L:")
print(L)
print("\nMatriks U:")
print(U)

print("\nLangkah-langkah LU Decomposition:")
for step in lu_steps:
    print(f"\nStep {step['Step']}:")
    print("Matriks L:")
    print(step["L"])
    print("\nMatriks U:")
    print(step["U"])


solution, iteration_table = gauss_seidel(A, B, x0)
print("Tabel Iterasi Gauss-Seidel:")
print(iteration_table)
print('Penyelesaian dengan metode Gauss-Seidel:', solution)