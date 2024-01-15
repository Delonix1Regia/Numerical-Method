import numpy as np
from prettytable import PrettyTable

def lagrange_interpolation(x, y, target_x):
    n = len(x)
    interpolated_value = 0

    # Inisialisasi tabel
    table = PrettyTable()
    table.field_names = ["Iterasi", "Interpolated Value", "Approximate Error"]

    for i in range(n):
        L_i = 1
        for j in range(n):
            if i != j:
                L_i *= (target_x - x[j]) / (x[i] - x[j])

        interpolated_value += L_i * y[i]

        # Tambahkan baris ke tabel
        table.add_row([i + 1, round(interpolated_value, 4), ""])

    # Hitung approximate error (dalam hal ini, selisih antara hasil interpolasi dan nilai eksak)
    exact_value = lagrange_exact(target_x)
    approx_error = abs(interpolated_value - exact_value)

    # Tambahkan baris untuk approximate error
    table.add_row(["Approx. Error", "", round(approx_error, 4)])

    # Tampilkan tabel
    print(table)

    return interpolated_value

# Fungsi untuk nilai eksak pada x yang diberikan (misalnya, fungsi yang diinterpolasi)
def lagrange_exact(x):
    return x**3 + 2*x**2 + 1




def divided_diffs(x, y):
    n = len(y)
    table = np.zeros((n, n))
    table[:, 0] = y
    
    for j in range(1, n):
        for i in range(n - j):
            table[i][j] = (table[i + 1][j - 1] - table[i][j - 1]) / (x[i + j] - x[i])

    return table[0]

def newton_interpolation(x_values, y_values, x_interpolate):
    coeffs = divided_diffs(x_values, y_values)
    result = coeffs[0]
    temp = 1

    # Inisialisasi tabel
    table = PrettyTable()
    table.field_names = ["Iterasi", "Interpolated Value", "Approximate Error"]

    # Tambahkan baris untuk iterasi pertama
    table.add_row([0, round(result, 4), ""])

    for i in range(1, len(coeffs)):
        temp *= (x_interpolate - x_values[i - 1])
        result += coeffs[i] * temp

        # Hitung approximate error (dalam hal ini, selisih antara hasil interpolasi dan nilai eksak)
        exact_value = newton_exact(x_interpolate)
        approx_error = abs(result - exact_value)

        # Tambahkan baris ke tabel
        table.add_row([i, round(result, 4), round(approx_error, 4)])

    # Tampilkan tabel
    print(table)

    return result

# Fungsi untuk nilai eksak pada x yang diberikan (misalnya, fungsi yang diinterpolasi)
def newton_exact(x):
    return x**3 + 2*x**2 + 1


def spline_cubic_interpolation(x_values, y_values, x_interpolate):
    n = len(x_values) - 1
    h = np.diff(x_values)
    
    # Matriks sistem persamaan linear
    A = np.zeros((n + 1, n + 1))
    b = np.zeros(n + 1)
    
    A[0, 0] = 1
    A[n, n] = 1
    
    for i in range(1, n):
        A[i, i - 1] = h[i - 1]
        A[i, i] = 2 * (h[i - 1] + h[i])
        A[i, i + 1] = h[i]
        b[i] = 3 * ((y_values[i + 1] - y_values[i]) / h[i] - (y_values[i] - y_values[i - 1]) / h[i - 1])
    
    # Solusi sistem persamaan linear
    c = np.linalg.solve(A, b)
    
    # Koefisien untuk interpolasi
    a = y_values[:-1]
    b = (y_values[1:] - y_values[:-1]) / h - h * (c[:-1] + 2 * c[1:]) / 3
    d = (c[1:] - c[:-1]) / (3 * h)
    
    # Interpolasi pada x_interpolate
    for i in range(n):
        if x_values[i] <= x_interpolate <= x_values[i + 1]:
            dx = x_interpolate - x_values[i]
            result = a[i] + b[i] * dx + c[i] * dx**2 + d[i] * dx**3

            # Inisialisasi tabel
            table = PrettyTable()
            table.field_names = ["Iterasi", "Interpolated Value", "Approximate Error"]

            # Tambahkan baris untuk iterasi pertama
            table.add_row([0, round(result, 4), ""])

            # Hitung approximate error (dalam hal ini, selisih antara hasil interpolasi dan nilai eksak)
            exact_value = spline_cubic_exact(x_interpolate)
            approx_error = abs(result - exact_value)

            # Tambahkan baris ke tabel
            table.add_row(["Approx. Error", "", round(approx_error, 4)])

            # Tampilkan tabel
            print(table)

            return result

def spline_cubic_exact(x):
    return x**3 + 2*x**2 + 1



# Data
x_values = np.array([100, 200, 300, 400, 500])
y_values = np.array([11, 13, 23, 25, 27])

# Pekerja yang akan diinterpolasi
x_interpolate = 350

# Interpolasi menggunakan metode Lagrange orde 3
result = lagrange_interpolation(x_values, y_values, x_interpolate)
print(f"Interpolasi pada x={x_interpolate}: {result}")

# Interpolasi menggunakan Metode Newton
newton_interpolated = newton_interpolation(x_values, y_values, x_interpolate)
print(f"Newton Interpolasi pada x = {x_interpolate} adalah {newton_interpolated:.4f}")


# Interpolasi menggunakan metode Spline Cubic
spline_interpolated = spline_cubic_interpolation(x_values, y_values, x_interpolate)
print(f"Spline Cubic Interpolasi pada x = {x_interpolate} adalah {spline_interpolated:.4f}")


