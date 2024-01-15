import numpy as np
from scipy.interpolate import lagrange, CubicSpline

# Metode Newton
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

    for i in range(1, len(coeffs)):
        temp *= (x_interpolate - x_values[i - 1])
        result += coeffs[i] * temp

    return result


# Data
x_values = np.array([100, 200, 300, 400, 500])
y_values = np.array([11, 13, 23, 25, 27])

# Pekerja yang akan diinterpolasi
x_interpolate = 350

# Metode Newton
newton_interpolated = newton_interpolation(x_values, y_values, x_interpolate)
print(f"Newton Interpolasi pada x={x_interpolate}: {newton_interpolated:.4f}")

# Metode Lagrange
lagrange_poly = lagrange(x_values, y_values)
lagrange_interpolated = lagrange_poly(x_interpolate)
print(f"Lagrange Interpolasi pada x={x_interpolate}: {lagrange_interpolated:.4f}")

# Metode Spline Cubic
spline_cubic = CubicSpline(x_values, y_values)
spline_interpolated = spline_cubic(x_interpolate)
print(f"Spline Cubic Interpolasi pada x={x_interpolate}: {spline_interpolated:.4f}")
