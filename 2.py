import numpy as np
import matplotlib.pyplot as plt

# Fungsi kecepatan
def v(t, n):
    return 50 + n * t - 5 * t**2

# Metode Bisection
def bisection(func, a, b, tol, max_iter, n):
    iterasi = 0
    error_list = []
    while iterasi < max_iter:
        c = (a + b) / 2
        if func(c, n) == 0 or (b - a) / 2 < tol:
            return c, error_list
        iterasi += 1
        error = np.abs((b - a) / 2)
        error_list.append(error)
        if np.sign(func(c, n)) == np.sign(func(a, n)):
            a = c
        else:
            b = c
    return None, error_list

# Metode Regula Falsi
def regula_falsi(func, a, b, tol, max_iter, n):
    iterasi = 0
    error_list = []
    while iterasi < max_iter:
        c = b - (func(b, n) * (b - a)) / (func(b, n) - func(a, n))
        if func(c, n) == 0 or np.abs(func(c, n)) < tol:
            return c, error_list
        iterasi += 1
        error = np.abs(func(c, n))
        error_list.append(error)
        if np.sign(func(c, n)) == np.sign(func(a, n)):
            a = c
        else:
            b = c
    return None, error_list

# Metode Newton-Raphson
def newton_raphson(func, func_derivative, x0, tol, max_iter, n):
    iterasi = 0
    error_list = []
    while iterasi < max_iter:
        x1 = x0 - func(x0, n) / func_derivative(x0, n)
        if np.abs(x1 - x0) < tol or func(x1, n) == 0:
            return x1, error_list
        iterasi += 1
        error = np.abs((x1 - x0) / x1)
        error_list.append(error)
        x0 = x1
    return None, error_list

# Metode Secant
def secant(func, x0, x1, tol, max_iter, n):
    iterasi = 0
    error_list = []
    while iterasi < max_iter:
        x2 = x1 - (func(x1, n) * (x1 - x0)) / (func(x1, n) - func(x0, n))
        if np.abs(x2 - x1) < tol or func(x2, n) == 0:
            return x2, error_list
        iterasi += 1
        error = np.abs((x2 - x1) / x2)
        error_list.append(error)
        x0 = x1
        x1 = x2
    return None, error_list

# Nilai dari 𝑛
n = 10

# Batas awal
a = 0
b = 10

# Toleransi dan jumlah iterasi maksimum
toleransi = 0.01
max_iterasi = 100

# Metode Bisection
hasil_bisection, error_bisection = bisection(v, a, b, toleransi, max_iterasi, n)
print("Metode Bisection:", hasil_bisection)

# Metode Regula Falsi
hasil_regula_falsi, error_regula_falsi = regula_falsi(v, a, b, toleransi, max_iterasi, n)
print("Metode Regula Falsi:", hasil_regula_falsi)

# Metode Newton-Raphson
def v_derivative(t, n):
    return n - 10 * t

x_awal_newton = 5  # Nilai awal yang dekat dengan akar
hasil_newton_raphson, error_newton_raphson = newton_raphson(v, v_derivative, x_awal_newton, toleransi, max_iterasi, n)
print("Metode Newton-Raphson:", hasil_newton_raphson)

# Metode Secant
x0_secant = 0
x1_secant = 10
hasil_secant, error_secant = secant(v, x0_secant, x1_secant, toleransi, max_iterasi, n)
print("Metode Secant:", hasil_secant)

# Grafik error untuk setiap metode
iterasi = np.arange(1, min(len(error_bisection), len(error_regula_falsi), len(error_newton_raphson), len(error_secant)) + 1)

plt.figure(figsize=(10, 6))
plt.plot(iterasi, error_bisection[:len(iterasi)], label='Bisection')
plt.plot(iterasi, error_regula_falsi[:len(iterasi)], label='Regula Falsi')
plt.plot(iterasi, error_newton_raphson[:len(iterasi)], label='Newton-Raphson')
plt.plot(iterasi, error_secant[:len(iterasi)], label='Secant')
plt.yscale('log')
plt.xlabel('Iterasi')
plt.ylabel('Error (log scale)')
plt.title('Grafik Error untuk Setiap Metode')
plt.legend()
plt.grid(True)
plt.show()