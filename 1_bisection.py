import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable

def f(x):
    return 500 - x**3 + 10*x

def bisection(a, b, tol, max_iter):
    table = PrettyTable()
    table.field_names = ["Iterasi", "a", "b", "c", "f(c)", "Error Relatif"]

    iterasi = 0
    errors = []

    while (b-a)/2 > tol and iterasi < max_iter:
        iterasi += 1
        c = (a+b)/2
        f_c = f(c)
        error_rel = np.abs((b-a)/2/c)
        errors.append(error_rel)
        table.add_row([iterasi, round(a, 4), round(b, 4), round(c, 4), round(f_c, 4), "{:4e}".format(error_rel)])

        if f_c == 0:
            break
        elif f_c * f(a) < 0:
            b = c
        else:
            a = c

    print(table)
    return c, errors

a = 0
b = 10
tol = 0.01
max_iter = 100

solusi, errors = bisection(a, b, tol, max_iter)
print("\nNilai x yang memenuhi f(x) = 0 adalah ", round(solusi, 4))

plt.plot(range(1, len(errors) + 1), errors, marker = 'o')
plt.yscale('log')
plt.xlabel('Iterasi')
plt.ylabel('Error relatif')
plt.title('Grafik Error Relatif pada Metode Bisection')
plt.show()