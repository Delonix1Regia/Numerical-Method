import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable

def f(x):
    return 500 - x**3 + 10*x

def regula_falsi(a, b, tol, max_iter):
    table = PrettyTable()
    table.field_names = ["Iterasi", "a", "b", "c", "f(c)", "Error Relatif"]

    iterasi = 0
    errors = []

    while (b-a)/2 > tol and iterasi < max_iter:
        iterasi += 1
        c=(a*f(b)-b*f(a))/(f(b)-f(a))

        if f(c)==0:
            break
        elif f(c)*f(a)<0:
            b=c
        else:
            a=c

        error_rel = np.abs((b-a)/2/c)
        errors.append(error_rel)
        table.add_row([iterasi, round(a, 4), round(b, 4), round(c, 4), round(f(c), 4), "{:4e}".format(error_rel)])


    print(table)
    return c, errors

a = 0
b = 10
tol = 0.01
max_iter = 100

solusi, errors = regula_falsi(a, b, tol, max_iter)
print("\nNilai x yang memenuhi f(x) = 0 adalah ", round(solusi, 4))

plt.plot(range(1, len(errors) + 1), errors, marker = 'o')
plt.yscale('log')
plt.xlabel('Iterasi')
plt.ylabel('Error relatif')
plt.title('Grafik Error Relatif pada Metode Regula Falsi')
plt.show()