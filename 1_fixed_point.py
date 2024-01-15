import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable

def f(x):
    return 500 - x**3 + 10*x

def g(x):
    return x - (f(x)/(10*x-500))

def fixed_point(guess, tol, max_iter):
    table = PrettyTable()
    table.field_names = ["Iterasi", "x", "g(x)", "Error Relatif"]

    iterasi = 0
    errors = []
    x=guess

    while iterasi < max_iter:
        iterasi += 1
        x_new = g(x)

        error_rel = np.abs((x_new - x)/x_new)
        errors.append(error_rel)
        table.add_row([iterasi, round(x, 4), round(x_new, 4), "{:4e}".format(error_rel)])

        if error_rel < tol:
            break
        
        x = x_new

    print(table)
    return x, errors

guess = 0.5

tol = 0.01
max_iter = 100

solusi, errors = fixed_point(guess, tol, max_iter)
print("\nNilai x yang memenuhi f(x) = 0 adalah ", round(solusi, 4))

plt.plot(range(1, len(errors) + 1), errors, marker = 'o')
plt.yscale('log')
plt.xlabel('Iterasi')
plt.ylabel('Error relatif')
plt.title('Grafik Error Relatif pada Metode Fixed Point')
plt.show()