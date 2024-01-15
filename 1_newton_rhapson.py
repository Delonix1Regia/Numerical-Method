import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable

def f(x):
    return 500 - x**3 + 10*x

def df(x):
    return -3*x**2 + 4

def newton_raphson(guess, tol, max_iter):
    table = PrettyTable()
    table.field_names = ["Iterasi", "x", "f(x)", "f'(x)", "Error Relatif"]

    iterasi = 0
    errors = []
    x = guess

    while iterasi < max_iter:
        iterasi += 1
        f_x = f(x)
        df_x = df(x)

        if df_x == 0:
            print("Metode Newton Raphson tidak konvergen")
            return None, None
        
        x_new = x - f_x / df_x

        error_rel = np.abs((x_new - x) / x_new)
        errors.append(error_rel)
        table.add_row([iterasi, round(x, 4), round(f_x, 4), round(df_x, 4), "{:4e}".format(error_rel)])

        if error_rel < tol:
            break
        
        x = x_new

    print(table)
    return x, errors

guess = 1.5
tol = 0.01
max_iter = 100

solusi, errors = newton_raphson(guess, tol, max_iter)
if solusi is not None:
    print("\nNilai x yang memenuhi f(x) = 0 adalah ", round(solusi, 4))

plt.plot(range(1, len(errors) + 1), errors, marker='o')
plt.yscale('log')
plt.xlabel('Iterasi')
plt.ylabel('Error relatif')
plt.title('Grafik Error Relatif pada Metode Newton Raphson')
plt.show()