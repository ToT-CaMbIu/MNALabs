#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import math as mth
import sympy as sp
import matplotlib.pyplot as plt

x, y = sp.symbols('x y')

A = x ** 2 * sp.ln(x)
func = sp.lambdify(x, A)
a_S = 1
b_S = 2

diff_A = 2*x**3*y**3-2*x*y
#diff_A = x * y ** 2 + y
diff_func = sp.lambdify((x, y), diff_A)
diff_aS = 0
diff_bS = 1
y0 = 1
x0 = 0


def trapez(func, a, b, n):
    h = (b - a) / n
    return h * sum([(func(a + h * i) + func(a + h * (i + 1))) / 2 for i in range(n)])


def simpson(func, a, b, n):
    h = (b - a) / n
    return h / 3 * sum(
        [func(a + h * i) + 4 * func(a + h * (i + 1)) + func(a + h * (i + 2)) for i in range(0, n - 1, 2)])


def newton_leibniz(A, a, b):
    h = sp.integrate(A, x)
    return h.subs(x, b) - h.subs(x, a)


def error_estimation(method, func, a, b, m, eps):
    n = 1
    while True:
        print(abs(method(func, a, b, n * 2) - method(func, a, b, n)))
        if abs(method(func, a, b, 2 * n) - method(func, a, b, n)) / (2 ** m - 1) < eps:
            break
        n *= 2
    return n * 2


def eiler(func, a, b, x0, f, n):
    h = (b - a) / n
    x = np.empty(n + 1, float)
    y = np.empty(n + 1, float)
    x[0] = x0
    y[0] = f
    for i in range(n):
        y[i + 1] = y[i] + h * func(x[i], y[i])
        x[i + 1] = x[i] + h
    return x, y


def runge(func, a, b, x0, _, n):
    h = (b - a) / n
    x = np.empty(n + 1, float)
    y = np.empty(n + 1, float)
    x[0] = x0
    y[0] = y0
    for i in range(n):
        F1 = h * func(x[i], y[i])
        F2 = h * func(x[i] + h / 2, y[i] + F1 / 2)
        F3 = h * func(x[i] + h / 2, y[i] + F2 / 2)
        F4 = h * func(x[i] + h, y[i] + F3)
        y[i + 1] = y[i] + 1 / 6 * (F1 + F4 + 2 * (F2 + F3))
        x[i + 1] = x[i] + h
    return x, y


def adams(f, a, b, x0, yf, n):
    h = (b - a) / n
    x = np.empty(n + 1)
    y = np.empty(n + 1)
    x[0] = x0
    y[0] = y0
    x[1] = x[0] + h
    y[1] = y[0] + h * f(x[0], y[0])
    for i in range(1, n):
        p = y[i] + h / 2 * (3 * f(x[i], y[i]) - f(x[i - 1], y[i - 1]))
        x[i + 1] = x[i] + h
        y[i + 1] = y[i] + h / 2 * (f(x[i], y[i]) + f(x[i + 1], p))
    return x, y


def de_error_estimation(ode_method, func, a, b, x0, y0, _, eps):
    n = 2
    while True:
        k, y1 = ode_method(func, a, b, x0, y0, n)
        k, y2 = ode_method(func, a, b, x0, y0, n//2)
        if abs(y2[-1] - y1[-1]) < eps:
            break
        n *= 2
    return 2 * n


def main():
    n = error_estimation(trapez, func, a_S, b_S, 2, 0.0001)
    print(n)
    print('trapezium:')
    t = trapez(func, a_S, b_S, n)
    print(t)
    t2 = trapez(func, a_S, b_S, n // 2)
    print(t2)
    print('simpson:')
    s = simpson(func, a_S, b_S, n)
    print(s)
    s2 = simpson(func, a_S, b_S, n // 2)
    print(s2)
    print('newton:')
    n = newton_leibniz(A, a_S, b_S)
    print(n)
    print('sympy:')
    sym = sp.integrate(A, (x, a_S, b_S))
    print(sym)

    nd = de_error_estimation(runge, diff_func, diff_aS, diff_bS, x0, y0, 4, 0.0001)
    print(nd)

    runge_x, runge_y = runge(diff_func, diff_aS, diff_bS, x0, y0, nd)
    runge_x_2, runge_y_2 = runge(diff_func, diff_aS, diff_bS, x0, x0, nd // 2)
    plt.plot(runge_x, runge_y, label='h')
    #plt.plot(runge_x_2, runge_y_2, label='2 * h')
    plt.grid()
    #plt.show()
    print("x[i]:")
    print(runge_x)
    print("y[i]:")
    print(runge_y)
    print("~x[i]")
    print([runge_x_2[i // 2] if i % 2 == 0 else None for i in range(len(runge_x))])
    print("~y[i]")
    print([runge_y_2[i // 2] if i % 2 == 0 else None for i in range(len(runge_y))])
    print("delta[i]")
    print([abs(runge_y_2[i // 2] - runge_y[i]) if i % 2 == 0 else None for i in range(len(runge_x))])

    adams_x, adams_y = adams(diff_func, diff_aS, diff_bS, x0, y0, nd)
    adams_x_2, adams_y_2 = adams(diff_func, diff_aS, diff_bS, x0, y0, nd // 2)
    plt.plot(adams_x, adams_y, label='h')
    #plt.plot(adams_x_2, adams_y_2, label='2 * h')
    plt.grid()
    #plt.show()
    print("x[i]:")
    print(adams_x)
    print("y[i]:")
    print(adams_y)
    print("~x[i]")
    print([adams_x_2[i // 2] if i % 2 == 0 else None for i in range(len(adams_x))])
    print("~y[i]")
    print([adams_y_2[i // 2] if i % 2 == 0 else None for i in range(len(adams_y))])
    print("delta[i]")
    print([abs(adams_y_2[i // 2] - adams_y[i]) if i % 2 == 0 else None for i in range(len(adams_x))])

    eiler_x, eiler_y = eiler(diff_func, diff_aS, diff_bS, x0, y0, nd)
    eiler_x_2, eiler_y_2 = eiler(diff_func, diff_aS, diff_bS, x0, y0, nd // 2)
    plt.plot(eiler_x, eiler_y, label='h')
    #plt.plot(eiler_x_2, eiler_y_2, label='2 * h')
    plt.grid()
    #plt.show()
    print("x[i]:")
    print(eiler_x)
    print("y[i]:")
    print(eiler_y)
    print("~x[i]")
    print([eiler_x_2[i // 2] if i % 2 == 0 else None for i in range(len(eiler_x))])
    print("~y[i]")
    print([eiler_y_2[i // 2] if i % 2 == 0 else None for i in range(len(eiler_y))])
    print("delta[i]")
    print([abs(eiler_y_2[i // 2] - eiler_y[i]) if i % 2 == 0 else None for i in range(len(eiler_x))])

    f = sp.Function('f')
    solve = sp.dsolve(sp.Eq(sp.diff(f(x), x) + 2*x*f(x) - 2*x**3 * f(x) ** 3), f(x))
    print(solve)
    solution = sp.lambdify(x, 1/sp.sqrt((2*x**2+1)*sp.exp(-x**2)*sp.exp(2*x**2)))
    xL = np.linspace(diff_aS, diff_bS)
    #plt.plot(xL, solution(xL))
    plt.grid()
    plt.show()





if __name__ == '__main__':
    main()
