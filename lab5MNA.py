#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import math as mth
import sympy as smp
import matplotlib.pyplot as plt


def f(x):
    return 4 * (1 + mth.sqrt(x)) * mth.log(x) - 1


def chr(xPrev=0.1, xCur=12.0, eps=0.001):
    xNext = 10 ** (-9)
    while abs(xNext - xCur) > eps:
        tmp = xNext
        xNext = xPrev - (f(xPrev) * (xCur - xPrev) / (f(xCur) - f(xPrev)))
        xPrev = xCur
        xCur = tmp
    return xNext


def newton(x0=0.1, eps=0.001):
    xPrev = x0
    x = smp.symbols('x')
    curF = 4 * (1 + smp.sqrt(x)) * smp.log(x) - 1
    tmp = curF.diff(x)
    curD = tmp.subs({x: xPrev}).n()
    xCur = xPrev - f(xPrev) / curD
    while abs(xCur - xPrev) > eps:
        xPrev = xCur
        x = smp.symbols('x')
        tmp = curF.diff(x)
        curD = tmp.subs({x: xPrev}).n()
        xCur = xPrev - f(xPrev) / curD
    return xCur


def iterations(xPrev=0.1, yPrev=1.6, eps=0.001):
    xNext = (1 - mth.sin(xPrev + yPrev)) / 1.2
    yNext = mth.sqrt(1 - xPrev ** 2)
    while abs(yNext - yPrev) > eps or abs(xNext - xPrev) > eps:
        xPrev = xNext
        yPrev = yNext
        xNext = (1 - mth.sin(xPrev + yPrev)) / 1.2
    return (xNext, yNext)


def jacobian(x, y, n=2):
    A = np.zeros((n, n))
    A[0][0] = np.cos(x + y) + 1.2
    A[0][1] = np.cos(x + y)
    A[1][0] = 2 * x
    A[1][1] = 2 * y
    return A


def func_s(x, y):
    A = np.zeros(2)
    A[0] = np.sin(x + y) + 1.2 * x - 1
    A[1] = x ** 2 + y ** 2 - 1
    return A


def newton_s(flag, n=0.1, eps=0.01):
    prevV = np.array([n, n])
    curV = prevV - np.linalg.inv(jacobian(prevV[0], prevV[1])) @ func_s(prevV[0], prevV[1])
    chkX = prevV[0]
    chkY = prevV[1]

    while abs(np.linalg.norm(curV - prevV)) > eps:
        prevV = curV
        if flag == 0:
            chkX = prevV[0]
            chkY = prevV[1]
        curV = prevV - np.linalg.inv(jacobian(chkX, chkY)) @ func_s(prevV[0], prevV[1])

    return curV


def main():
    x = np.arange(0.1, 6.0, 0.1)
    print(chr())
    print(newton())
    print(iterations())
    print(newton_s(0))
    print(newton_s(1, n=0.9))
    plt.plot(x, 4 * (1 + np.sqrt(x)) * np.log(x) - 1)
    plt.show()

    delta = 0.025
    x, y = np.meshgrid(
        np.arange(-2, 2, delta),
        np.arange(-5, 5, delta)
    )
    plt.contour(
        x, y,
        np.sin(x + y) + 1.2 * x,
        [1]
    )
    plt.contour(
        x, y,
        x ** 2 + y ** 2,
        [1]
    )
    plt.show()


if __name__ == '__main__':
    main()

"""
Задание:
НУ:
4*(1 + x^(1/2))*ln(x) - 1 = 0
Система НУ
sin(x + y) + 1.2x = 1
x^2 + y^2 = 1
"""
