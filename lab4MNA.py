#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import math as mth


def s(A, n):
    S = np.zeros((n, n))
    S[0][0] = mth.sqrt(A[0][0])

    for i in range(0, n):
        for j in range(i, n):
            if i == 0:
                S[i][j] = A[0][j] / S[0][0]
                continue
            if j == i:
                S[i][j] = mth.sqrt(A[i][j] - sum([S[k][i] * S[k][i] for k in range(0, i)]))
            else:
                S[i][j] = (A[i][j] - sum([S[k][i] * S[k][j] for k in range(0, i)])) / S[i][i]
    return S


def findmax(A, n):
    max = 0
    ifinal = 0
    jfinal = 0
    for i in range(n):
        for j in range(n):
            if A[i][j] > max and i != j:
                ifinal = i
                jfinal = j
                max = A[i][j]
    return (ifinal, jfinal, max)


def rotate(A, n):
    Ufinal = np.eye(n)
    max = 10 ** (9)
    while max >= 10 ** (-9):
        i, j, curMax = findmax(A, n)
        max = curMax
        U = np.eye(n)
        if A[i][i] == A[j][j]:
            cur = mth.pi / 4
        else:
            cur = 0.5 * mth.atan(2 * A[i][j] / (A[j][j] - A[i][i]))
        s = mth.sin(cur)
        c = mth.cos(cur)
        U[i][i] = U[j][j] = c
        U[j][i] = -s
        U[i][j] = s
        A = U.T @ A @ U
        Ufinal = Ufinal @ U
    print([A[i][i] for i in range(0, n)])
    print(Ufinal)


def main():
    num = list(map(int, input().split()))
    a = []
    for i in range(1, num[0] + 1):
        temp = list(map(float, input().split()))
        a.append(temp)

    A = np.array(a)

    A = A @ A.T

    U = s(A, num[0]) @ s(A, num[0]).T

    print(U)

    rotate(U, num[0])
    print(np.linalg.eig(U))


if __name__ == "__main__":
    main()

"""
4 4
3.389 0.273 0.126 0.418
0.329 2.796 0.179 0.278
0.186 0.275 2.987 0.316 
0.197 0.219 0.274 3.127
"""
