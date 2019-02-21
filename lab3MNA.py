#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import math as mth


def s(A,n):
    S = np.zeros((n, n))
    S[0][0] = mth.sqrt(A[0][0])

    for i in range(0,n):
        for j in range(i,n):
            if i == 0:
                S[i][j] = A[0][j] / S[0][0]
                continue
            if j == i:
                S[i][j] = mth.sqrt(A[i][j] - sum([S[k][i] * S[k][i] for k in range(0, i)]))
            else:
                S[i][j] = (A[i][j] - sum([S[k][i] * S[k][j] for k in range(0, i)]))/S[i][i]
    return S


def zeidel(A, B, N):
    n = len(A)
    x = [1.0]*n

    for temp in range(N):
        x_new = np.copy(x)
        for i in range(n):
            s1 = sum(float(A[i][j]) * x_new[j] for j in range(i))
            s2 = sum(float(A[i][j]) * x[j] for j in range(i + 1, n))
            x_new[i] = (B[i] - s1 - s2) / float(A[i][i])
        x = x_new
    return x


def mydet(A, n):
    ans = 1
    for i in range(n):
        ans *= A[i][i] * A[i][i]
    return np.array(ans)


def myinv(A, n):
    ans = []
    cur = [0]*n
    for i in range(n):
        cur[i] = 1
        Y = zeidel(A.T, cur, n)
        ans.append(zeidel(A, Y ,n))
        cur[i] = 0
    return np.array(ans)


def main():
    num = list(map(int, input().split()))
    a = []
    for i in range(1, num[0] + 1):
        temp = list(map(float, input().split()))
        a.append(temp)

    b = []
    for i in range(1, num[0] + 1):
        temp = float(input())
        b.append(temp)

    A = np.array(a)
    B = np.array(b)

    U = s(A,num[0])
    print(U)
    print(f"determinant - {mydet(U,num[0])}")
    Y = zeidel(U.T,B,num[0])
    X = zeidel(U, Y,num[0])
    print(f"solve - {X}")
    print(f"inv - \n {myinv(U,num[0])}")

    print(U @ U.T)


if __name__ == '__main__':
    main()
"""
4 4
3.389 0.273 0.126 0.418
0.329 2.796 0.179 0.278
0.186 0.275 2.987 0.316 
0.197 0.219 0.274 3.127
0.144
0.297
0.529
0.869
"""
