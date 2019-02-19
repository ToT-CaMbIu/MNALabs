#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import math as mth


def rotate(A ,B ,n):
    C = A.copy()
    for i in range(n):
        for j in range(i+1,n):
            curC = float(C[i][i])/mth.sqrt(C[i][i]*C[i][i] + C[j][i] * C[j][i])
            curS = float(C[j][i])/mth.sqrt(C[i][i]*C[i][i] + C[j][i] * C[j][i])
            temp = C.copy()
            temp[i] = [curC * C[i][k] + curS * C[j][k] for k in range(n)]
            temp[j] = [-curS * C[i][k] + curC * C[j][k] for k in range(n)]
            tempB = B.copy()
            tempB[i] = curC * B[i] + curS * B[j]
            tempB[j] = -curS * B[i] + curC * B[j]
            C = temp.copy()
            B = tempB.copy()
    S = C.T @ A @ C
    print([S[i][i] for i in range(0,n)])
    print(C)
    print(solve(C, B, n))


def solve(A,B, n):
    X = np.array([0.0]*n)
    for i in range(n):
        r = B[n - i - 1] - sum(A[n - i - 1][k] * X[k] for k in range(n - i - 1,n))
        X[n - i - 1] = r / A[n - i - 1][n - i - 1]
    return X


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

    rotate(A,B,num[0])

if __name__ == "__main__":
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
#-0.00068779  0.07125395  0.14303305  0.26042207
