#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def jacobi(A, B, x):
    eps = 0.01
    D = np.diag(A)
    R = A - np.diagflat(D)
    if np.linalg.norm(R) >= 1:
        return None
    cnt = 1
    pre = x.copy()
    x = (B - np.dot(R, x)) / D
    while np.linalg.norm(x - pre) > eps:
        pre = x.copy()
        x = (B - np.dot(R, x)) / D
        cnt += 1
    print(f"k := {cnt}")
    return x


def zeidel(A, B, N):
    n = len(A)
    x = [1.0]*n

    for temp in range(N):
        x1 = np.copy(x)
        for i in range(n):
            s1 = sum(float(A[i][j]) * x1[j] for j in range(i))
            s2 = sum(float(A[i][j]) * x[j] for j in range(i + 1, n))
            x1[i] = (B[i] - s1 - s2) / float(A[i][i])
        x = x1
    return x


def checkConverge(A,n):
    for i in range(n):
        if max(A[i]) != A[i][i]:
            return False
    return True

def main():
    num = list(map(int, input().split()))
    a = []
    for i in range(0, num[0]):
        temp = list(map(float, input().split()))
        a.append(temp)

    b = []
    for i in range(0, num[0]):
        temp = float(input())
        b.append(temp)

    A = np.array(a)
    B = np.array(b)
    guess = np.array([1.0]*num[0])
    result = jacobi(A, B, guess)
    print(result if result is not None else None)
    print(zeidel(A, B, 5) if checkConverge(A,num[0]) else None)


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
