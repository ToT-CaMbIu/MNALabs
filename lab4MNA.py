#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import math as mth


def rotate(A , n , total : int):
    Ufinal = np.eye(n)
    for _ in range(total):
        for i in range(n):
            for j in range(i+1,n):
                U = np.eye(n)
                if A[i][i] == A[j][j]:
                    cur = mth.pi/4
                else:
                    cur = 0.5 * mth.atan(2 * A[i][j] / (A[j][j] - A[i][i]))
                s = mth.sin(cur)
                c = mth.cos(cur)
                U[i][i] = U[j][j] = c
                U[j][i] = -s
                U[i][j] = s
                A = U.T @ A @ U
                Ufinal = Ufinal @ U
    print([A[i][i] for i in range(0,n)])
    print(Ufinal)


def main():
    num = list(map(int, input().split()))
    a = []
    for i in range(1, num[0] + 1):
        temp = list(map(float, input().split()))
        a.append(temp)

    A = np.array(a)

    rotate(A,num[0],25)


if __name__ == "__main__":
    main()
"""
4 4
3.389 0.273 0.126 0.418
0.329 2.796 0.179 0.278
0.186 0.275 2.987 0.316 
0.197 0.219 0.274 3.127
"""
