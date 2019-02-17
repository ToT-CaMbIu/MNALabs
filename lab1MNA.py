#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np


def gauss(matrix, n):
    for i in range(0,n):
        temp = matrix[i][i]
        matrix[i] = [x / temp for x in matrix[i]]
        for j in range(0,n):
            if j == i:
                continue
            else:
                tempList = matrix[i]
                tempList = [x * (-matrix[j][i]) for x in tempList]
                matrix[j] = [x + y for x, y in zip(tempList, matrix[j])]
        print(f"{np.array(matrix)} \n")


def main():
    num = list(map(int, input().split()))
    b = []
    c = []
    fr = []
    ans = [0] * num[1]
    for i in range(0, num[0]):
        a = list(map(float, input().split()))
        l = []
        temp = a[len(a) - 1]
        while len(a) > num[0]:
            a.pop(len(a) - 1)
        a.append(temp)
        fr.append(temp)
        l = a.copy()
        l.pop(len(l) - 1)
        b.append(a)
        c.append(l)

    c = np.array(c)

    if np.linalg.det(c) != 0:
        print("Step by step solve:")
        gauss(b,num[0])
        for i in range(0,num[0]):
            ans[i] = b[i][num[0]]
        print(f"Solve of linear system - {ans}")
    else:
        print("Not any solutions!!!")
        return
    if num[0] == num[1]:
        inverse = np.linalg.inv(np.array(c))
        print(f"Inverse Matrix \n {inverse}")
    else:
        print("Cannot find inverse matrix!!!")
    absolute = np.linalg.norm(inverse,np.inf)*0.001
    relative = np.linalg.norm(inverse,np.inf)*np.linalg.norm(c,np.inf)*(0.001/np.linalg.norm(np.array(fr),np.inf))
    print(f"Absolute Error - {absolute}")
    print(f"Relative Error - {relative}")

    #mt = np.array([[3.389,0.273,0.126,0.418,1,0,0,0],[0.329,2.796,0.179,0.278,0,1,0,0],
    #[0.186,0.275,2.987,0.316,0,0,1,0],[0.197,0.219,0.274,3.127,0,0,0,1]]
    #)

    #gauss(mt,4)


if __name__ == '__main__':
    main()


"""
4 4
3.389 0.273 0.126 0.418 0.144
0.329 2.796 0.179 0.278 0.297
0.186 0.275 2.987 0.316 0.529
0.197 0.219 0.274 3.127 0.869
"""