# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def generatePoint(x, y, numOfPoints, rangeOfX, rangeOfY):
    res = np.zeros((numOfPoints, 2))
    for i in range(numOfPoints):
        res[i, 0] = x + np.random.uniform(-rangeOfX/2, rangeOfX/2)
        res[i, 1] = y + np.random.uniform(-rangeOfY/2, rangeOfY/2)
    return res

if __name__ == '__main__':
    res1 = generatePoint(3, 3, 16, 1, 1)
    res2 = generatePoint(5, 5, 18, 1, 1)
    res3 = generatePoint(7, 7, 16, 1, 1)
    res = np.append(res1, res2, axis=0)
    res = np.append(res, res3, axis=0)
    # np.savetxt('data/data2.csv', res, fmt='%.2f')
    plt.scatter(res[:, 0], res[:, 1])
    plt.show()