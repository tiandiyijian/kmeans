# -*- coding: utf-8 -*-
import numpy as np
import mat4py
import matplotlib.pyplot as plt

def distance(pointA, pointB):
    return np.linalg.norm(pointA - pointB)

def kmeans(data, k, iterations=100):
    '''
    :param data: points
    :param k: number Of classes
    :param iterations:
    :return: centroids, classes
    '''
    numOfPoints = data.shape[0]
    classes = np.zeros(numOfPoints)
    centroids = data[np.random.randint(0, numOfPoints, k), :]
    distance_matrix = np.zeros((numOfPoints, k))
    for i in range(iterations):
        preClasses = classes.copy()
        for p in range(numOfPoints):
            for q in range(k):
                distance_matrix[p, q] = distance(data[p], centroids[q])
        classes = np.argmin(distance_matrix, 1)
        for iterK in range(k):
            centroids[iterK] = np.mean(data[np.where(classes==iterK)], 0)

        print('iteration:', i+1)
        print((classes - preClasses).sum())
        if (classes - preClasses).sum() == 0:
            break
    return (centroids, classes)

if __name__ == '__main__':
    data = mat4py.loadmat('data/toy_clustering.mat')['r1']
    data = np.array(data)

    centroids, classes =  kmeans(data, 4, 5000)
    print(centroids)
    plt.scatter(data[:, 0], data[: , 1])
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red')
    plt.show()

    # a = np.array([1,1])
    # b = np.array([2,1])
    # print(distance(a, b))