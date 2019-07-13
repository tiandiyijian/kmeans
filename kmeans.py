# -*- coding: utf-8 -*-
import numpy as np
import mat4py
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from sklearn import preprocessing

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
    data = preprocessing.MinMaxScaler().fit_transform(data)

    centroids, classes =  kmeans(data, 3, 5000)
    print(centroids)

    colors = list(colors.cnames.keys())
    colors[0] = 'red'
    colors[1] = 'blue'
    colors[2] = 'green'
    colors[3] = 'purple'
    for i in range(centroids.shape[0]):
        plt.scatter(data[np.where(classes == i), 0], data[np.where(classes == i), 1], c=colors[i], alpha=0.4)
        plt.scatter(centroids[i, 0], centroids[i, 1], c=colors[i], marker='+', s=1000)
    plt.show()