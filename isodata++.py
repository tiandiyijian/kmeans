# -*- coding: utf-8 -*-
import numpy as np
import mat4py
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from sklearn import preprocessing
import sys

def distance(pointA, pointB):
    return np.linalg.norm(pointA - pointB)

def getMinDistance(point, centroids, numOfcentroids):
    min_distance = sys.maxsize
    for i in range(numOfcentroids):
        dist = distance(point, centroids[i])
        if dist < min_distance:
            min_distance = dist
    return min_distance

def split(data, k, centroids, currentClasses, minimumN, maximumStd):
    m, n = centroids.shape
    std_matrix = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            std_matrix[i, j] = np.std(data[currentClasses==i, j], ddof=1)
    classesStd = np.max(std_matrix, axis=1)
    print('split', classesStd)
    for i in range(m):
        if classesStd[i] > maximumStd and np.sum(currentClasses==i) > 2 * minimumN:
            centroids[i] = centroids[i] + maximumStd
            newCentroid = centroids[i] - maximumStd
            newCentroid = newCentroid[np.newaxis, :]
            centroids = np.append(centroids, newCentroid, axis=0)
            print('add success')
            # print(centroids.shape)
            if centroids.shape[0] >= 2 * k:
                break
    distance_matrix = np.zeros((data.shape[0], centroids.shape[0]))
    for p in range(data.shape[0]):
        for q in range(centroids.shape[0]):
            distance_matrix[p, q] = distance(data[p], centroids[q])
    classes = np.argmin(distance_matrix, 1)
    return centroids, classes

def combine(data, k, centroids, currentClasses, minimumDist):
    numOfCentroids = centroids.shape[0]
    distance_matrix = np.zeros((numOfCentroids, numOfCentroids))

    # 计算每个中心到其它中心的距离
    for i in range(numOfCentroids):
        for j in range(i+1, numOfCentroids):
            distance_matrix[i, j] = distance(centroids[i], centroids[j])
    print(distance_matrix)
    for i in range(numOfCentroids):
        shouldStop = False
        for j in range(i+1, numOfCentroids):
            # if centroids.shape[0] <= k / 2 or centroids.shape[0] <= 2:
            if centroids.shape[0] <= k / 2 :
                break
            if distance_matrix[i, j] < minimumDist:
                n1 = np.sum(currentClasses==i)
                n2 = np.sum(currentClasses==j)
                print(centroids.shape, i, j)
                centroids[i] = (1/(n1+n2)) * (n1 * centroids[i] + n2 * centroids[j])
                centroids =  np.delete(centroids, j, axis=0)
                print('combine')
                shouldStop = True
        if shouldStop:
            break

    distance_matrix = np.zeros((data.shape[0], centroids.shape[0]))
    for p in range(data.shape[0]):
        for q in range(centroids.shape[0]):
            distance_matrix[p, q] = distance(data[p], centroids[q])
    classes = np.argmin(distance_matrix, 1)
    return centroids, classes

def isolandpp(data, k, iterations=100, minimumN=15, maximumStd=0.2, minimumDist=0.2):
    '''
    :param data: points
    :param k: number Of classes
    :param iterations:
    :return: centroids, classes
    '''
    numOfPoints = data.shape[0]
    classes = np.zeros(numOfPoints)
    centroids = np.zeros((k, data.shape[1]))
    centroids[0] = data[np.random.randint(0, numOfPoints)]
    distanceArray = [0] * numOfPoints
    for iterK in range(1, k):
        total = 0.0
        for i, point in enumerate(data):
            distanceArray[i] = getMinDistance(point, centroids, iterK)
            total += distanceArray[i]
        total *= np.random.rand()
        for i, di in enumerate(distanceArray):
            total -= di
            if total > 0:
                continue
            centroids[iterK] = data[i]
            break

    print(centroids)

    for i in range(iterations):
        preClasses = classes.copy()
        preCentroids = centroids.copy()
        distance_matrix = np.zeros((numOfPoints, centroids.shape[0]))
        for p in range(numOfPoints):
            for q in range(centroids.shape[0]):
                distance_matrix[p, q] = distance(data[p], centroids[q])
        classes = np.argmin(distance_matrix, 1)
        # for iterK in range(centroids.shape[0]):
        #     centroids[iterK] = np.mean(data[np.where(classes==iterK)], 0)

        iter1 = 0
        while True:
            if np.sum(classes==iter1) < minimumN:
                centroids = np.delete(centroids, iter1, axis=0)
                distance_matrix = np.delete(distance_matrix, iter1, axis=1)
                classes = np.argmin(distance_matrix, 1)
                print('delete centroid')
                # for iterK in range(centroids.shape[0]):
                #     centroids[iterK] = np.mean(data[np.where(classes == iterK)], 0)
            else:
                iter1 += 1
            if iter1 == centroids.shape[0] - 1:
                break

        for iterK in range(centroids.shape[0]):
            centroids[iterK] = np.mean(data[np.where(classes == iterK)], 0)

        # too few classes
        # if centroids.shape[0] < k/2:
        centroids, classes = split(data, k, centroids, classes, minimumN, maximumStd)
        # too many classes
        # elif centroids.shape[0] > k*2:
        if centroids.shape[0] <= preCentroids.shape[0]:
            centroids, classes = combine(data, k, centroids, classes, minimumDist)

        print('iteration:', i+1)
        print('centroids:', centroids.shape[0])

        if  preCentroids.size == centroids.size and np.abs(centroids - preCentroids).sum() == 0\
                and (classes - preClasses).sum() == 0:
            break
    return (centroids, classes)

if __name__ == '__main__':
    # data = mat4py.loadmat('data/toy_clustering.mat')['r3']
    data = np.loadtxt('data/data1.csv')
    data = np.array(data)
    data = preprocessing.MinMaxScaler().fit_transform(data)

    centroids, classes =  isolandpp(data, 4, 5000, minimumN=10, maximumStd=0.15, minimumDist=0.3)
    print('calsses:', classes)
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