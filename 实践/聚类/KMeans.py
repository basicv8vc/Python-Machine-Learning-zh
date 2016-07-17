#-*- coding:utf-8
"""
k-means algorithm for cluster,

"""

import numpy as np
from collections import defaultdict
from random import uniform

class KMeans(object):
    """
    kmeans 聚类算法, 简单
    """
    def __init__(self):
        pass

    def cluster(self, X, k):
        self.X = X
        self.k = k
        self.centers = defaultdict(list)
        self.assignment = np.zeros(self.X.shape[0]) #记录每个样本属于哪个中心点

        self._initialize() #初始化得到k个点
        self._assign() #将数据集中每个样本分配给最近的中心点
        self.next_centers = None
        while self.centers != self.next_centers:
            self.next_centers = self.centers
            self._getcenters()
            self._assign()
        return self.assignment

    def _initialize(self):
        """
        初始化，即初始化k个中心点和每个样本属于哪个中心点
        这k个样本点是随机产生的
        """
        feature_min_max = defaultdict(list) #保存每个特征值的最小值和最大值
        feature_dimensions = self.X.shape[1]
        get_min_max = lambda x, y: (x,y) if x<y else (y,x)
        for i in xrange(feature_dimensions):
            i_min, i_max = get_min_max(self.X[0][i], self.X[1][i])
            for j in range(self.X.shape[0])[2:-1:2]:
                tmp_min, tmp_max = get_min_max(self.X[j][i], self.X[j+1][i])#self.X[j+1][i], self.X[j][i] if self.X[j][i] > self.X[j+1][i] else self.X[j][i], self.X[j+1][i]
                if tmp_min < i_min:
                    i_min = tmp_min
                if tmp_max > i_max:
                    i_max = tmp_max
            tmp_min, tmp_max = get_min_max(self.X[-1][i], self.X[-2][i])#self.X[-1][i], self.X[-2][i] if self.X[-2][i] > self.X[-1][i] else self.X[-2][i], self.X[-1][i]
            i_min = tmp_min if tmp_min < i_min else i_min
            i_max = tmp_max if tmp_max > i_max else i_max
            feature_min_max[i] = [i_min, i_max]
        for i in xrange(self.k):
            this_k = []
            for j in xrange(feature_dimensions):
                value = uniform(feature_min_max[j][0], feature_min_max[j][1])
                this_k.append(value)
            self.centers[i] = this_k

    def _distance(self, point1, point2):
        """
        计算点point1 和 point2 之间的欧氏距离
        """
        dd = 0
        for i in xrange(len(point1)):
            dd += (point1[i] - point2[i]) ** 2
        return np.sqrt(dd)

    def _assign(self):

        feature_dimensions = self.X.shape[1]

        for i in xrange(self.X.shape[0]):
            min_distance = float("inf")
            current_assignment = self.k
            for j in xrange(self.k):
                tmp_d = self._distance(self.X[i], self.centers[j])
                if tmp_d < min_distance:
                    min_distance = tmp_d
                    current_assignment = j
            self.assignment[i] = current_assignment

    def _getcenters(self):
        """
        计算每个中心点的平均值，作为新的中心点
        """
        cluster_numbers = defaultdict(int) #记录每个分类的样本数目
        cluster_sum = np.zeros((self.k, len(self.X[0]))) #记录每个分类的所有样本相加之和
        for index, center in enumerate(self.assignment):
            cluster_numbers[center] = cluster_numbers.get(center, 0) + 1
            cluster_sum[center] += self.X[index]
        for i in xrange(self.k):
            self.centers[i] = list(cluster_sum[i]/float(cluster_numbers[i]))
