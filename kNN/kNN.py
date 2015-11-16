#!/usr/bin/env python
# encoding: utf-8

"""
@version: 1.0
@author: Juay
@file: kNN.py
@time: 2015/11/16 13:03
"""

import numpy as np
import operator


def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    distances = sqDiffMat.sum(axis=1) ** 0.5
    # noinspection PyUnresolvedReferences
    sortedIndicies = distances.argsort()
    classCount = {}
    for i in xrange(k):
        voteIlabel = labels[sortedIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

    maxClass = max(classCount.iteritems(), key=operator.itemgetter(1))
    return maxClass[0]


if __name__ == '__main__':
    group, lables = createDataSet()
    print classify(np.array([0.5, 0]), group, lables, 3)
    print classify(np.array([1, 2]), group, lables, 3)
