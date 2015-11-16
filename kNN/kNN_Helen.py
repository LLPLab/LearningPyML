#!/usr/bin/env python
# encoding: utf-8

"""
@version: 1.0
@author: Juay
@file: kNN_Helen.py
@time: 2015/11/16 13:39
"""

import numpy as np
import operator


def label_tran(lable_str):
    tran_dict = {'didntLike': 0, 'smallDoses': 1, 'largeDoses': 2}
    return tran_dict[lable_str]


def file2matrix(file_path):
    fr = open(file_path)
    arrayOfLines = fr.readlines()
    numberOfLines = len(arrayOfLines)
    returnMat = np.zeros((numberOfLines, 3))
    classLabelVector = []
    for index, line in enumerate(arrayOfLines):
        line = line.strip()
        listFromline = line.split()
        returnMat[index, :] = listFromline[:-1]
        classLabelVector.append(label_tran(listFromline[-1]))

    return returnMat, classLabelVector


def autoNorm(dataSet):
    maxVals = np.max(dataSet, axis=0)
    minVals = np.min(dataSet, axis=0)
    ranges = maxVals - minVals
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    normDataSet = normDataSet / np.tile(ranges, (m, 1))
    return normDataSet


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
    trainingMat, traininglabels = file2matrix('dataSet/trainingSet.txt')
    trainingMat = autoNorm(trainingMat)

    testMat, testLabels = file2matrix('dataSet/testSet.txt')
    testMat = autoNorm(testMat)

    k = 8
    errorCount = 0.0
    for index, X in enumerate(testMat):
        predict_val = classify(X, trainingMat, traininglabels, k)
        print '%s to %s, the read answer is %s,' % (X, predict_val, testLabels[index]),
        if predict_val != testLabels[index]:
            print "Error!"
            errorCount += 1.0
        else:
            print "Correct!"
    print 'Error Rate: %%%f' % (errorCount * 100 / testMat.shape[0])
