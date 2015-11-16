#!/usr/bin/env python
# encoding: utf-8

"""
@version: 1.0
@author: Juay
@file: handWritingRecognition.py
@time: 2015/11/16 16:06
"""
import numpy as np
import os
from sklearn import neighbors


def img2vector(filepath):
    vector = np.genfromtxt(filepath, dtype=np.int8, delimiter=1)
    return vector.flatten()


def gener_data_set(path_prefix):
    training_files = os.listdir(path_prefix)
    dataset = []
    labels = []
    dataset_append = dataset.append
    labels_append = labels.append
    for trf in training_files:
        dataset_append(img2vector(path_prefix + '/' + trf))
        labels_append(int(trf[0]))

    return dataset, labels


def test_data_set(path_prefix, dataSet, labels):
    testData, testLabels = gener_data_set(path_prefix)
    test_files = os.listdir(path_prefix)
    clf = neighbors.KNeighborsClassifier(algorithm='auto')
    clf.fit(dataSet, labels)
    error = 0
    clf_predict = clf.predict
    for index, t in enumerate(testData):
        pre_value = clf_predict(t)
        if pre_value[0] != testLabels[index]:
            print "Predict value is %s, the read value is %s, filePath is %s" % \
                  (pre_value[0], testLabels[index], test_files[index])
            error += 1

    print error
    print float(error) * 100.0 / len(testData)


if __name__ == "__main__":
    dataset, labels = gener_data_set('trainingDigits')
    test_data_set('testDigits', dataset, labels)
