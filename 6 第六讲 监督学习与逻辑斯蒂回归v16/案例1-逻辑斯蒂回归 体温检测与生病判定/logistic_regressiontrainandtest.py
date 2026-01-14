#!/usr/bin/env python
# coding: utf-8


# logistic_regression_experiment
import os
import struct
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model  # 调用sklearn里面的linear_regression函数
from sklearn.metrics import classification_report  # 预测回归结果分析工具


def load_datafromexcel(path):
    # 0._____________________
    return X, y


if __name__ == '__main__':
    X_train, y_train = load_datafromexcel(path='logistic_regression_illness_train.csv')  # 读取训练集数据
    clf = linear_model.LogisticRegression(C=1e10)  # 使用Logistic回归模型进行拟合
    print(clf)
    # 1.______________________
    print(clf.coef_)
    a = clf.coef_[0]
    b = clf.intercept_

    # 画图
    plt.figure(1)
    plt.plot(X_train, y_train, 'b+')  # 画出原始数据
    plt.xlabel('Temperature')
    plt.ylabel('Illness')
    xx = np.arange(36, 40, 0.01)

    # 画出模型拟合
    plt.figure(2)
    plt.plot(X_train, y_train, 'b+')  #
    plt.xlabel('Temperature')
    plt.ylabel('Illness')
    xx = np.arange(36, 40, 0.01)
    zz = b + a * xx
    yy = 1 / (1 + np.exp(-zz))
    plt.plot(xx, yy, 'k-')

    # 3.________________________
    plt.plot(X_test, y_test, '^r')

    # training set的拟合准确率
    y_predict_train = clf.predict(X_train)
    print(classification_report(y_train, y_predict_train))

    # testing set的拟合准确率
    # 4.__________________________
    print(classification_report(y_test, y_predict_test))
