#!/usr/bin/env python
# coding: utf-8

# 学习成绩与学习时间的关系

# linearregression_experiment
import os
import struct
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model  # 调用sklearn里面的linear_regression函数
from sklearn.metrics import mean_squared_error  # 预测回归结果分析工具


def load_datafromexcel(path):
    # 从dataset文件夹里面读取csv文件，为了方便起见，我们所有的文件都从dataset的文件夹里面读取,
    # 统一格式为：第0行为header行，第0列为index列，最后一列是label
    Sample_csv = pd.read_csv(path, header=0, index_col=0)
    Sample = Sample_csv.values  # 转成numpy数组
    y = Sample[:, Sample.shape[1] - 1]
    # print(y)
    X = Sample[:, 0:Sample.shape[1] - 1]
    # print(X)
    return X, y


if __name__ == '__main__':
    X_train, y_train = load_datafromexcel(path='linear_regression_train.csv')  # 读取训练集数据
    X_test, y_test = load_datafromexcel(path='linear_regression_test.csv')  # 读取测试集数据
    clf = linear_model.LinearRegression()  # 使用线性回归模型进行拟合
    # print(clf)
    clf.fit(X_train, y_train)  # 生成模型
    # print(clf.coef_)
    b = clf.coef_[0]  # 斜率
    a = clf.intercept_  # 截距
    # print(a)
    # print(b)

    # 画图
    plt.plot(X_train, y_train, 'b+')  # 画出训练数据
    xx = np.arange(0, 9, 0.01)
    yy = a + b * xx
    plt.plot(xx, yy, 'k--')  # 画出拟合曲线
    plt.plot(X_test, y_test, 'r^')  # 画出测试集数据
    plt.xlabel('Time for Study (Hours)')
    plt.ylabel('Score')
    plt.show()
