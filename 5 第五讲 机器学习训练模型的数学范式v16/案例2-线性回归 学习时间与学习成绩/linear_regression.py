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
from scipy import linalg

def load_datafromexcel(path):
    # 从dataset文件夹里面读取csv文件，为了方便起见，我们所有的文件都从dataset的文件夹里面读取,
    # 统一格式为：第0行为header行，第0列为index列，最后一列是label
    # 1.----------------------------
    df = pd.read_csv(path)
   # print(df)
    return np.array(df['time_perday']), np.array(df['score'])
    


if __name__ == '__main__':
    X_train, y_train = load_datafromexcel(path='/Users/daniel/Documents/5 第五讲 机器学习训练模型的数学范式v16/案例2-线性回归 学习时间与学习成绩/linear_regression_train.csv')  # 读取训练集数据
    X_test, y_test = load_datafromexcel(path='/Users/daniel/Documents/5 第五讲 机器学习训练模型的数学范式v16/案例2-线性回归 学习时间与学习成绩/'+'linear_regression_test.csv')  # 读取测试集数据
    clf = linear_model.LinearRegression()  # 使用线性回归模型进行拟合
    # print(clf)
    # 2.-------------------------------------------------------生成模型
    # print(clf.coef_)
    clf.fit(X_train.reshape(-1,1),y_train.reshape(-1,1))
    
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
