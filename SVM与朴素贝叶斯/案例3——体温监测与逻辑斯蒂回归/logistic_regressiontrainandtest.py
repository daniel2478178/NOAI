#!/usr/bin/env python
# coding: utf-8


# logistic_regression_experiment
import os
import struct
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model #调用sklearn里面的linear_regression函数
from sklearn.metrics import classification_report  # 预测回归结果分析工具

def load_datafromexcel(path):
    #从dataset文件夹里面读取csv文件，为了方便起见，我们所有的文件都从dataset的文件夹里面读取,
    #统一格式为：第0行为header行，第0列为index列，最后一列是label
    Sample_csv = pd.read_excel(path, header = 0, index_col = 0)
    Sample = Sample_csv.values  #转成numpy数组
    y = Sample[:,Sample.shape[1]-1]
    #print(y)
    X = Sample[:,0:Sample.shape[1]-1]
    #print(X)
    return X,y

if __name__ == '__main__':
    X_train, y_train = load_datafromexcel(path = 'logistic_regression_illness_train.xlsx')  #读取训练集数据
    clf =  linear_model.LogisticRegression(C=1e10) #使用Logistic回归模型进行拟合
    print(clf)
    clf.fit(X_train,y_train) #生成模型
    print(clf.coef_)
    a = clf.coef_[0]
    b = clf.intercept_

    #画图
    plt.figure(1)
    plt.plot(X_train, y_train,'b+') #画出原始数据
    plt.xlabel('Temperature')
    plt.ylabel('Illness')
    xx = np.arange(36,40,0.01)
    zz = b + a * xx
    yy = 1/(1 + np.exp(-zz))
    plt.plot(xx,yy,'k-')

    X_test, y_test = load_datafromexcel(path = 'logistic_regression_illness_test.xlsx')  #读取测试集数据
    plt.plot(X_test,y_test,'^r')
    y_predict = clf.predict(X_test) #对测试集数据进行预测
    print(classification_report(y_test,y_predict))






