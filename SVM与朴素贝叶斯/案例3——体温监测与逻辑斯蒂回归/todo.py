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
    df = pd.read_excel(path, header = 0, index_col = 0)
    X = df[df.columns[1:-1]]
    y = df[df.columns[-1]]

    return X,y

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    clf =  linear_model.LogisticRegression(C=1e10) #使用Logistic回归模型进行拟合
    X_train, y_train = load_datafromexcel(path = 'logistic_regression_illness_train.xlsx')  #读取训练集数据
    print(X_train,y_train)
    clf()
    






