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
    print(df)
    X = df["temperature"].values.reshape(-1, 1)
    y = df["illness"].values

    return X,y

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    clf =  linear_model.LogisticRegression(C=1e10) #使用Logistic回归模型进行拟合
    X_train, y_train = load_datafromexcel(path = 'logistic_regression_illness_train.xlsx')  #读取训练集数据
    print(f"X_train{X_train},y_train{y_train}")
    clf.fit(X_train,y_train) #不管算法怎么执行的，直接进行模型建立
    X_test, y_test = load_datafromexcel(path = 'logistic_regression_illness_test.xlsx')  #读取测试集数据
    print(X_test,y_test)
    y_predict = clf.predict(X_test) #预测测试集
    print(classification_report(y_test,y_predict))
    
    






