#!/usr/bin/env python
# coding: utf-8


# logistic_regression_experiment
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib.colors import ListedColormap
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report  # 预测回归结果分析工具
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def load_datafromexcel(path): 
    #统一格式为：第0行为header行，第0列为index列，最后一列是label
    Sample_excel = pd.read_excel(path, header = 0, index_col = 0)
    Sample = Sample_excel.values  #转成numpy数组
    y = Sample[:,Sample.shape[1]-1]
    #print(y)
    X = Sample[:,0:Sample.shape[1]-1]
    #print(X)
    return X,y

def draw_figure(X,y,clf): #画图函数
    h = .02  # step size in the mesh
    # Create color maps
    cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue'])
    cmap_bold = ListedColormap(['darkorange', 'c', 'darkblue'])
    # Draw mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    y_predict=clf.predict(np.c_[xx.ravel(), yy.ravel()])    
    #Put the result into a color plot
    y_predict = y_predict.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, y_predict, cmap=cmap_light)
    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("Iris 2 feature 2 label")   
    return

if __name__ == '__main__':
    X_train, y_train = load_datafromexcel(path = 'iris_2feature2label_train.xlsx')  #读取训练集数据
    X_train = X_train[:,:2]
    #测试集=训练集
    X_test, y_test = load_datafromexcel(path = 'iris_2feature2label_test.xlsx')  #读取训练集数据
    X_test = X_test[:,:2]
    #训练模型
    clf = GaussianNB() #gaussian密度函数的naive bayes
    clf.fit(X_train,y_train) #不管算法怎么执行的，直接进行模型建立
    
    # training set的拟合准确率
    y_predict_train = clf.predict(X_train)
    print(classification_report(y_train,y_predict_train))
    
    # testing set的拟合准确率
    y_predict_test = clf.predict(X_test)
    print(classification_report(y_test,y_predict_test))

    
    draw_figure(X_train,y_train,clf) #train画图
    draw_figure(X_test,y_test,clf) #test画图
    





