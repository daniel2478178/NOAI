# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn import linear_model
from sklearn import neighbors 
from sklearn.neural_network import MLPClassifier 

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
def load_datafromexcel(path): 
    # 统一格式为：第0行为header行，第0列为index列，最后一列是label
    Sample_excel = pd.read_excel(path, header = 0, index_col = 0)
    Sample = Sample_excel.values  # 转成numpy数组
    y = Sample[:,Sample.shape[1]-1]
    X = Sample[:,0:Sample.shape[1]-1]
    return X,y

def draw_figure(X,y,clf): # 画图函数
    h = .02  # step size in the mesh
    # Create color maps
    cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue'])
    cmap_bold = ListedColormap(['darkorange', 'c', 'darkblue'])
    # Draw mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1   
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # 补全代码1：拼接2个网格特征 + 固定值的后2个特征（2和3），完成4特征输入
    
    
                     
    
    y_predict=clf.predict(np.c_[xx.ravel(), yy.ravel(), 
                               np.full_like(xx.ravel(), X[:,2][49]),   
                               np.full_like(xx.ravel(), X[:,3][49])])    
    # Put the result into a color plot
    y_predict = y_predict.reshape(xx.shape)
    print(np.where(y_predict == 0))
    plt.figure()
    plt.pcolormesh(xx.ravel(), yy.ravel(), y_predict, cmap=cmap_light)
    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("Iris")   
    return

if __name__ == '__main__':
    # 补全代码2：读取4特征鸢尾花训练集数据
    X_train, y_train = load_datafromexcel(path = 'iris_4feature3label_train.xlsx')  
    # 测试集（无需补全，仅作参考）
    X_test, y_test = load_datafromexcel(path = 'iris_4feature3label_test.xlsx')    
    # 训练逻辑回归模型
    #clf =  linear_model.LogisticRegression(penalty = 'l2', C=1e10)
    clf = DecisionTreeClassifier(criterion='gini', max_depth=1)
    # 补全代码3：训练模型
    clf.fit(X_train, y_train)
    # 调用画图函数
    draw_figure(X_train,y_train,clf)
    plt.show()
    predict = clf.predict(X_test)
    print("逻辑回归模型预测结果分析：")
    print(classification_report(y_test, predict))