# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 21:00:43 2020

@author: zzj
"""


# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 20:12:48 2020

@author: zzj
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn import svm #调用sklearn里面的svm函数
from sklearn.tree import DecisionTreeClassifier
from sklearn import linear_model #调用sklearn里面的linear_regression函数
from sklearn import neighbors 
from sklearn.neural_network import MLPClassifier 

import os
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
    y_predict=clf.predict(np.c_[xx.ravel(), yy.ravel(),2*np.ones(len(xx.ravel())),3*np.ones(len(yy.ravel()))])    
    #Put the result into a color plot
    y_predict = y_predict.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, y_predict, cmap=cmap_light)
    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("Iris")   
    return

if __name__ == '__main__':
    X_train, y_train = load_datafromexcel(path = 'iris_4feature3label_train.xlsx')  #读取训练集数据
    #X_train = X_train[:,:2]
    #测试集
    X_test, y_test = load_datafromexcel(path = 'iris_4feature3label_test.xlsx')  #读取测试集数据
    #X_test = X_test[:,:2]    
    #训练模型 可以换成不同样子的
    clf = svm.SVC(C=10000000000, kernel = 'rbf',degree = 1, tol = 1e-3) #SVM rbf
    #clf = svm.SVC(C=10000000000, kernel = 'linear',degree = 1, tol = 1e-3) #SVM linear
    #clf = DecisionTreeClassifier() # decision tree
    #clf =  linear_model.LogisticRegression(penalty = 'l2', C=1e10) #logistic regression
    #n_neighbors=15
    #clf =  neighbors.KNeighborsClassifier(n_neighbors) #KNN
    #clf =  MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 5), random_state=1) #含有两个隐含层的神经网络模型
    clf.fit(X_train,y_train) #不管算法怎么执行的，直接进行模型建立
    draw_figure(X_train,y_train,clf) #基于训练集画图
    draw_figure(X_test,y_test,clf) #基于测试集画图
    
