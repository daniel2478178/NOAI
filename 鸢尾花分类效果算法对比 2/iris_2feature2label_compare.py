# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 20:12:48 2020
@author: Gengchen Dong 
功能：对比KNN、逻辑回归、神经网络 三种算法在鸢尾花数据集的分类效果 + 决策边界可视化
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn import neighbors, linear_model
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score  # 新增：模型评估，计算准确率
import glob
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
n_neighbors = 15  # KNN的邻居数
def load_datafromexcel(path):
    """
    从Excel读取数据集的函数
    :param path: excel文件路径
    :return: X(特征矩阵), y(标签数组)
    数据格式约定：excel第0行=表头，第0列=样本索引列，最后一列=分类标签(label)，其他列=特征
    """
    df = pd.read_csv(path)
    x = df[df.columns[1:-1]].to_numpy()
    y = df['Types'].to_numpy()
    return x,y


def draw_figure(X, y, clf):
    """
    绘制【分类决策边界+样本点】的核心可视化函数
    :param X: 特征数据（二维）
    :param y: 标签数据
    :param clf: 训练好的分类模型（KNN/逻辑回归/神经网络）
    """
    type1place = (y==0)
    type2place = (y==1)
    for data in X[type1place]:
   
        plt.scatter(data[0],data[1],color = 'r')
    for data in X[type2place]:
        plt.scatter(data[0],data[1],color = 'b')
    xaxis = np.arange(np.min(X[:,0],axis = 0),np.max(X[:,0],axis = 0),0.08)
    yaxis = np.arange(np.min(X[:,1],axis = 0),np.max(X[:,1],axis = 0),0.08)
    #print(xaxis,yaxis)
    xxdots = []
    yydots = []
    for i,xx in enumerate(xaxis):
        for j,yy in enumerate(yaxis):
            if i == 0 or j == 0:
                continue
            prediction = lambda xx,yy:int(clf(np.array([[xx,yy]])))
            if  (prediction(xx,yy) != prediction(xaxis[i-1],yy)) or ((prediction(xx,yaxis[j-1]) != prediction(xx,yy))):   
                xxdots.append(xx)
                yydots.append(yy)
    #print(xxdots,yydots)
          
    plt.plot(xxdots,yydots)
    
    plt.savefig('./figure.png')
        
        



    # ========== 1. 读取数据 ==========
X_test, y_test = load_datafromexcel("iris_2feature2label_train.csv")

    # ========== 2. 模型1：K近邻(KNN) 分类 ==========
clf = neighbors.KNeighborsClassifier(n_neighbors)  # 初始化KNN分类器
clf.fit(X_test,y_test)
    # 模型评估，测试集准确率
knn_pred = clf.predict(X_test)
print(accuracy_score(y_test, knn_pred))
   # print(X_test.shape)

draw_figure(X_test,y_test,clf.predict)
    
    # ========== 3. 模型2：逻辑斯蒂回归 分类 ==========
clf = linear_model.LogisticRegression(C=1e10)  # 初始化逻辑回归分类器
    # C=1e10：正则化系数的倒数，C越大，正则化惩罚越弱，拟合效果越强
clf.fit(X_test,y_test)
    # 模型评估，测试集准确率
knn_pred = clf.predict(X_test)
print(accuracy_score(y_test, knn_pred))
   # print(X_test.shape)

draw_figure(X_test,y_test,clf.predict)

    # ========== 4. 模型3：神经网络(多层感知机MLP) 分类 ==========
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,), random_state=1)
clf.fit(X_test,y_test)
    # 模型评估，测试集准确率
    
knn_pred = clf.predict(X_test)
print(accuracy_score(y_test, knn_pred))
   # print(X_test.shape)

draw_figure(X_test,y_test,clf.predict)