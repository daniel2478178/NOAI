# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn import linear_model
from sklearn import neighbors 
from sklearn.neural_network import MLPClassifier 
from sklearn.metrics import classification_report  # 预测回归结果分析工具
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
def load_datafromexcel(path): 
    # 统一格式为：第0行为header行，第0列为index列，最后一列是label
    Sample_excel = pd.read_excel(path, header = 0, index_col = 0)
    Sample = Sample_excel.values  # 转成numpy数组
    # 补全代码1：提取最后一列作为标签y
    y = Sample[:, Sample.shape[1]-1]
    # 补全代码2：提取除最后一列外的所有列作为特征X
    X = Sample[:, :-1]
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
    # 补全代码3：拼接网格点并预测分类结果
    y_predict=clf.predict(np.c_[xx.ravel(), yy.ravel()])    
    # Put the result into a color plot
    y_predict = y_predict.reshape(xx.shape)
    #plt.figure()
    plt.pcolormesh(xx, yy, y_predict, cmap=cmap_light)
    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("Iris") 
    #plt.savefig('iris_2feature3label_logistic_regression.png')  
    plt.show()

if __name__ == '__main__':
    # 补全代码4：读取2特征鸢尾花训练集数据
    X_train, y_train = load_datafromexcel(path = 'iris_2feature3label_train.xlsx')  
    # 截取前2个特征
    X_train = X_train[:,:2]
    # 测试集（无需补全，仅作参考）
    X_test, y_test = load_datafromexcel(path = 'iris_2feature3label_test.xlsx')  
    X_test = X_test[:,:2]    
    # 训练逻辑回归模型
    clf =  linear_model.LogisticRegression(penalty = 'l2', C=1e10)
    # 补全代码5：训练模型
    clf.fit(X_train, y_train)
    predict = clf.predict(X_test)
    print(classification_report(y_test, predict))
    # 调用画图函数
    draw_figure(X_train,y_train,clf)
    