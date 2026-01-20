#!/usr/bin/env python
# coding: utf-8

# logistic_regression_experiment
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import classification_report  # 预测回归结果分析工具
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.inspection import DecisionBoundaryDisplay

def load_datafromexcel(path): 
    #统一格式为：第0行为header行，第0列为index列，最后一列是label
    Sample_excel = pd.read_csv(path, header = 0, index_col = 0)
    Sample = Sample_excel.values  #转成numpy数组
    y = Sample[:,Sample.shape[1]-1]
    #print(y)
    X = Sample[:,0:Sample.shape[1]-1]
    #print(X)
    return X,y


if __name__ == '__main__':
    data_X, data_y = load_datafromexcel(path = 'iris.csv')  #读取数据（不区分训练集和测试集）
    print(data_X)
    # Parameters
    n_classes = 3 
    plot_colors = "ryb"
    plot_step = 0.02
    
    
    for pairidx, pair in enumerate([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]):#选择不同列，对比不同决策树效果
        # We only take the two corresponding features
        X = data_X[:, pair]
        y = data_y
    
        # Train
        clf = DecisionTreeClassifier().fit(X, y)  #选择不同列，对比不同决策树效果
    
        # 绘制决策树
        ax = plt.subplot(2, 3, pairidx + 1)
        plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
        DecisionBoundaryDisplay.from_estimator(
            clf,
            X,
            cmap=plt.cm.RdYlBu,
            response_method="predict",
            ax=ax,
            #xlabel=X[pair[0]],
            #ylabel=X[pair[1]],
        )
    
        # Plot the training points
        for i, color in zip(range(n_classes), plot_colors):
            idx = np.where(y == i)
            plt.scatter(
                X[idx, 0],
                X[idx, 1],
                c=color,
                #label=iris.target_names[i],
                edgecolor="black",
                s=15,
            )
    
    plt.suptitle("6 Decision Trees Each with 2 Features")
    plt.legend(loc="lower right", borderpad=0, handletextpad=0)
    _ = plt.axis("tight")
        





