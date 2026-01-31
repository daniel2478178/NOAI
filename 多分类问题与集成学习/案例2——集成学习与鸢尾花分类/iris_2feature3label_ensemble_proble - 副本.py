# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
# 补全代码1：导入集成学习所需的AdaBoost和随机森林分类器
_______________
_______________

def load_datafromexcel(path): 
    # 统一格式为：第0行为header行，第0列为index列，最后一列是label
    Sample_excel = pd.read_excel(path, header = 0, index_col = 0)
    Sample = Sample_excel.values  # 转成numpy数组
    # 补全代码2：提取数据集最后一列作为标签y
    y = _______________
    # 补全代码3：提取数据集除最后一列外的所有列作为特征X
    X = _______________
    return X,y

def draw_figure(X,y,clf): # 画图函数：绘制分类决策边界
    h = .02  # step size in the mesh
    # Create color maps：定义分类区域和样本点的颜色映射
    cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue'])
    cmap_bold = ListedColormap(['darkorange', 'c', 'darkblue'])
    # Draw mesh：确定画图范围
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    # 补全代码4：生成网格点坐标矩阵（覆盖整个画图范围）
    xx, yy = _______________
    # 补全代码5：拼接网格点并使用模型预测分类结果
    y_predict=clf.predict(_______________)    
    # Put the result into a color plot：还原预测结果形状并绘制彩色决策区域
    y_predict = y_predict.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, y_predict, cmap=cmap_light)
    # Plot also the training points：绘制训练样本点
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("Iris Classification by Ensemble Learning")   
    return

if __name__ == '__main__':
    # 补全代码6：读取2特征鸢尾花训练集数据
    X_train, y_train = load_datafromexcel(path = _______________)  
    # 截取前2个特征（确保二维特征用于可视化）
    X_train = X_train[:,:2]
    # 读取测试集数据（无需补全，仅作参考）
    X_test, y_test = load_datafromexcel(path = 'iris_2feature3label_test.xlsx')  
    X_test = X_test[:,:2]    
    # 训练集成学习模型
    # 可选1：AdaBoost模型（n_estimators=100表示100个弱分类器）
    # clf = AdaBoostClassifier(n_estimators=100)
    # 补全代码7：初始化随机森林分类器
    clf = _______________
    # 补全代码8：使用训练集训练模型
    _______________
    # 调用画图函数展示决策边界
    draw_figure(X_train,y_train,clf)