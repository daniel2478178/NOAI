# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 20:12:48 2020
@author: zzj
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


def load_datafromexcel(path):
    """
    从Excel读取数据集的函数
    :param path: excel文件路径
    :return: X(特征矩阵), y(标签数组)
    数据格式约定：excel第0行=表头，第0列=样本索引列，最后一列=分类标签(label)，其他列=特征
    """
    Sample_excel = pd.read_excel(path, header=0, index_col=0)  # 读取excel，指定表头和索引列
    Sample = Sample_excel.values  # pandas.DataFrame 转 numpy.ndarray 数组，方便后续计算
    y = Sample[:, Sample.shape[1] - 1]  # 最后一列作为标签值
    X = Sample[:, 0:Sample.shape[1] - 1]  # 除最后一列外，其他列都是特征
    return X, y


def draw_figure(X, y, clf):
    """
    绘制【分类决策边界+样本点】的核心可视化函数
    :param X: 特征数据（二维）
    :param y: 标签数据
    :param clf: 训练好的分类模型（KNN/逻辑回归/神经网络）
    """
    h = .02  # 网格步长，越小网格越密，决策边界越平滑，值越小绘图越慢
    # 定义绘图的颜色映射：浅色填充决策区域，深色绘制样本点
    cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue'])  # 决策区域颜色
    cmap_bold = ListedColormap(['darkorange', 'c', 'darkblue'])  # 样本点颜色

    # 1. 生成网格矩阵：确定画布的x/y轴范围，扩1个单位防止样本点贴边
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    # np.meshgrid：生成二维网格坐标，覆盖整个特征空间
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # 2. 对网格中每一个点做预测，得到该点的分类结果
    y_predict = clf.predict(np.c_[xx.ravel(), yy.ravel()])  # ravel展平成一维，np.c_按列拼接
    y_predict = y_predict.reshape(xx.shape)  # 把预测结果恢复成网格形状，用于填充颜色

    # 3. 绘制决策边界的颜色填充
    plt.figure(figsize=(6, 4))  # 新增：指定画布大小，防止图片过小
    plt.pcolormesh(xx, yy, y_predict, cmap=cmap_light)

    # 4. 绘制原始的样本点
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
    # edgecolor='k'：样本点加黑色边框，区分更明显；s=20：样本点大小

    # 5. 坐标轴范围+标题
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("2 Features, Iris Classification")
    plt.show()  # 新增：显示图片，你的原代码缺少这句，可能导致图片不弹出
    return


if __name__ == '__main__':
    # ========== 1. 读取数据 ==========
    X_train, y_train = load_datafromexcel(path='iris_2feature2label_train.xlsx')
    X_train = X_train[:, :2]  # 强制只取前2列特征，保证是二维特征，才能画平面决策边界
    X_test, y_test = load_datafromexcel(path='iris_2feature2label_test.xlsx')
    X_test = X_test[:, :2]  # 同上

    # ========== 2. 模型1：K近邻(KNN) 分类 ==========
    n_neighbors = 5  # 设置K值：选取最近的5个样本做投票
    clf = neighbors.KNeighborsClassifier(n_neighbors)  # 初始化KNN分类器
    clf.fit(X_train, y_train)  # 训练模型：KNN是惰性学习，fit只是加载样本，无训练过程
    draw_figure(X_train, y_train, clf)  # 绘制KNN的决策边界
    # 新增：模型评估，打印测试集准确率
    knn_pred = clf.predict(X_test)
    print(f"KNN模型 测试集准确率：{accuracy_score(y_test, knn_pred):.4f}")

    # ========== 3. 模型2：逻辑斯蒂回归 分类 ==========
    clf = linear_model.LogisticRegression(C=1e10)  # 初始化逻辑回归分类器
    # C=1e10：正则化系数的倒数，C越大，正则化惩罚越弱，拟合效果越强
    clf.fit(X_train, y_train)  # 训练模型：求解最优参数θ
    draw_figure(X_train, y_train, clf)  # 绘制逻辑回归的决策边界
    # 新增：模型评估
    lr_pred = clf.predict(X_test)
    print(f"逻辑回归模型 测试集准确率：{accuracy_score(y_test, lr_pred):.4f}")

    # ========== 4. 模型3：神经网络(多层感知机MLP) 分类 ==========
    # 修正语法：hidden_layer_sizes=(5) → hidden_layer_sizes=(5,) 标准元组格式
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,), random_state=1)
    clf.fit(X_train, y_train)  # 训练模型：前向传播+反向传播+参数更新
    draw_figure(X_train, y_train, clf)  # 绘制神经网络的决策边界
    # 新增：模型评估
    mlp_pred = clf.predict(X_test)
    print(f"神经网络模型 测试集准确率：{accuracy_score(y_test, mlp_pred):.4f}")