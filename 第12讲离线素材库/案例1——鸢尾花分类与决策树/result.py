#!/usr/bin/env python
# coding: utf-8

# logistic_regression_experiment + 决策树阈值计算
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import classification_report  # 预测回归结果分析工具
from sklearn.tree import DecisionTreeClassifier, plot_tree


# ===================== 新增：计算特征最优阈值的核心函数 =====================
def cal_entropy(y):
    """计算标签的信息熵"""
    unique_labels, counts = np.unique(y, return_counts=True)
    entropy = 0.0
    total = len(y)
    for cnt in counts:
        p = cnt / total
        entropy -= p * np.log2(p) if p > 0 else 0
    return entropy


def cal_best_threshold(X_feature, y):
    """计算单个连续特征的最优分界阈值（基于信息熵最小化）"""
    # 去重并排序
    unique_vals = np.unique(X_feature)
    if len(unique_vals) == 1:
        return None, 0.0  # 特征值唯一，无法划分

    base_entropy = cal_entropy(y)
    best_threshold = None
    min_entropy = float('inf')

    # 遍历所有候选阈值（相邻值中点）
    for i in range(1, len(unique_vals)):
        threshold = (unique_vals[i - 1] + unique_vals[i]) / 2
        # 按阈值划分样本
        left_mask = X_feature <= threshold
        right_mask = X_feature > threshold
        if len(y[left_mask]) == 0 or len(y[right_mask]) == 0:
            continue
        # 计算加权熵
        left_ent = cal_entropy(y[left_mask])
        right_ent = cal_entropy(y[right_mask])
        weighted_ent = (len(y[left_mask]) / len(y)) * left_ent + (len(y[right_mask]) / len(y)) * right_ent
        # 更新最优阈值
        if weighted_ent < min_entropy:
            min_entropy = weighted_ent
            best_threshold = threshold

    info_gain = base_entropy - min_entropy  # 信息增益
    return best_threshold, min_entropy, info_gain


# ===================== 复用你的原有函数 =====================
def load_datafromexcel(path):
    # 统一格式为：第0行为header行，第0列为index列，最后一列是label
    Sample_excel = pd.read_csv(path, header=0, index_col=0)
    Sample = Sample_excel.values  # 转成numpy数组
    y = Sample[:, Sample.shape[1] - 1]
    X = Sample[:, 0:Sample.shape[1] - 1]
    return X, y


def draw_figure(X, y, clf):  # 画图函数
    h = .02  # step size in the mesh
    # Create color maps
    cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue'])
    cmap_bold = ListedColormap(['darkorange', 'c', 'darkblue'])
    # Draw mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    y_predict = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # Put the result into a color plot
    y_predict = y_predict.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, y_predict, cmap=cmap_light)
    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("Iris 2 feature 2 label")
    return


# ===================== 主函数（保留你的逻辑+新增阈值计算） =====================
if __name__ == '__main__':
    # 1. 加载数据（完全复用你的路径和逻辑）
    X_train, y_train = load_datafromexcel(path='iris_2feature2label_train.csv')
    X_train = X_train[:, :2]  # 只取前2个特征
    X_test, y_test = load_datafromexcel(path='iris_2feature2label_test.csv')
    X_test = X_test[:, :2]

    # 2. 新增：计算训练集两个特征的最优阈值
    print("===== 鸢尾花数据集-特征最优阈值计算 =====")
    feature_names = ['特征1（花萼长度/花瓣长度）', '特征2（花萼宽度/花瓣宽度）']
    for idx in range(X_train.shape[1]):
        feature_vals = X_train[:, idx]
        best_thresh, min_ent, info_gain = cal_best_threshold(feature_vals, y_train)
        print(f"\n{feature_names[idx]}:")
        print(f"  最优分界阈值：{best_thresh:.4f}")
        print(f"  划分后最小加权熵：{min_ent:.4f}")
        print(f"  信息增益：{info_gain:.4f}")

    # 3. 训练模型（完全复用你的逻辑）
    clf = DecisionTreeClassifier(max_depth=5)
    clf.fit(X_train, y_train)

    # 4. 训练集评估（复用）
    y_predict_train = clf.predict(X_train)
    print("\n===== 训练集评估报告 =====")
    print(classification_report(y_train, y_predict_train))

    # 5. 测试集评估（复用）
    y_predict_test = clf.predict(X_test)
    print("\n===== 测试集评估报告 =====")
    print(classification_report(y_test, y_predict_test))

    # 6. 绘图（复用）
    draw_figure(X_train, y_train, clf)  # train画图
    plt.figure(2)
    plot_tree(clf, filled=True)  # 决策树结构可视化
    plt.figure(3)
    draw_figure(X_test, y_test, clf)  # test画图

    plt.show()