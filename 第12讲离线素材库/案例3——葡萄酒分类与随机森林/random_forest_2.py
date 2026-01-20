# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 21:06:40 2024

@author: NING MEI
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine  #直接从sklearn里面读取红酒数据集
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree
# 加载wine数据集

def load_datafromexcel(path): 
    #统一格式为：第0行为header行，第0列为index列，最后一列是label
    Sample_excel = pd.read_csv(path, header = 0, index_col = 0)
    Sample = Sample_excel.values  #转成numpy数组
    y = Sample[:,Sample.shape[1]-1]
    #print(y)
    X = Sample[:,0:Sample.shape[1]-1]
    #print(X)
    return X,y

#wine = load_wine()
X, y = load_datafromexcel("wine.csv")  #读取红酒数据库
df = pd.read_csv("wine.csv")
headers = df.columns.tolist()
headers = headers[0:len(headers)-1]
print(headers)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 实例化随机森林分类器
rf = RandomForestClassifier(n_estimators=100, random_state=42, max_features = 'log2')

# 训练模型
rf.fit(X_train, y_train)

# 预测测试集
y_pred = rf.predict(X_test)

# 计算准确率
#accuracy = accuracy_score(y_test, y_pred)
#print(accuracy)

# 重新定义一个函数来评估不同n_estimators下的准确率
def evaluate_n_estimators_wine(n_estimators_range):
    accuracies = []
    for n_estimators in n_estimators_range:
        rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42, max_features = 'log2')
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))
    return accuracies

# 定义n_estimators的范围
n_estimators_range = np.arange(1, 101, 10)

# 计算准确率
accuracies_wine = evaluate_n_estimators_wine(n_estimators_range)

# 绘制图形
plt.figure(figsize=(10, 6))
plt.plot(n_estimators_range, accuracies_wine, marker='o', label='Wine Dataset')
plt.title('Accrucy changes with n_estimators change')
plt.xlabel('n_estimators')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()  


# 设置n_estimators为5
rf_small = RandomForestClassifier(n_estimators=5, random_state=42)
rf_small.fit(X_train, y_train)

# 为每棵树绘制结构
plt.figure(figsize=(15, 10))
for i, estimator in enumerate(rf_small.estimators_):
    plt.subplot(1, 5, i + 1)
    plot_tree(estimator, filled=True, max_depth=3, feature_names = headers)
    plt.title(f"Decision {i + 1}")

plt.tight_layout()
plt.show()