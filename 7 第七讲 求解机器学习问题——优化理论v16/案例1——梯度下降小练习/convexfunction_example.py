# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 图1：二次函数整体
plt.figure(1, figsize=(8, 4))
x = np.arange(-5, 5, 0.01)
y = x * x + 2 * x
plt.plot(x, y, color='blue')
plt.axvline(x=-1, color='red', linestyle='--', label='最小值点x=-1')  # 标注最小值点
plt.xlabel('x')
plt.ylabel('y')
plt.title('二次函数 y = x² + 2x')
plt.legend()

# 图2：二次函数局部
plt.figure(2, figsize=(8, 4))
x = np.arange(-2, 0, 0.01)
y = x * x + 2 * x
plt.plot(x, y, color='green')
plt.axvline(x=-1, color='red', linestyle='--', label='最小值点x=-1')
plt.xlabel('x')
plt.ylabel('y')
plt.title('二次函数在最小值附近的形态')
plt.legend()

# 图3：绝对值函数
plt.figure(3, figsize=(8, 4))
x = np.arange(-5, 5, 0.01)
y = abs(x)
plt.plot(x, y, color='orange')
plt.axvline(x=0, color='red', linestyle='--', label='最小值点x=0（不可导）')
plt.xlabel('x')
plt.ylabel('y')
plt.title('绝对值函数 y = |x|')
plt.legend()

# 图4：二维损失曲面（修改后）
fig4 = plt.figure(4, figsize=(10, 8))
# 替换原Axes3D初始化方式，用add_subplot创建3D子图
ax = fig4.add_subplot(111, projection='3d')

a = np.arange(-3, 5, 0.01)
b = np.arange(-3, 5, 0.01)
A, B = np.meshgrid(a, b)
x1, y1 = 0, 1
x2, y2 = 1, 2
los = (A + B * x1 - y1) ** 2 + (A + B * x2 - y2) ** 2

# 绘制3D曲面和等高线
ax.plot_surface(A, B, los, alpha=0.8, cmap='rainbow')
ax.contour(A, B, los, zdir='z', offset=0, cmap="rainbow")
ax.set_xlabel('截距a')
ax.set_ylabel('斜率b')
ax.set_zlabel('损失值')
ax.set_title('线性回归平方损失曲面（两个样本）')

# 确保图形渲染（必要时添加）
plt.tight_layout()
plt.show()
