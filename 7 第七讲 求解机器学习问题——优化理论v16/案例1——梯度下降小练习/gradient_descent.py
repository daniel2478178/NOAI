# -*- coding: utf-8 -*-
"""
Gradient Descent Method for y= x^4+2x
author: Zheng Zijie
优化点：修复停止条件、调整初始值、增加可视化、支持中文显示
"""
import numpy as np
import matplotlib.pyplot as plt

# 解决Matplotlib中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# --------------- 梯度下降参数设置 ---------------
x_init = 2  # 调整初始值（离最小值更近，减少迭代次数）
x_j = x_init  # 当前迭代的x值
alpha = 0.005  # 调整学习率（0.005比0.0005收敛更快，且不震荡）
tol = 0.001  # 收敛阈值
max_iter = 10000  # 最大迭代次数（防止死循环）

# 记录迭代过程
x_mem = [x_j]
y_mem = [x_j ** 4 + 2 * x_j]

# --------------- 梯度下降迭代 ---------------
iter_num = 0
while iter_num < max_iter:
    # 计算梯度（导数）和负梯度方向
    grad = 4 * (x_j ** 3) + 2  # 函数y=x^4+2x的导数
    delta = -grad  # 负梯度：下降方向

    # 迭代更新x
    x_temp = x_j + alpha * delta
    y_temp = x_temp ** 4 + 2 * x_temp

    # 记录迭代结果
    x_mem.append(x_temp)
    y_mem.append(y_temp)

    # 停止条件：相邻x的差值 < 阈值（比函数值更稳定）
    if abs(x_temp - x_j) < tol:
        break

    # 更新迭代变量
    x_j = x_temp
    iter_num += 1

# --------------- 结果输出 ---------------
print(f"迭代次数：{iter_num}")
print(f"最小值点x≈{x_j:.4f}，最小值y≈{y_mem[-1]:.4f}")
print(f"理论最小值点x=-∛0.5≈{-np.power(0.5, 1 / 3):.4f}")

# --------------- 可视化 ---------------
# 绘制函数曲线
x_curve = np.arange(-2, 3, 0.01)  # 聚焦最小值附近区间，图形更清晰
y_curve = x_curve ** 4 + 2 * x_curve

plt.figure(1, figsize=(10, 6))
plt.plot(x_curve, y_curve, 'k-', label='函数y=x⁴+2x')
# 绘制迭代路径（红色星号，突出起点和终点）
plt.plot(x_mem, y_mem, 'r*-', linewidth=1, markersize=6, label='梯度下降迭代路径')
# 标注最小值点
plt.scatter(x_j, y_mem[-1], color='blue', s=100, label=f'迭代最小值点({x_j:.4f}, {y_mem[-1]:.4f})')

plt.title('梯度下降求解y=x⁴+2x的最小值', fontsize=12)
plt.xlabel('x', fontsize=10)
plt.ylabel('y', fontsize=10)
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()