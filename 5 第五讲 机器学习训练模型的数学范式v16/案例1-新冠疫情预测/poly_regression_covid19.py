# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import os

# --- 1. 环境设置 ---
# 解决Matplotlib中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False   # 解决保存图像是负号'-'显示为方块的问题

# --- 2. 拉格朗日插值函数 (保持不变) ---
def lagrangian_interplot(x_sample, y_sample, x):
    y = np.zeros(len(x))
    for n in range(len(y)):
        for i in range(len(x_sample)):
            y_addtemp = y_sample.iloc[i] # 使用.iloc确保正确索引
            for j in range(len(x_sample)):
                if j != i:
                    y_addtemp *= (x[n] - x_sample.iloc[j]) / (x_sample.iloc[i] - x_sample.iloc[j])
            y[n] += y_addtemp
    return y
print(os.getcwd())
# --- 3. 主程序 ---
if __name__ == "__main__":
    # 检查数据文件是否存在
    filename = "/Users/daniel/Documents/5 第五讲 机器学习训练模型的数学范式v16/案例1-新冠疫情预测/data_covid19mar.csv"
    if not os.path.exists(filename):
        print(f"错误：数据文件 '{filename}' 未找到。请确保文件与脚本在同一目录下。")
    else:
        data = pd.read_csv(filename)
        x_sample = data['Mar']
        y_sample = data['Num_of_cases']

        # 为了绘图，生成平滑的x轴
        x_smooth = np.arange(x_sample.min(), x_sample.max() + 0.1, 0.1).reshape(-1, 1)

        # --- 4. 创建图形并绘制所有模型 ---
        plt.figure(figsize=(12, 8))
        plt.title('新冠疫情数据拟合对比 (3月)')
        plt.scatter(x_sample, y_sample, color='red', label='原始数据点')
        plt.xlabel("3月日期")
        plt.ylabel("病例数")
        plt.grid(True)

        # 绘制拉格朗日插值
        y_lagrangian = lagrangian_interplot(x_sample, y_sample, x_smooth.flatten())
        plt.plot(x_smooth, y_lagrangian, 'g--', label='拉格朗日插值 (过拟合示例)')

        # 绘制线性回归 (使用make_pipeline更简洁)
        model_linear = make_pipeline(Ridge(alpha=1e-6))
        model_linear.fit(x_sample.values.reshape(-1, 1), y_sample)
        y_linear = model_linear.predict(x_smooth)
        plt.plot(x_smooth, y_linear, 'b-', label='线性回归 (欠拟合示例)')

        # 绘制二次多项式回归
        model_poly2 = make_pipeline(PolynomialFeatures(degree=2), Ridge(alpha=0.01))
        model_poly2.fit(x_sample.values.reshape(-1, 1), y_sample)
        y_poly2 = model_poly2.predict(x_smooth)
        plt.plot(x_smooth, y_poly2, 'm-', label='二次多项式回归')

        # 绘制10次多项式回归 (带正则化)
        model_poly10 = make_pipeline(PolynomialFeatures(degree=10), Ridge(alpha=5))
        model_poly10.fit(x_sample.values.reshape(-1, 1), y_sample)
        y_poly10 = model_poly10.predict(x_smooth)
        plt.plot(x_smooth, y_poly10, 'c-', label='10次多项式回归 (带正则化)')

        # 添加图例并显示
        plt.legend()
        plt.show()