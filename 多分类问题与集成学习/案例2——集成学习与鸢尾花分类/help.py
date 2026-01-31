"""
numpy	np.meshgrid(x, y)	生成网格点坐标矩阵，输入两个一维数组（x/y 轴坐标），输出二维网格矩阵，用于绘制连续的决策边界
numpy	np.arange(start, stop, step)	生成指定范围、指定步长的等差数组，作为网格点的坐标值（控制决策边界的精细度）
numpy	np.c_[arr1, arr2, ...]	按列拼接多个数组，核心用于将展平后的网格点（xx.ravel ()、yy.ravel ()）拼接成模型可识别的特征矩阵
numpy	ndarray.ravel()	将多维数组展平为一维数组，用于网格点的特征矩阵拼接
pandas	pd.read_excel(path, header, index_col)	读取 Excel 格式数据集，header=0指定第 0 行为表头，index_col=0指定第 0 列为索引列

matplotlib.pyplot	plt.pcolormesh(xx, yy, z, cmap)	根据网格点和对应的分类结果，绘制彩色的分类决策区域，cmap指定颜色映射表
matplotlib.pyplot	plt.scatter(x, y, c, cmap, edgecolor, s)	绘制散点图，c指定样本颜色（对应标签），edgecolor指定点的边框颜色，s指定点的大小
matplotlib.colors	ListedColormap(colors_list)	自定义颜色映射表，用于区分不同类别的分类区域和样本点

sklearn.ensemble	AdaBoostClassifier(n_estimators)	初始化 AdaBoost 分类器，n_estimators指定弱分类器数量（默认 50），属于 Boosting 集成策略
sklearn.ensemble	RandomForestClassifier()	初始化随机森林分类器，默认使用 100 棵决策树，属于 Bagging 集成策略
所有分类模型	model.fit(X_train, y_train)	训练模型，输入训练集特征矩阵X_train和标签向量y_train，完成模型参数学习
所有分类模型	model.predict(X)	对输入特征矩阵X进行分类预测，返回每个样本的预测标签数组
"""