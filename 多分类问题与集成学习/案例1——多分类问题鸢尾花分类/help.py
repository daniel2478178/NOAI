"""
numpy	np.meshgrid()	生成网格点坐标矩阵，用于绘制决策边界
numpy	np.arange()	生成指定范围的等差数组，作为网格点的坐标值
numpy	np.c_[]	按列拼接数组（核心：将 xx、yy 展平后拼接成特征矩阵）
numpy	np.ones()	生成全 1 数组，用于补充固定值的特征
pandas	pd.read_excel()	读取 Excel 格式的数据集，支持指定 header 和 index 列

matplotlib.pyplot	plt.pcolormesh()	绘制分类决策边界的彩色网格图
matplotlib.pyplot	plt.scatter()	绘制训练样本的散点图
matplotlib.colors	ListedColormap()	自定义颜色映射表，区分不同类别

sklearn.linear_model	LogisticRegression()	构建逻辑回归分类模型，参数penalty指定正则化方式，C指定正则化强度
sklearn.svm	SVC()	构建支持向量机分类模型，参数kernel指定核函数（linear/rbf）
sklearn.tree	DecisionTreeClassifier()	构建决策树分类模型
sklearn.neighbors	KNeighborsClassifier()	构建 K 近邻分类模型，参数n_neighbors指定邻居数
sklearn.neural_network	MLPClassifier()	构建多层感知机神经网络模型，参数hidden_layer_sizes指定隐含层结构
所有分类模型	model.fit(X, y)	训练模型，输入特征矩阵 X 和标签向量 y
所有分类模型	model.predict(X)	对输入特征矩阵 X 进行分类预测，返回标签数组
"""
