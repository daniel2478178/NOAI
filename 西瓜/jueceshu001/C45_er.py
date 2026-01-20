import pandas as pd
import numpy as np

#计算信息熵
def cal_information_entropy(data):
    data_label = data.iloc[:,-1]
    label_class =data_label.value_counts() #总共有多少类
    Ent = 0
    for k in label_class.keys():
        p_k = label_class[k]/len(data_label)
        Ent += -p_k*np.log2(p_k)
    return Ent

#计算给定数据属性a的信息增益
def cal_information_gain(data, a):
    Ent = cal_information_entropy(data)
    feature_class = data[a].value_counts() #特征有多少种可能
    gain = 0
    for v in feature_class.keys():
        weight = feature_class[v]/data.shape[0]
        Ent_v = cal_information_entropy(data.loc[data[a] == v])
        gain += weight*Ent_v
    return Ent - gain

def cal_gain_ratio(data , a):
    #先计算固有值intrinsic_value
    IV_a = 0
    feature_class = data[a].value_counts()  # 特征有多少种可能
    for v in feature_class.keys():
        weight = feature_class[v]/data.shape[0]
        IV_a += -weight*np.log2(weight)
    gain_ration = cal_information_gain(data,a)/IV_a
    return gain_ration

#获取标签最多的那一类
def get_most_label(data):
    data_label = data.iloc[:,-1]
    label_sort = data_label.value_counts(sort=True)
    return label_sort.keys()[0]

#挑选最优特征，即在信息增益大于平均水平的特征中选取增益率最高的特征
def get_best_feature(data):
    features = data.columns[:-1]
    res = {}
    for a in features:
        temp = cal_information_gain(data, a)
        gain_ration = cal_gain_ratio(data,a)
        res[a] = (temp,gain_ration)
    res = sorted(res.items(),key=lambda x:x[1][0],reverse=True) #按信息增益排名
    res_avg = sum([x[1][0] for x in res])/len(res) #信息增益平均水平
    good_res = [x for x in res if x[1][0] >= res_avg] #选取信息增益高于平均水平的特征
    result =sorted(good_res,key=lambda x:x[1][1],reverse=True) #将信息增益高的特征按照增益率进行排名
    return result[0][0] #返回高信息增益中增益率最大的特征

##将数据转化为（属性值：数据）的元组形式返回，并删除之前的特征列
def drop_exist_feature(data, best_feature):
    attr = pd.unique(data[best_feature])
    new_data = [(nd, data[data[best_feature] == nd]) for nd in attr]
    new_data = [(n[0], n[1].drop([best_feature], axis=1)) for n in new_data]
    return new_data

#创建决策树
def create_tree(data):
    data_label = data.iloc[:,-1]
    if len(data_label.value_counts()) == 1: #只有一类
        return data_label.values[0]
    if all(len(data[i].value_counts()) == 1 for i in data.iloc[:,:-1].columns): #所有数据的特征值一样，选样本最多的类作为分类结果
        return get_most_label(data)
    best_feature = get_best_feature(data) #根据信息增益得到的最优划分特征
    Tree = {best_feature:{}} #用字典形式存储决策树
    exist_vals = pd.unique(data[best_feature])  # 当前数据下最佳特征的取值
    if len(exist_vals) != len(column_count[best_feature]):  # 如果特征的取值相比于原来的少了
        no_exist_attr = set(column_count[best_feature]) - set(exist_vals)  # 少的那些特征
        for no_feat in no_exist_attr:
            Tree[best_feature][no_feat] = get_most_label(data)  # 缺失的特征分类为当前类别最多的
    for item in drop_exist_feature(data,best_feature): #根据特征值的不同递归创建决策树
        Tree[best_feature][item[0]] = create_tree(item[1])
    return Tree

def predict(Tree , test_data):
    first_feature = list(Tree.keys())[0]
    second_dict = Tree[first_feature]
    input_first = test_data.get(first_feature)
    input_value = second_dict[input_first]
    if isinstance(input_value , dict): #判断分支还是不是字典
        class_label = predict(input_value, test_data)
    else:
        class_label = input_value
    return class_label

if __name__ == '__main__':
    #读取数据
    data = pd.read_csv('data_word.csv')
    # 统计每个特征的取值情况作为全局变量
    column_count = dict([(ds, list(pd.unique(data[ds]))) for ds in data.iloc[:, :-1].columns])

    #创建决策树
    dicision_Tree = create_tree(data)
    print(dicision_Tree)
    #测试数据
    test_data_1 = {'色泽':'青绿','根蒂':'蜷缩','敲声':'浊响','纹理':'稍糊','脐部':'凹陷','触感':'硬滑'}
    test_data_2 = {'色泽': '乌黑', '根蒂': '稍蜷', '敲声': '浊响', '纹理': '清晰', '脐部': '凹陷', '触感': '硬滑'}
    result = predict(dicision_Tree,test_data_1)
    print('test_data_1的分类结果为：'+'好瓜'if result == 1 else 'test_data_1的分类结果为：坏瓜')

    result = predict(dicision_Tree,test_data_2)
    print('test_data_2的分类结果为：'+'好瓜'if result == 1 else 'test_data_2的分类结果为：坏瓜')



    # 绘制可视化树

    import matplotlib.pylab as plt
    import matplotlib

    # 能够显示中文
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['font.serif'] = ['SimHei']

    # 分叉节点，也就是决策节点
    decisionNode = dict(boxstyle="sawtooth", fc="0.8")

    # 叶子节点
    leafNode = dict(boxstyle="round4", fc="0.8")

    # 箭头样式
    arrow_args = dict(arrowstyle="<-")


    def plotNode(nodeTxt, centerPt, parentPt, nodeType):
        """
        绘制一个节点
        :param nodeTxt: 描述该节点的文本信息
        :param centerPt: 文本的坐标
        :param parentPt: 点的坐标，这里也是指父节点的坐标
        :param nodeType: 节点类型,分为叶子节点和决策节点
        :return:
        """
        createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                                xytext=centerPt, textcoords='axes fraction',
                                va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)


    def getNumLeafs(myTree):
        """
        获取叶节点的数目
        :param myTree:
        :return:
        """
        # 统计叶子节点的总数
        numLeafs = 0

        # 得到当前第一个key，也就是根节点
        firstStr = list(myTree.keys())[0]

        # 得到第一个key对应的内容
        secondDict = myTree[firstStr]

        # 递归遍历叶子节点
        for key in secondDict.keys():
            # 如果key对应的是一个字典，就递归调用
            if type(secondDict[key]).__name__ == 'dict':
                numLeafs += getNumLeafs(secondDict[key])
            # 不是的话，说明此时是一个叶子节点
            else:
                numLeafs += 1
        return numLeafs


    def getTreeDepth(myTree):
        """
        得到数的深度层数
        :param myTree:
        :return:
        """
        # 用来保存最大层数
        maxDepth = 0

        # 得到根节点
        firstStr = list(myTree.keys())[0]

        # 得到key对应的内容
        secondDic = myTree[firstStr]

        # 遍历所有子节点
        for key in secondDic.keys():
            # 如果该节点是字典，就递归调用
            if type(secondDic[key]).__name__ == 'dict':
                # 子节点的深度加1
                thisDepth = 1 + getTreeDepth(secondDic[key])

            # 说明此时是叶子节点
            else:
                thisDepth = 1

            # 替换最大层数
            if thisDepth > maxDepth:
                maxDepth = thisDepth

        return maxDepth


    def plotMidText(cntrPt, parentPt, txtString):
        """
        计算出父节点和子节点的中间位置，填充信息
        :param cntrPt: 子节点坐标
        :param parentPt: 父节点坐标
        :param txtString: 填充的文本信息
        :return:
        """
        # 计算x轴的中间位置
        xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
        # 计算y轴的中间位置
        yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
        # 进行绘制
        createPlot.ax1.text(xMid, yMid, txtString)


    def plotTree(myTree, parentPt, nodeTxt):
        """
        绘制出树的所有节点，递归绘制
        :param myTree: 树
        :param parentPt: 父节点的坐标
        :param nodeTxt: 节点的文本信息
        :return:
        """
        # 计算叶子节点数
        numLeafs = getNumLeafs(myTree=myTree)

        # 计算树的深度
        depth = getTreeDepth(myTree=myTree)

        # 得到根节点的信息内容
        firstStr = list(myTree.keys())[0]

        # 计算出当前根节点在所有子节点的中间坐标,也就是当前x轴的偏移量加上计算出来的根节点的中心位置作为x轴（比如说第一次：初始的x偏移量为：-1/2W,计算出来的根节点中心位置为：(1+W)/2W，相加得到：1/2），当前y轴偏移量作为y轴
        cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)

        # 绘制该节点与父节点的联系
        plotMidText(cntrPt, parentPt, nodeTxt)

        # 绘制该节点
        plotNode(firstStr, cntrPt, parentPt, decisionNode)

        # 得到当前根节点对应的子树
        secondDict = myTree[firstStr]

        # 计算出新的y轴偏移量，向下移动1/D，也就是下一层的绘制y轴
        plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD

        # 循环遍历所有的key
        for key in secondDict.keys():
            # 如果当前的key是字典的话，代表还有子树，则递归遍历
            if isinstance(secondDict[key], dict):
                plotTree(secondDict[key], cntrPt, str(key))
            else:
                # 计算新的x轴偏移量，也就是下个叶子绘制的x轴坐标向右移动了1/W
                plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
                # 打开注释可以观察叶子节点的坐标变化
                # print((plotTree.xOff, plotTree.yOff), secondDict[key])
                # 绘制叶子节点
                plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
                # 绘制叶子节点和父节点的中间连线内容
                plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))

        # 返回递归之前，需要将y轴的偏移量增加，向上移动1/D，也就是返回去绘制上一层的y轴
        plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD


    def createPlot(inTree):
        """
        需要绘制的决策树
        :param inTree: 决策树字典
        :return:
        """
        # 创建一个图像
        fig = plt.figure(1, facecolor='white')
        fig.clf()
        axprops = dict(xticks=[], yticks=[])
        createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
        # 计算出决策树的总宽度
        plotTree.totalW = float(getNumLeafs(inTree))
        # 计算出决策树的总深度
        plotTree.totalD = float(getTreeDepth(inTree))
        # 初始的x轴偏移量，也就是-1/2W，每次向右移动1/W，也就是第一个叶子节点绘制的x坐标为：1/2W，第二个：3/2W，第三个：5/2W，最后一个：(W-1)/2W
        plotTree.xOff = -0.5 / plotTree.totalW
        # 初始的y轴偏移量，每次向下或者向上移动1/D
        plotTree.yOff = 1.0
        # 调用函数进行绘制节点图像
        plotTree(inTree, (0.5, 1.0), '')
        # 绘制
        plt.show()


    if __name__ == '__main__':
        createPlot(dicision_Tree)