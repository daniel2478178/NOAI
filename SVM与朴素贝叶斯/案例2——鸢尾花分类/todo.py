import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn import neighbors, linear_model
from sklearn.neural_network import MLPClassifier
from sklearn import svm #调用sklearn里面的svm函数
from sklearn.metrics import classification_report #调用sklearn里面的svm函数
os.chdir(os.path.dirname(os.path.abspath(__file__)))
def load_data(path): 
    #统一格式为：第0行为header行，第0列为index列，最后一列是label
    Sample_excel = pd.read_csv(path, header = 0, index_col = 0)
    Sample = Sample_excel.values  #转成numpy数组
    y = Sample[:,Sample.shape[1]-1]
    X = Sample[:,0:Sample.shape[1]-1]
    return X,y
def draw_figure(X, y, clf):
    """
    绘制【分类决策边界+样本点】的核心可视化函数
    :param X: 特征数据（二维）
    :param y: 标签数据
    :param clf: 训练好的分类模型（KNN/逻辑回归/神经网络）
    """
    type1place = (y==0)
    type2place = (y==1)
    for data in X[type1place]:
   
        plt.scatter(data[0],data[1],color = 'r')
    for data in X[type2place]:
        plt.scatter(data[0],data[1],color = 'b')
    xaxis = np.arange(np.min(X[:,0],axis = 0),np.max(X[:,0],axis = 0),0.08)
    yaxis = np.arange(np.min(X[:,1],axis = 0),np.max(X[:,1],axis = 0),0.08)
    #print(xaxis,yaxis)
    xxdots = []
    yydots = []
    for i,xx in enumerate(xaxis):
        for j,yy in enumerate(yaxis):
            if i == 0 or j == 0:
                continue
            prediction = lambda xx,yy:int(clf(np.array([[xx,yy]])))
            if  (prediction(xx,yy) != prediction(xaxis[i-1],yy)) or ((prediction(xx,yaxis[j-1]) != prediction(xx,yy))):   
                xxdots.append(xx)
                yydots.append(yy)
    #print(xxdots,yydots)
          
    plt.plot(xxdots,yydots)
    
    
# 支持向量机训练模型
clf = svm.SVC(C=10000000000, kernel='rbf', tol=1e-3)  # rbf核支持向量机

# 支持向量机训练模型
clf = svm.SVC(C=10000000000, kernel='linear', tol=1e-3)  # 线性支持向量机

X_train, y_train = load_data(path = 'iris_2feature2label_train.csv')  #读取训练集数据
X_test, y_test = load_data(path = 'iris_2feature2label_test.csv')  #读取训练集数据

clf.fit(X_train,y_train) #生成模型
y_predict = clf.predict(X_test) #测试集预测
print(classification_report(y_test, y_predict)) #输出预测结果分析报告
draw_figure(X_test, y_test, clf.predict)  # 画图
plt.title("SVM with linear kernel on Iris 2 feature 2 label")
plt.show()
