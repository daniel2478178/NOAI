#!/usr/bin/env python
# coding: utf-8

# In[2]:


# svm_experiment
import os
import struct
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import svm #调用sklearn里面的svm函数
from sklearn.metrics import classification_report  # 预测结果分析工具

def load_datafromexcel(path): 
    #从dataset文件夹里面读取csv文件，为了方便起见，我们所有的文件都从dataset的文件夹里面读取, 
    #统一格式为：第0行为header行，第0列为index列，最后一列是label
    Sample_excel = pd.read_csv(path, header = 0, index_col = 0)
    Sample = Sample_excel.values  #转成numpy数组
    y = Sample[:,Sample.shape[1]-1]
    #print(y)
    X = Sample[:,0:Sample.shape[1]-1]
    #print(X)
    return X,y

if __name__ == '__main__':
    X_train, y_train = load_datafromexcel(path = 'svm_train.csv')  #读取训练集数据
    clf =  svm.SVC(C=10000000000, kernel = 'linear',degree = 1, tol = 1e-3) 
    #使用线性核函数的svm,C跟正则化系数成反比，所以C越大，表示越不考虑正则化
    # ，C取一个极大值代表完全不考虑正则化
    print(clf)
    clf.fit(X_train,y_train) #生成模型
    w = clf.coef_[0]
    b = clf.intercept_
    #print(w)
    a = -w[0]/w[1] #求支持向量的斜率
    #print(a)     

    b1 = clf.support_vectors_[1,1]-a*clf.support_vectors_[1,0]
    plt.plot(X_train[0:30,0], X_train[0:30,1],'b+') #画出label=0的数据
    plt.plot(X_train[30:60,0], X_train[30:60,1],'ro') #画出label=1的数据
    plt.plot(clf.support_vectors_[:,0],clf.support_vectors_[:,1],'kx')  #打印支持向量
    xx = np.arange(0,1,0.01)
    
    #----------SVM分类器的下边界————————————#
    temp0=0
    b0 = 0
    i = 0
    size = clf.support_.shape[0]
    while temp0 == 0 and i<size:
        b_temp = clf.support_vectors_[i,1]-a*clf.support_vectors_[i,0]
        if b_temp<b:
            b0 = b_temp
            temp0 = 1
        i = i + 1
    #---------SVM分类器的上边界-----------------------#
    temp1=0
    b1 = 0
    i = 0
    size = clf.support_.shape[0]
    while temp1 == 0 and i<size:
        b_temp = clf.support_vectors_[i,1]-a*clf.support_vectors_[i,0]
        if b_temp>b:
            b1 = b_temp
            temp1 = 1
        i = i + 1
        
    y0 = b0 + a * xx
    y1 = b1 + a * xx
    ycenter = b + a * xx
    #plt.plot(xx,ycenter,'k--')
    plt.plot(xx,y0,'k--')
    plt.plot(xx,y1,'k--')
    X_test, y_test = load_datafromexcel(path = 'svm_test.csv')  #读取测试集数据
    y_predict = clf.predict(X_test) #对测试集数据进行预测
    #plt.plot(X_test[:,0],X_test[:,1],'k^')
    plt.title('Supporting Vector Machine, the black is test data')
    print(classification_report(y_test, y_predict))
    
     



