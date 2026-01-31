import numpy as np
import os 
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import classification_report  # 预测回归结果分析工具
from sklearn.tree import plot_tree
epislon = np.finfo(np.float64).tiny
def cal_entropy(y):
    """计算标签的信息熵"""
    unique_labels, counts = np.unique(y, return_counts=True)
    prob = counts / len(y)
    
    entropy = - np.sum(prob * np.log2(prob +epislon)) 
    
    return entropy


   
def load_datafromcsv(path): 
    #统一格式为：第0行为header行，第0列为index列，最后一列是label
    df = pd.read_csv(path, header = 0, index_col = 0)
    X = df[df.columns[:-1]].values
    y = df[df.columns[-1]].values

    return X,y

def draw_figure(X,y,clf): #画图函数
    h = .02  # step size in the mesh
    # Create color maps
    cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue'])
    cmap_bold = ListedColormap(['darkorange', 'c', 'darkblue'])
    # Draw mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    y_predict=clf.predict(np.c_[xx.ravel(), yy.ravel()])    
    #Put the result into a color plot
    y_predict = y_predict.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, y_predict, cmap=cmap_light)
    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("Iris 2 feature 2 label")   
    return

class DDTreeNode:
    def __init__(self,featureind,threshhold = None):
        self.featureind = featureind

        self.threshhold = threshhold
        self.left = None
        self.right = None
        self.isleaf = False
        self.label = None
    def calc_weighted_entropy(self,features,y):
        left_mask = features <= self.threshhold
        right_mask = features > self.threshhold
        left_y = y[left_mask]
        right_y = y[right_mask]
        left_ent = cal_entropy(left_y)
        right_ent = cal_entropy(right_y)
        weighted_ent = (len(left_y) / len(y)) * left_ent + (len(right_y) / len(y)) * right_ent
        return weighted_ent
    def bestThreshhold(self,features,y):
        if len(np.unique(y)) == 1:
            return None, 0, 0
        features = features[:,self.featureind]
        sorted_indices = np.argsort(features)
        features = features[sorted_indices]
        y = y[sorted_indices]
        baseEntropy = cal_entropy(y)
        min_entropy = float('inf')
        possibleTreshhold = (features[:-1] + features[1:])/2
        best_info_gain = -np.inf
        best_threshhold = None
        min_entropy = float('inf')
        for threshhold in possibleTreshhold[1:-1]:
            self.threshhold = threshhold
            weighted_ent = self.calc_weighted_entropy(features,y)
            info_gain = baseEntropy - weighted_ent
            if weighted_ent < min_entropy:
                min_entropy = weighted_ent
                best_threshhold = threshhold
                best_info_gain = info_gain
        
        
        return best_threshhold, min_entropy, best_info_gain
class DecisionTree:
    def __init__(self,features_names,max_depth = np.inf):
        self.feature_names = features_names
        self.decisionTree = DDTreeNode(featureind = None)
        self.max_depth = max_depth  
    def makeNode(self,features,y,depth):
        if len(np.unique(y)) == 1 and depth <= self.max_depth:
            leafNode = DDTreeNode(featureind = None)
            leafNode.isleaf = True
            leafNode.label = np.unique(y)[0]
            return leafNode
        possibleTests = []
        
        for f in range(len(self.feature_names)):
            nod = DDTreeNode(f)
            nod.threshhold, min_entropy, info_gain = nod.bestThreshhold(features,y)
            possibleTests.append((nod, min_entropy, info_gain))
        possibleTests.sort(key=lambda x: x[2], reverse=True)
        bestNode, best_entropy, best_info_gain = possibleTests[0]
        if best_info_gain <= 0 :
            leafNode = DDTreeNode(featureind = None)
            leafNode.isleaf = True
            l, coun = np.unique(y,True )
            leafNode.label = l[np.argmax(coun)]
            return leafNode
        
        leftmask = features[:,bestNode.featureind] <= bestNode.threshhold
        rightmask = features[:,bestNode.featureind] > bestNode.threshhold
        bestNode.left = self.makeNode(features[leftmask],y[leftmask], depth+1)
        bestNode.right = self.makeNode(features[rightmask],y[rightmask], depth+1)
        return bestNode
    def fit(self,features,y):
        self.decisionTree = self.makeNode(features,y, depth = 0)
    def p(self,sample):
        curNode = self.decisionTree
        while not curNode.isleaf:
            if sample[curNode.featureind] <= curNode.threshhold:
                curNode = curNode.left
            else:
                curNode = curNode.right
        return curNode.label
    def predict(self,features):
        return [self.p(sample) for sample in features]
class RandomForest:
    def __init__(self, n_estimators, max_depth = np.inf,random_state=42):
        np.random.seed(random_state)
        self.n_trees = n_estimators
        self.trees = self.estimators_  = []
        self.max_depth = max_depth
    def fit(self, features, y):
        n_samples, n_features = features.shape
        for _ in range(self.n_trees):
            choiceindices = np.random.choice(n_samples, n_samples, replace=True)
            ychosed = y[choiceindices]
            featurechosed = features[choiceindices]
            choicefeatures = np.random.choice(n_features,int(np.sqrt(n_features)),replace = False)
            choosedfeature = featurechosed[:,choicefeatures]
            tree = DecisionTree(["f{i}"for i in range (choosedfeature.shape[1])],self.max_depth)
            tree.fit(choosedfeature,ychosed)
            self.trees.append((tree, choicefeatures))
    def predict(self, features):
        tree_preds = []
        for tree, feature_indices in self.trees:
            selected_features = features[:, feature_indices]
            preds = tree.predict(selected_features)
            tree_preds.append(preds)
        tree_preds = np.array(tree_preds)
        final_preds = []
        for i in range(features.shape[0]):
            counts = np.bincount(tree_preds[:, i].astype(np.int64))
            final_preds.append(np.argmax(counts))
        return final_preds
    def __str__(self):
        # plotting tree structure and visulize using sklearn, return string of tree structure
        return f"RandomForest with {self.n_trees} trees, max_depth={self.max_depth}"
        
        
        
            

    

            
            
        
        
        
        
                
            
            
            
        
        
    

