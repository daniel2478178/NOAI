# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 15:05:33 2024

@author: NING MEI
"""

import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_gaussian_quantiles
from sklearn.ensemble import AdaBoostClassifier
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.tree import DecisionTreeClassifier

# Construct dataset
X1, y1 = make_gaussian_quantiles(
    cov=2.0, n_samples=200, n_features=2, n_classes=2, random_state=1
)
X2, y2 = make_gaussian_quantiles(
    mean=(3, 3), cov=1.5, n_samples=300, n_features=2, n_classes=2, random_state=1
)
X = np.concatenate((X1, X2))
y = np.concatenate((y1, -y2 + 1))

def plot_Adaboost(X,y,n_estimators):
    # Create and fit an AdaBoosted decision tree （为了体现效果，决策树的层级只有一层）
    bdt = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=n_estimators
    )
    
    bdt.fit(X, y)
    
    plot_colors = "br"
    plot_step = 0.02
    class_names = "AB"
    
    plt.figure(figsize=(10, 5))
    
    # Plot the decision boundaries
    ax = plt.subplot(121)
    disp = DecisionBoundaryDisplay.from_estimator(
        bdt,
        X,
        cmap=plt.cm.Paired,
        response_method="predict",
        ax=ax,
        xlabel="x",
        ylabel="y",
    )
    x_min, x_max = disp.xx0.min(), disp.xx0.max()
    y_min, y_max = disp.xx1.min(), disp.xx1.max()
    plt.axis("tight")
    
    # Plot the training points
    for i, n, c in zip(range(2), class_names, plot_colors):
        idx = np.where(y == i)
        plt.scatter(
            X[idx, 0],
            X[idx, 1],
            c=c,
            s=20,
            edgecolor="k",
            label="Class %s" % n,
        )
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.legend(loc="upper right")
    plt.title("Decision Boundary with " + str(n_estimators) + " Estimators")
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.35)

plot_Adaboost(X,y,n_estimators = 1)
plot_Adaboost(X,y,n_estimators = 5)
plot_Adaboost(X,y,n_estimators = 10)
plot_Adaboost(X,y,n_estimators = 50)
plot_Adaboost(X,y,n_estimators = 200)
plot_Adaboost(X,y,n_estimators = 1000)

