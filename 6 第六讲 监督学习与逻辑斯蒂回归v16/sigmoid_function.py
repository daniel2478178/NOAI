# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 21:05:50 2023

@author: NING MEI
"""


import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-10,10,0.01)
theta0 = 2
theta1 = 1
# theta1 改变
y = 1/(1+np.exp(-(theta0+theta1*x)))
plt.figure(1)
plt.plot(x,y)
theta1 = 2
y = 1/(1+np.exp(-(theta0+theta1*x)))
plt.plot(x,y,'--')
theta1 = 0.5
y = 1/(1+np.exp(-(theta0+theta1*x)))
plt.plot(x,y,'--')


# theta0 改变
theta1 = 1
theta0 = -2
plt.figure(2)
y = 1/(1+np.exp(-(theta0+theta1*x)))
plt.plot(x,y)
theta0 = 0
y = 1/(1+np.exp(-(theta0+theta1*x)))
plt.plot(x,y,'--')
theta0 = 2
y = 1/(1+np.exp(-(theta0+theta1*x)))
plt.plot(x,y,'--')
