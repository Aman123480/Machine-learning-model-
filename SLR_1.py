# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 14:02:55 2022

@author: Hi
"""

import numpy as np
x = np.array([23 ,26 ,26 ,27 ,28 ,29 ,29 ,30 ,35])
x.ndim
X = x[:, np.newaxis]
X.ndim

y = np.array([70 ,69 ,72 ,75 ,75 ,75 ,80 ,81 ,83])
y.ndim

from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X,y) #--> mx + c

lm.intercept_  #--> constant C
lm.coef_ #  m-->slope value

lm.predict(X)
y
# 39.0462500000000 + [1.2987*X]
# C                +   m x  
#=================================================================

# step1: import the data set
import pandas as pd
df = pd.read_csv("Viralcount_Drug.csv")
df

#
df.plot.scatter(x='drug',y='Viralcount')


# step2: split as X and Y variables
X = df[['drug']]
X.ndim
Y = df['Viralcount']
 

# step3: check the relationships
import matplotlib.pyplot as plt
plt.scatter(X, Y,  color='black')
plt.show()

df.corr()

# -0.949653  --> strong negative relationship

# step4: Fit the model
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)

LR.intercept_ # Bo
LR.coef_ # B1

# step5: prediction
Y_pred = LR.predict(X)

import matplotlib.pyplot as plt
plt.scatter(X, Y,  color='black')
plt.scatter(X, Y_pred,  color='red')
plt.show()

import matplotlib.pyplot as plt
plt.scatter(X, Y,  color='black')
plt.plot(X, Y_pred,  color='red')
plt.show()

# step6: Evaluating the model
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y,Y_pred)
mse

RMSE = np.sqrt(mse)
print("Root mean squared error for the given model is: ", RMSE.round(3))

#####################################################
# test data point

x2 = np.array([[10],[15],[20]])
LR.predict(x2)






























