# -*- coding: utf-8 -*-
"""
Created on Tue May  3 12:15:26 2022

@author: Hi
"""
import numpy as np
import pandas as pd

###############################################################################
# Loading the data
df = pd.read_csv("Anydomain.csv")

Y = df['X2'] # Y
X = df[['X4']]

# Import Linear Regression
from sklearn.linear_model import LinearRegression
lm = LinearRegression().fit(X,Y)   #lm.fit(input,output)


# Predicted with our model coeffcients
y_pred = lm.predict(X)
type(y_pred)
y_pred.shape

##### manual calculations
RSS = np.sum((y_pred - Y)**2) # Residual sum of squares
y_mean = np.mean(Y)
TSS = np.sum((Y - y_mean)**2)
R2 = 1 - (RSS/TSS)
vif = 1/(1-R2)
print ("VIF value :",vif)



###############################################################################
'''
0  const  10668.509471
1     X2    254.423166
2     X1     38.496211
3     X3     46.868386
4     X4    282.512865
'''
