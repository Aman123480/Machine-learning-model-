# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 13:55:08 2022

@author: Hi
"""

import pandas as pd  

df = pd.read_csv("D:\\CARRER\\My_Course\\Data Science Classes\\3 Module\\1 Supervised\\10 Decision Tree\\2 GINI\\Cricket.csv")  
df.shape
list(df)
df.head()

# Label encode
from sklearn.preprocessing import LabelEncoder 
LE = LabelEncoder()
df['Gender'] = LE.fit_transform(df['Gender'])
df['Class'] = LE.fit_transform(df['Class'])
df.head()

X = df.drop('Cricket', axis=1)  #Whether to drop labels from the index (0 or ‘index’) or columns (1 or ‘columns’)
Y = df['Cricket']

# model fitting
from sklearn.tree import DecisionTreeClassifier  
DT = DecisionTreeClassifier() 

DT.fit(X,Y)
Y_pred = DT.predict(X)

DT.tree_.node_count
DT.tree_.max_depth

from sklearn import metrics
cm = metrics.confusion_matrix(Y, Y_pred)
metrics.accuracy_score(Y,Y_pred).round(2)


##############################################################################
# pip install graphviz
from sklearn import tree
import graphviz 
dot_data = tree.export_graphviz(DT, out_file=None, 
                    filled=True, rounded=True,  
                    special_characters=True)  
graph = graphviz.Source(dot_data)  
graph


##############################################################################
import numpy as np           
x_new = np.array([[1,1]])  #-->  Yes
x_new = np.array([[1,0]])
x_new = np.array([[0,1]])
x_new = np.array([[0,0]])

DT.predict(x_new)
