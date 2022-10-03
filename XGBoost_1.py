"""
Created on Tue Aug 17 09:35:48 2021

"""

import warnings
warnings.filterwarnings('ignore')

# pip install xgboost

#Importing various models to compare

# loading  data from csv
#from numpy import loadtxt
#mydata = loadtxt('pima-indians-diabetes.csv', delimiter=",")
import pandas as pd
mydata = pd.read_csv("pima-indians-diabetes.csv")

# spliting data into independent and dependent features
X = mydata.iloc[:,0:8]
Y = mydata.iloc[:,8]

import pandas as pd
pd.DataFrame(mydata).shape

# split data into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=1)

#Running various models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# pip install xgboost
from xgboost import XGBClassifier
XGBClassifier() # eta=0.001,gamma=10,learning_rate=1,reg_lambda=1,n_estimators=100

models = []
models.append(('LogisticRegression', LogisticRegression()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('SVM', SVC()))
models.append(('Decision Tree Classifier', DecisionTreeClassifier()))
models.append(('Random Forest Classifier', RandomForestClassifier(max_depth=0.7)))
models.append(('XGB',XGBClassifier(gamma=10,reg_lambda=4))) #eta = 0.01,gamma = 10

type(models[0])
models[0][0] # name
models[0][1] # function

# Importing needed packages
from sklearn.metrics import accuracy_score

#import time
# evaluate each model in turn
#results = []
#names = []

for name, model in models:
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]

    # evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0),name)
    


'''
from xgboost import XGBClassifier
XGBc = XGBClassifier()

from xgboost import XGBRegressor
XGBr = XGBRegressor()
'''

                                 
    