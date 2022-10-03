"""
Created on Tue Jul  5 13:39:51 2022
"""

import pandas as pd
df = pd.read_csv("Boston.csv")
df.shape
df.head()

# splt the vairbales as x and y 
X = df.iloc[:,1:14]
list(X)

Y = df["medv"]

# standardization
from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
X_scale = SS.fit_transform(X)

# train test split commands
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X_scale,Y,test_size=0.20,random_state=42)

X_train.shape
X_test.shape
Y_train.shape
Y_test.shape

# model development
from sklearn.linear_model import LinearRegression
lr = LinearRegression()

lr.fit(X_train, Y_train)

# predictins on Train data
Y_pred_train = lr.predict(X_train)
# predictins on Test data
Y_pred_test = lr.predict(X_test)

from sklearn.metrics import mean_squared_error
Training_error = mean_squared_error(Y_train,Y_pred_train)
Test_error = mean_squared_error(Y_test,Y_pred_test)

print("MSE on Training data:",Training_error.round(3))
print("MSE on Test data:",Test_error.round(3))


