"""
Created on Fri Jun 10 13:09:15 2022
"""
# step1: import the data set

import pandas as pd
df = pd.read_csv("D:\\Recordings\\GeeKLurn\\Data\\Regression\\Advertising.csv")
df.shape
list(df)

# step2: split as X and Y variables
Y = df["sales"]
X = df[['TV','radio','newspaper']]



X.iloc[:,0]

# step3: check the relationships
import matplotlib.pyplot as plt
plt.scatter(X.iloc[:,0], Y,  color='green')
plt.show()

import matplotlib.pyplot as plt
plt.scatter(X.iloc[:,1], Y,  color='blue')
plt.show()

import matplotlib.pyplot as plt
plt.scatter(X.iloc[:,2], Y,  color='purple')
plt.show()

df.corr()

#--------------------------------------------------
#X = df[['TV']]
#X = df[['TV','radio']]
X = df[['TV','radio','newspaper']]


# step4: Fit the model
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)

LR.intercept_ # Bo
LR.coef_ # B1

# step5: prediction
Y_pred = LR.predict(X)

# step6: Evaluating the model
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y,Y_pred)
mse

import numpy as np
RMSE = np.sqrt(mse)
print("Root mean squared error for the given model is: ", RMSE.round(3))


