"""
Created on Fri Jul  8 13:05:13 2022
"""
import numpy as np
import pandas as pd

df = pd.read_csv("Hitters_final.csv")
df.shape
list(df)

Y = df["Salary"]
X = df.iloc[:,1:17]
list(X)

# standardization
from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
X_scale = SS.fit_transform(X)

#=================================================================

# data partition
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X_scale,Y, test_size=0.20, random_state=10)
X_train.shape,X_test.shape,Y_train.shape,Y_test.shape

from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X_train,Y_train) # training model
Y_Pred_train = LR.predict(X_train)
Y_Pred_test = LR.predict(X_test)

from sklearn.metrics import mean_squared_error
training_error = np.sqrt(mean_squared_error(Y_train,Y_Pred_train))
test_error     = np.sqrt(mean_squared_error(Y_test,Y_Pred_test) )
print("MSE on Training data:", training_error.round(3))
print("MSE on Test data:", test_error.round(3))

#=================================================================
# Ridge Regression
from sklearn.linear_model import Ridge
RR = Ridge(alpha=1000)
RR.fit(X_scale,Y)
RR.coef_

from sklearn.metrics import mean_squared_error
Y_pred = RR.predict(X_scale)
mse = mean_squared_error(Y,Y_pred)
print("Mean squared Error: ", mse.round(3))
import numpy as np
(np.sqrt(mse)).round(3)

pd.concat([pd.DataFrame(X.columns),pd.DataFrame(RR.coef_)], axis=1)

#=================================================================

# Lasso Regression
from sklearn.linear_model import Lasso
LS = Lasso(alpha=100)
LS.fit(X_scale,Y)
LS.coef_

from sklearn.metrics import mean_squared_error
Y_pred = LS.predict(X_scale)
mse = mean_squared_error(Y,Y_pred)
print("Mean squared Error: ", mse.round(3))
import numpy as np
print("RMSE: " , (np.sqrt(mse)).round(3))

pd.concat([pd.DataFrame(X.columns),pd.DataFrame(LS.coef_)], axis=1)

#=================================================================
X_scale = pd.DataFrame(X_scale)
X_new = X_scale.drop(X_scale.columns[[0,2,3,6,7,8,9,12,13,14,15]], axis=1)
list(X_new)

#=================================================================

# data partition
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X_new,Y, test_size=0.20, random_state=10)
X_train.shape,X_test.shape,Y_train.shape,Y_test.shape

from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X_train,Y_train) # training model
Y_Pred_train = LR.predict(X_train)
Y_Pred_test = LR.predict(X_test)

from sklearn.metrics import mean_squared_error
training_error = np.sqrt(mean_squared_error(Y_train,Y_Pred_train))
test_error     = np.sqrt(mean_squared_error(Y_test,Y_Pred_test) )
print("MSE on Training data:", training_error.round(3))
print("MSE on Test data:", test_error.round(3))




