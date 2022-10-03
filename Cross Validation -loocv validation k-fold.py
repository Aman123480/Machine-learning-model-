"""
Created on Tue Jul 12 13:01:28 2022
"""

import numpy as np
import pandas as pd
pd.options.display.float_format = '{:.3f}'.format

df = pd.read_csv("Hitters_final.csv")
df.shape
list(df)
###############################################################################

X = df.iloc[:,1:17]
X.shape
list(X)
X_new = X.drop(X.columns[[0,2,3,4,6,9,12,15]], axis=1)
list(X_new)

Y = df['Salary']

#=============================================================================

from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
X_scale = SS.fit_transform(X_new)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X_scale,Y, test_size=0.20)

#=============================================================================
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

LR = LinearRegression()

LR.fit(X_train,Y_train) # training model
Y_Pred_train = LR.predict(X_train)
Y_Pred_test = LR.predict(X_test)
train_error = np.sqrt(mean_squared_error(Y_train,Y_Pred_train)).round(2)
test_error = np.sqrt(mean_squared_error(Y_test,Y_Pred_test)).round(2)

print("Training error: ",train_error)
print("Test error: ",test_error)

#=============================================================================
# Validation set approach 

trainingerror = []
testerror = []


for i in range(1,1000):
    X_train,X_test,Y_train,Y_test = train_test_split(X_scale,Y, test_size=0.20,random_state=i)
    LR.fit(X_train,Y_train)
    Y_Pred_train = LR.predict(X_train)
    Y_Pred_test = LR.predict(X_test)
    trainingerror.append(np.sqrt(mean_squared_error(Y_train,Y_Pred_train)).round(2))
    testerror.append(np.sqrt(mean_squared_error(Y_test,Y_Pred_test)).round(2))
    
#print(trainingerror)
#print(testerror)

#min(testerror)
#max(testerror)
    
y_out = pd.DataFrame(testerror)
yt_out = pd.DataFrame(trainingerror)

#y_out.boxplot(column=[0])
#y_out.hist()
y_out.mean()
vs_error = yt_out.mean()
print("validationset - testerror",vs_error)


#=============================================================================

# Evaluate using Cross Validation
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score

loocv = LeaveOneOut()
results = abs(cross_val_score(LR, X_scale, Y, cv=loocv, scoring='neg_mean_squared_error'))

results
loocv = np.sqrt(np.mean(results)).round(2)
print("Loocv - testerror",loocv)

#=============================================================================

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
kfold = KFold(n_splits=5)
results = abs(cross_val_score(LR, X_scale, Y, cv=kfold, scoring='neg_mean_squared_error'))
results
kfoldvalue = np.sqrt(np.mean(results)).round(2)
print("K fold - testerror",kfoldvalue)

