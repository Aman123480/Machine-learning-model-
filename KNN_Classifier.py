"""
Created on Fri Jul 15 12:49:21 2022

"""

import pandas as pd
df = pd.read_csv("breast-cancer-wisconsin-data.csv")
df.shape
df.head()

# Split the data in to independent and Dependent
X = df.iloc[:,2:] # Only Independent variables
list(X)
Y = df.iloc[:,1]  # Only Dependent variable Sales
list(Y)


# standardization
from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
SS

X_scale = SS.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X_scale,Y,random_state = 42)
# test size = 0.25, random_state = 

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5, p=2) # p=2 --> eucledian
knn.fit(X_train,Y_train)

Y_pred_train = knn.predict(X_train)
Y_pred_test = knn.predict(X_test)

from sklearn.metrics import accuracy_score
Training_accuracy = accuracy_score(Y_train,Y_pred_train)
Test_accuracy = accuracy_score(Y_test,Y_pred_test)

print("Training_accuracy:" , Training_accuracy.round(3))
print("Test_accuracy:" , Test_accuracy.round(3))


#========================================================

Training_accuracy = []
Test_accuracy = []

kvalue = range(1,20,2)

for i in kvalue:
    knn = KNeighborsClassifier(n_neighbors=i, p=2) # p=2 --> eucledian
    knn.fit(X_train,Y_train)
    Y_pred_train = knn.predict(X_train)
    Y_pred_test = knn.predict(X_test)
    Training_accuracy.append(accuracy_score(Y_train,Y_pred_train).round(3))
    Test_accuracy.append(accuracy_score(Y_test,Y_pred_test).round(3))

    
print(Training_accuracy)    
print(Test_accuracy)


t1 = pd.DataFrame(range(1,20,2))
t2 = pd.DataFrame(Training_accuracy)
t3 = pd.DataFrame(Test_accuracy)

l1 = pd.concat([t1,t2,t3], axis=1) # axis =0
l1


#===============================================================
















