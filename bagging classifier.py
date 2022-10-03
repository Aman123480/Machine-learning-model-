"""
Created on Tue Aug  2 12:53:41 2022
"""

import pandas as pd
df = pd.read_csv("sales.csv")
df.shape

list(df)

df.dtypes

# Label encode
from sklearn.preprocessing import LabelEncoder 
LE = LabelEncoder()
df['ShelveLoc'] = LE.fit_transform(df['ShelveLoc'])
df['Urban'] = LE.fit_transform(df['Urban'])
df['US'] = LE.fit_transform(df['US'])
df.head()

# split the X and Y
Y = df["high"]
X = df.iloc[:,1:11]
list(X)

# split the data in train and test
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,random_state=42)


# model fitting
from sklearn.tree import DecisionTreeClassifier  
DT = DecisionTreeClassifier(max_depth=6) 

DT.fit(X_train,Y_train)

Y_pred_train = DT.predict(X_train)
Y_pred_test = DT.predict(X_test)

DT.tree_.node_count
DT.tree_.max_depth

# metrics
from sklearn.metrics import accuracy_score
Training_accuracy = accuracy_score(Y_train,Y_pred_train).round(2)
Test_accuracy = accuracy_score(Y_test,Y_pred_test).round(2)

print("Training_accuracy: ",Training_accuracy)
print("Test_accuracy: ",Test_accuracy)


#==================================================================
# bagging classifier  =========== bagging regressor

from sklearn.ensemble import BaggingClassifier
DT = DecisionTreeClassifier() 


# case1:
Bag = BaggingClassifier(base_estimator = DT,max_samples=0.6,bootstrap=True)
Bag.fit(X,Y)
Y_pred = Bag.predict(X)
ac_score = accuracy_score(Y,Y_pred).round(2)
print(ac_score)

# case2:
Bag = BaggingClassifier(base_estimator = DT,max_samples=0.6,bootstrap=False)
Bag.fit(X,Y)
Y_pred = Bag.predict(X)
ac_score = accuracy_score(Y,Y_pred).round(2)
print(ac_score)

# case3:
Bag = BaggingClassifier(base_estimator = DT,max_samples=0.6,bootstrap=True,max_features=0.7)
Bag.fit(X,Y)
Y_pred = Bag.predict(X)
ac_score = accuracy_score(Y,Y_pred).round(2)
print(ac_score)

# case4:
Bag = BaggingClassifier(base_estimator = DT,max_samples=0.6,bootstrap=False,max_features=0.7)
Bag.fit(X,Y)
Y_pred = Bag.predict(X)
ac_score = accuracy_score(Y,Y_pred).round(2)
print(ac_score)


# case5:
Bag = BaggingClassifier(base_estimator = DT,bootstrap=True)
Bag.fit(X,Y)
Y_pred = Bag.predict(X)
ac_score = accuracy_score(Y,Y_pred).round(2)
print(ac_score)
