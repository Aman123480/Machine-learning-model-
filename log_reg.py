"""
Created on Mon Jun 27 13:44:33 2022

"""

#import pandas
import pandas as pd
df = pd.read_csv("D:\\CARRER\\My_Course\\Data Science Classes\\3 Module\\1 Supervised\\7 Logistic Regression\\mlbench\\breast_cancer.csv")
df.shape
df.head()

df.info()

# working on label encoding
from sklearn.preprocessing import LabelEncoder
lb_Class = LabelEncoder()
df["Class"] = lb_Class.fit_transform(df["Class"])
df.head()

# Split the data in to independent and Dependent
X = df.iloc[:,1:10] # Only Independent variables
list(X)
Y = df.iloc[:,10]  # Only Dependent variable Sales
list(Y)

# import model  --> always expect numerical inputs
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X,Y)

# predictions
Y_pred = logreg.predict(X)

#=============================================================

# metrics
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y,Y_pred)
cm

from sklearn.metrics import accuracy_score, recall_score, precision_score,f1_score
ac = accuracy_score(Y,Y_pred)
rs = recall_score(Y,Y_pred)
ps = precision_score(Y,Y_pred)
fs = f1_score(Y,Y_pred)

TN = cm[0,0]
TP = cm[1,1]
FP = cm[0,1]
FN = cm[1,0]
sp = TN/(TN + FP)

print("Accuracy score: ",ac.round(2))
print("Sensitivity/Recall score: ",rs.round(2))
print("Specificity score: ",sp.round(2))
print("precision score: ",ps.round(2))
print("F1 score: ",fs.round(2))

#=============================================================
