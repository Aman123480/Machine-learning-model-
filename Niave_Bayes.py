"""
Created on Tue Jul 26 12:15:30 2022
"""
#============================================================
import pandas as pd
df =  pd.read_csv("mushroom.csv")
df.shape
df.info()

#============================================================
df.isnull().sum()


df["Typeofmushroom"].value_counts()

df.groupby(df.columns.values[0]).size()

for i in range(0,23):
    print(df.groupby(df.columns.values[i]).size(),"\n ******************************")
    
    
df["stalkroot"].value_counts()    
df["veiltype"].value_counts()

#============================================================
list(df)
df.drop(df.columns[[11,16]],axis=1,inplace=True)
list(df)
df.shape

#============================================================
#
# Label encode
from sklearn import preprocessing
LE = preprocessing.LabelEncoder()

for i in range(1,21):
     df.iloc[:,i] = LE.fit_transform(df.iloc[:,i])
    
df.head()
#============================================================

Y = df["Typeofmushroom"]
X = df.iloc[:,1:]

#============================================================
# splitting the data into Train and Test
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,stratify=Y,test_size = 0.30)
X_train.shape, X_test.shape, Y_train.shape, Y_test.shape


#============== MultinomialNB ==================================================

from sklearn.naive_bayes import MultinomialNB
MNB = MultinomialNB()
MNB_model = MNB.fit(X_train,Y_train)

# apply the model on test
Y_pred = MNB_model.predict(X_test)
print(Y_pred)

# import confusion matrix and Accuracy
from sklearn import metrics
cm = metrics.confusion_matrix(Y_test, Y_pred)
print(cm)  
metrics.accuracy_score(Y_test,Y_pred).round(5)

#========================================================================

# apply the logistic regression --> target variable should be label encoded
# apply the KNN classifier

















