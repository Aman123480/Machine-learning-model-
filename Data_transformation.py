# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 13:18:59 2022

@author: Hi
"""

import numpy as np
import pandas as pd
###############################################################################
# Loading the data
df = pd.read_csv("Carseats.csv")
df.shape
df.head()
list(df)

Y = df["Sales"]



# ShelveLoc
# Urban
# US
df["ShelveLoc"].value_counts()



from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()

df["ShelveLoc"] = LE.fit_transform(df["ShelveLoc"]) 
df["Urban"] = LE.fit_transform(df["Urban"]) 
df["US"] = LE.fit_transform(df["US"]) 

df["ShelveLoc"].value_counts()

X = df.iloc[:,2:12]
list(X)
X.info()

## Individual Coeffcient Test,  R2 Square, R2 Adjusted with statsmodels
import statsmodels.api as sm
X1 = sm.add_constant(X) ## let's add an intercept (beta_0) to our model
lm2 = sm.OLS(Y,X1).fit()

#he summary of our model
lm2.summary()

#==========================================================================
#==========================================================================
#==========================================================================

# Loading the data
df = pd.read_csv("Carseats.csv")
df.shape
df.head()
list(df)

X_cat = df[['ShelveLoc','Urban','US']]

from sklearn.preprocessing import OneHotEncoder
OHE = OneHotEncoder()

dummyx = OHE.fit_transform(X_cat).toarray()

dummyx = pd.DataFrame(dummyx)

df_new = pd.concat([df,dummyx],axis=1)

list(df_new)

X_new = df_new.drop(df.columns[[0,1,5,7,9,10,11,12]], axis=1)
X_new

Y

## Individual Coeffcient Test,  R2 Square, R2 Adjusted with statsmodels
import statsmodels.api as sm
X1 = sm.add_constant(X_new) ## let's add an intercept (beta_0) to our model
lm2 = sm.OLS(Y,X1).fit()

#he summary of our model
lm2.summary()

















