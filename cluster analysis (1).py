"""
Created on Thu Sep  1 13:24:47 2022
"""

import pandas as pd
df = pd.read_csv("D:\\Recordings\\GeeKLurn\\Data\\shopping_data.csv")
df

X = df.iloc[:,2:]

from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
X_scale = SS.fit_transform(X)

X_scale[:,0:]


# construct the dendrogram
import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 7))  
plt.title("Customer Dendograms")  
dend = shc.dendrogram(shc.linkage(X_scale[:,1:], method='average')) 


import matplotlib.pyplot as plt
plt.scatter(X_scale[:,1:2],X_scale[:,2:])
plt.show

# algorithm selection
from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='average')
Y = cluster.fit_predict(X_scale)

Y_new = pd.DataFrame(Y)
Y_new[0].value_counts()

#====================================================

%matplotlib qt
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (16, 9)

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X.iloc[:, 0], X.iloc[:, 1], X.iloc[:, 2])
plt.show()


# Initializing KMeans
from sklearn.cluster import KMeans
KMeans()
kmeans = KMeans(n_clusters=5,n_init=20)

# Fitting with inputs
kmeans = kmeans.fit(X_scale)

# Predicting the clusters
labels = kmeans.predict(X_scale)

y = pd.DataFrame(labels)
y[0].value_counts()

kmeans.inertia_

#===========================================================




















