import pandas as pd

from sklearn.datasets import make_blobs

X1, Y1 = make_blobs(n_features=100, 
         n_samples=1000,
         centers=2, random_state=4,
         cluster_std=2)

Y = pd.DataFrame(Y1)
Y[0].value_counts()
##########################

df = pd.read_csv("D:\\CARRER\\My_Course\\Data Science Classes\\3 Module\\2 Unsupervised\\2 Principle Component Analysis\\PCA\\100_data_names.csv")
df

list(df)

X1 = pd.DataFrame(df)
X1.shape
X1.head()

#X1["X1_20"].hist()


##########################
# load decomposition to do PCA analysis with sklearn
from sklearn.decomposition import PCA
PCA()
pca = PCA()

pc = pca.fit_transform(X1)
pca.explained_variance_ratio_
sum(pca.explained_variance_ratio_)

pc.shape
pd.DataFrame(pc).head()
type(pc)



pc_df = pd.DataFrame(data = pc , columns = ['P0C1', 'P0C2','P0C3','P0C4','P0C5','P0C6','P0C7','P0C8','P096','P0C10','P1C1', 'P1C2','P1C3','P1C4','P1C5','P1C6','P1C7','P1C8','P196','P1C10','P2C1', 'P2C2','P2C3','P2C4','P2C5','P2C6','P2C7','P2C8','P296','P2C10','P3C1', 'P3C2','P3C3','P3C4','P3C5','P3C6','P3C7','P3C8','P396','P3C10','P4C1', 'P4C2','P4C3','P4C4','P4C5','P4C6','P4C7','P4C8','P496','P4C10','P5C1', 'P5C2','P5C3','P5C4','P5C5','P5C6','P5C7','P5C8','P596','P5C10','P6C1', 'P6C2','P6C3','P6C4','P6C5','P6C6','P6C7','P6C8','P696','P6C10','P7C1', 'P7C2','P7C3','P7C4','P7C5','P7C6','P7C7','P7C8','P796','P7C10','P8C1', 'P8C2','P8C3','P8C4','P8C5','P8C6','P8C7','P8C8','P896','P8C10','P9C1', 'P9C2','P9C3','P9C4','P9C5','P9C6','P9C7','P9C8','P996','P9C10'])
pc_df.head()
pc_df.shape
type(pc_df)

#pc_df["P9C10"].describe()

#pc_df.to_csv("D:/CARRER/My_Course/DigitalLync/2 Module/Unsupervised/2 Principle Component Analysis/PC100.csv")


"""
variance explained by each principal component is called Scree plot.
"""
import seaborn as sns
df1 = pd.DataFrame({'var':pca.explained_variance_ratio_,
                  'PC':['P0C1', 'P0C2','P0C3','P0C4','P0C5','P0C6','P0C7','P0C8','P096','P0C10','P1C1', 'P1C2','P1C3','P1C4','P1C5','P1C6','P1C7','P1C8','P196','P1C10','P2C1', 'P2C2','P2C3','P2C4','P2C5','P2C6','P2C7','P2C8','P296','P2C10','P3C1', 'P3C2','P3C3','P3C4','P3C5','P3C6','P3C7','P3C8','P396','P3C10','P4C1', 'P4C2','P4C3','P4C4','P4C5','P4C6','P4C7','P4C8','P496','P4C10','P5C1', 'P5C2','P5C3','P5C4','P5C5','P5C6','P5C7','P5C8','P596','P5C10','P6C1', 'P6C2','P6C3','P6C4','P6C5','P6C6','P6C7','P6C8','P696','P6C10','P7C1', 'P7C2','P7C3','P7C4','P7C5','P7C6','P7C7','P7C8','P796','P7C10','P8C1', 'P8C2','P8C3','P8C4','P8C5','P8C6','P8C7','P8C8','P896','P8C10','P9C1', 'P9C2','P9C3','P9C4','P9C5','P9C6','P9C7','P9C8','P996','P9C10']})
sns.barplot(x='PC',y="var", data=df1, color="c");
            
'''
pc_df['P0C1'].std()
pc_df['P0C2'].std()
pc_df['P0C3'].std()
pc_df['P0C4'].std()
pc_df['P0C5'].std()


pc_df['P0C4'].describe()
'''

#########################

pca = PCA(n_components=5)
pc = pca.fit_transform(X1)
pc.shape

pc_df = pd.DataFrame(data = pc , columns = ['P0C1', 'P0C2','P0C3','P0C4','P0C5'])
pc_df.head()

pca.explained_variance_ratio_


"""variance explained by each principal component is called Scree plot.
"""
import seaborn as sns
df = pd.DataFrame({'var':pca.explained_variance_ratio_,
                  'PC':['P0C1', 'P0C2','P0C3','P0C4','P0C5']})
sns.barplot(x='PC',y="var", data=df, color="c");

#########################



