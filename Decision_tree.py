import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.model_selection  import train_test_split

features = list()  #存放特征
labels = list()    #存放分类标签

data=pd.read_csv('.\Iris_Data.csv')
ln=data.shape[0]
data_list=[]
for i in range(ln):
    temp=[]
    for n in data:
        if n == 'species':
            break
        a=data.loc[i,n]
        temp.append(a)
    data_list.append(temp)
    #print(temp)

label=[]
for i in range(ln):
    for n in data:
        if n == 'species':
            label.append(data.loc[i,n])
            break

feature_train,feature_test,label_train,label_test=train_test_split(data_list,label,test_size=0.3,shuffle='species')
#print(feature_train)
#print(feature_test)
#print(label_train)
#print(label_test)

DSTree = tree.DecisionTreeClassifier(criterion='entropy')
DSTree.fit(feature_train,label_train)
acc=np.mean(DSTree.predict(feature_test)==label_test)
print(acc)